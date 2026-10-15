import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class AdaLNZero(nn.Module):
    def __init__(self, hidden_size, cond_dim):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 6 * hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond):
        params = self.linear(cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

class DiTBlock(nn.Module):
    def __init__(self, hidden_size=1152, num_heads=16, cond_dim=1024, cross_dim=768, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.ln_msa = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.ln_cross = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_proj = nn.Linear(cross_dim, hidden_size)

        self.ln_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.adaln = AdaLNZero(hidden_size, cond_dim)

    def forward(self, x, cond, cross_cond=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(cond)

        norm_x = (1 + scale_msa.unsqueeze(1)) * self.ln_msa(x) + shift_msa.unsqueeze(1)
        b, seq_len, c = norm_x.shape

        qkv = self.qkv_proj(norm_x).reshape(b, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(b, seq_len, c)
        msa_out = self.out_proj(attn_out)

        x = x + gate_msa.unsqueeze(1) * msa_out

        if cross_cond is not None:
            c_proj = self.cross_proj(cross_cond)
            c_out, _ = self.cross_attn(self.ln_cross(x), c_proj, c_proj)
            x = x + c_out

        norm_mlp = (1 + scale_mlp.unsqueeze(1)) * self.ln_mlp(x) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(norm_mlp)

        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        in_channels=64,
        hidden_size=1152,
        num_layers=28,
        num_heads=16,
        patch_size=4,
        style_dim=512,
        cross_dim=768,
        max_patches=4096,
        dropout=0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.patch_dim = in_channels * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, hidden_size))

        self.t_embed = TimestepEmbedding(256)
        cond_in_dim = 256 + style_dim + 1
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024)
        )

        self.semantic_proj = nn.Linear(256, cross_dim)
        self.text_proj = nn.Linear(768, cross_dim)
        self.melody_proj = nn.Linear(256, cross_dim)
        self.rhythm_proj = nn.Linear(64, cross_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, cond_dim=1024, cross_dim=cross_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Linear(1024, 2 * hidden_size)
        self.final_proj = nn.Linear(hidden_size, self.patch_dim)

    def forward(
        self,
        x,
        t,
        style=None,
        loudness=None,
        text=None,
        melody=None,
        rhythm=None,
        sem=None
    ):
        b, c, length = x.shape
        pad = (self.patch_size - (length % self.patch_size)) % self.patch_size
        if pad > 0:
            x = F.pad(x, (0, pad))

        num_patches = x.shape[-1] // self.patch_size
        x_patches = x.view(b, c, num_patches, self.patch_size).permute(0, 2, 1, 3).reshape(b, num_patches, self.patch_dim)
        h = self.patch_embed(x_patches) + self.pos_embed[:, :num_patches, :]

        t_emb = self.t_embed(t)
        if style is None:
            style = torch.zeros((b, 512), device=x.device)
        if loudness is None:
            loudness = torch.zeros((b, 1), device=x.device)
        elif loudness.dim() == 1:
            loudness = loudness.unsqueeze(-1)

        cond_input = torch.cat([t_emb, style, loudness], dim=-1)
        cond_vec = self.cond_mlp(cond_input)

        cross_tokens = []
        if text is not None:
            if text.dim() == 2:
                text = text.unsqueeze(1)
            cross_tokens.append(self.text_proj(text))
        if melody is not None:
            if melody.dim() == 2:
                melody = melody.unsqueeze(1)
            cross_tokens.append(self.melody_proj(melody))
        if rhythm is not None:
            if rhythm.dim() == 2:
                rhythm = rhythm.unsqueeze(1)
            cross_tokens.append(self.rhythm_proj(rhythm))
        if sem is not None:
            if sem.dim() == 3:
                sem_reduced = sem.float().mean(dim=-1).unsqueeze(-1).repeat(1, 1, 256)
            else:
                sem_reduced = sem.float()
            cross_tokens.append(self.semantic_proj(sem_reduced))

        cross_cond = torch.cat(cross_tokens, dim=1) if len(cross_tokens) > 0 else None

        for block in self.blocks:
            h = block(h, cond_vec, cross_cond)

        shift, scale = self.final_adaln(cond_vec).chunk(2, dim=-1)
        h = (1 + scale.unsqueeze(1)) * self.final_ln(h) + shift.unsqueeze(1)
        out_patches = self.final_proj(h)

        v_pred = out_patches.view(b, num_patches, c, self.patch_size).permute(0, 2, 1, 3).reshape(b, c, num_patches * self.patch_size)
        if pad > 0:
            v_pred = v_pred[:, :, :length]
        return v_pred
