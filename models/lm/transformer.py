import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conditioning import AdaLN, MelodyEncoder, RhythmEncoder, TextConditioningEncoder

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, :q.shape[-1]]
        sin = self.sin_cached[:, :, :seq_len, :q.shape[-1]]
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot

class MultiCodebookEmbedding(nn.Module):
    def __init__(self, num_codebooks=4, codebook_size=2048, hidden_size=1536):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size + 1, hidden_size)
            for _ in range(num_codebooks)
        ])

    def forward(self, tokens):
        emb_sum = 0
        for i in range(self.num_codebooks):
            emb_sum = emb_sum + self.embeddings[i](tokens[..., i])
        return emb_sum

class LMTransformerBlock(nn.Module):
    def __init__(self, hidden_size=1536, num_heads=16, style_dim=512, cross_dim=768, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.adaln1 = AdaLN(hidden_size, style_dim)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.ln_cross = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_proj = nn.Linear(cross_dim, hidden_size)

        self.adaln2 = AdaLN(hidden_size, style_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, rope, style_vec, cross_cond=None, causal_mask=None):
        norm_x = self.adaln1(x, style_vec)
        b, seq_len, _ = norm_x.shape

        q = self.q_proj(norm_x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            scores = scores + causal_mask[:seq_len, :seq_len]
        probs = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(b, seq_len, self.hidden_size)
        x = x + self.out_proj(attn_out)

        if cross_cond is not None:
            cond_proj = self.cross_proj(cross_cond)
            c_out, _ = self.cross_attn(self.ln_cross(x), cond_proj, cond_proj)
            x = x + c_out

        x = x + self.mlp(self.adaln2(x, style_vec))
        return x

class SemanticTokenLM(nn.Module):
    def __init__(
        self,
        vocab_size=2048,
        num_codebooks=4,
        hidden_size=1536,
        num_layers=24,
        num_heads=16,
        max_seq_len=8192,
        style_dim=512,
        cross_dim=768,
        text_dim=768,
        dropout=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.token_embeddings = MultiCodebookEmbedding(num_codebooks, vocab_size, hidden_size)
        self.rope = RotaryEmbedding(hidden_size // num_heads, max_seq_len)

        self.text_proj = nn.Linear(text_dim, cross_dim)
        self.melody_encoder = MelodyEncoder(in_f0_dim=1, in_chroma_dim=12, out_dim=cross_dim)
        self.rhythm_encoder = RhythmEncoder(in_dim=2, out_dim=cross_dim)

        self.layers = nn.ModuleList([
            LMTransformerBlock(hidden_size, num_heads, style_dim, cross_dim, dropout)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(hidden_size)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_codebooks)
        ])

    def build_causal_mask(self, seq_len, device):
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, tokens, style_vec, text_emb=None, melody_f0=None, melody_chroma=None, beat_grid=None, condition_dropout=0.0):
        b, seq_len, _ = tokens.shape
        device = tokens.device

        cond_parts = []
        if text_emb is not None and (not self.training or torch.rand(1).item() > condition_dropout):
            if text_emb.dim() == 2:
                text_emb = text_emb.unsqueeze(1)
            cond_parts.append(self.text_proj(text_emb))

        if melody_f0 is not None and melody_chroma is not None and (not self.training or torch.rand(1).item() > condition_dropout):
            mel_emb = self.melody_encoder(melody_f0, melody_chroma)
            cond_parts.append(mel_emb)

        if beat_grid is not None and (not self.training or torch.rand(1).item() > condition_dropout):
            rhythm_emb = self.rhythm_encoder(beat_grid)
            cond_parts.append(rhythm_emb)

        cross_cond = torch.cat(cond_parts, dim=1) if len(cond_parts) > 0 else None

        if self.training and torch.rand(1).item() < condition_dropout:
            style_vec = torch.zeros_like(style_vec)

        x = self.token_embeddings(tokens)
        causal_mask = self.build_causal_mask(seq_len, device)

        for layer in self.layers:
            x = layer(x, self.rope, style_vec, cross_cond=cross_cond, causal_mask=causal_mask)

        x = self.final_ln(x)
        logits = torch.stack([head(x) for head in self.heads], dim=-2)
        return logits

    @torch.no_grad()
    def sample(self, prefix_tokens, style_vec, text_emb=None, melody_f0=None, melody_chroma=None, beat_grid=None, max_new_tokens=500, cfg=3.0, temperature=0.85):
        curr_tokens = prefix_tokens
        b = prefix_tokens.shape[0]

        for _ in range(max_new_tokens):
            if curr_tokens.shape[1] >= self.max_seq_len:
                break

            cond_logits = self.forward(curr_tokens, style_vec, text_emb=text_emb, melody_f0=melody_f0, melody_chroma=melody_chroma, beat_grid=beat_grid)
            uncond_logits = self.forward(curr_tokens, torch.zeros_like(style_vec), text_emb=None, melody_f0=None, melody_chroma=None, beat_grid=None)

            guided_logits = uncond_logits + cfg * (cond_logits - uncond_logits)
            last_step_logits = guided_logits[:, -1, :, :] / max(temperature, 1e-4)

            step_tokens = []
            for k in range(self.num_codebooks):
                probs = F.softmax(last_step_logits[:, k, :], dim=-1)
                tok = torch.multinomial(probs, num_samples=1)
                step_tokens.append(tok)

            next_token = torch.cat(step_tokens, dim=-1).unsqueeze(1)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)

        return curr_tokens
