import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLN(nn.Module):
    def __init__(self, hidden_size, cond_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, hidden_size * 2)

    def forward(self, x, cond):
        params = self.linear(cond)
        if params.dim() == 2:
            params = params.unsqueeze(1)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        return (1 + gamma) * self.ln(x) + beta

class MelodyEncoder(nn.Module):
    def __init__(self, in_f0_dim=1, in_chroma_dim=12, out_dim=256):
        super().__init__()
        in_dim = in_f0_dim + in_chroma_dim
        self.conv_in = nn.Conv1d(in_dim, 128, kernel_size=5, padding=2)
        self.res_blocks = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, out_dim, kernel_size=3, padding=1),
        )
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, f0, chroma):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        if chroma.dim() == 2:
            chroma = chroma.unsqueeze(0)
        if chroma.shape[0] != f0.shape[0]:
            chroma = chroma.repeat(f0.shape[0], 1, 1)
        if chroma.shape[-1] != f0.shape[-1]:
            chroma = F.interpolate(chroma, size=f0.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([f0, chroma], dim=1)
        h = self.conv_in(x)
        h = self.res_blocks(h)
        h = h.transpose(1, 2)
        return self.proj(h)

class RhythmEncoder(nn.Module):
    def __init__(self, in_dim=2, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, beat_grid_tensor):
        if beat_grid_tensor.dim() == 2:
            beat_grid_tensor = beat_grid_tensor.unsqueeze(0)
        return self.mlp(beat_grid_tensor)

class TextConditioningEncoder(nn.Module):
    def __init__(self, in_dim=768, out_dim=768, pretrained_name="t5-small"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pretrained_name = pretrained_name
        self.tokenizer = None
        self.encoder_model = None
        self.cache = {}
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        self.fallback_emb = nn.Embedding(256, in_dim)

    def _init_hf(self, device):
        if self.tokenizer is not None and self.encoder_model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_name)
            self.encoder_model = AutoModel.from_pretrained(self.pretrained_name).encoder.to(device)
            self.encoder_model.eval()
            for p in self.encoder_model.parameters():
                p.requires_grad = False
        except Exception:
            self.tokenizer = False
            self.encoder_model = False

    def forward(self, text_or_emb, device="cpu"):
        if isinstance(text_or_emb, torch.Tensor):
            return self.proj(text_or_emb)

        if isinstance(text_or_emb, str):
            if text_or_emb in self.cache:
                return self.cache[text_or_emb]
            texts = [text_or_emb]
            is_single = True
        else:
            texts = text_or_emb
            is_single = False

        self._init_hf(device)
        if self.tokenizer and self.encoder_model:
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.encoder_model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1)
                if emb.shape[-1] != self.in_dim:
                    emb = F.pad(emb, (0, self.in_dim - emb.shape[-1])) if emb.shape[-1] < self.in_dim else emb[:, :self.in_dim]
            out = self.proj(emb)
            if is_single:
                self.cache[text_or_emb] = out
            return out

        batch_chars = []
        max_len = 0
        for text in texts:
            b = [min(ord(c), 255) for c in text[:128]]
            if len(b) == 0:
                b = [0]
            max_len = max(max_len, len(b))
            batch_chars.append(b)

        padded = []
        for b in batch_chars:
            padded.append(b + [0] * (max_len - len(b)))

        idx_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        char_embs = self.fallback_emb(idx_tensor).mean(dim=1)
        out = self.proj(char_embs)
        if is_single:
            self.cache[text_or_emb] = out
        return out
