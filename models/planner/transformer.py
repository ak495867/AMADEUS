import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class StructurePlanner(nn.Module):
    def __init__(self, vocab_size=4096, hidden_size=768, num_layers=12, num_heads=12, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        b, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).repeat(b, 1)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_ln(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=64, temperature=0.8, top_k=50, top_p=0.9):
        curr_ids = prompt_ids
        for _ in range(max_new_tokens):
            if curr_ids.shape[1] >= self.max_seq_len:
                break
            logits = self.forward(curr_ids)[:, -1, :]
            logits = logits / max(temperature, 1e-5)

            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_value = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_value, torch.full_like(logits, -float("inf")), logits)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                for b_idx in range(curr_ids.size(0)):
                    indices_to_remove = sorted_indices[b_idx, sorted_indices_to_remove[b_idx]]
                    logits[b_idx, indices_to_remove] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

        return curr_ids
