import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.ln1 = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, past_kv=None):
        norm_x = self.ln1(x)
        b, seq_len, _ = norm_x.shape

        q = self.q_proj(norm_x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v)

        if past_kv is not None:
            attn_out = F.scaled_dot_product_attention(q, k, v)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, seq_len, self.hidden_size)
        x = x + self.out_proj(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x, present_kv

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

    def forward(self, input_ids, past_kvs=None, offset=0):
        b, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(offset, offset + seq_len, device=device).unsqueeze(0).repeat(b, 1)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        new_kvs = []
        for i, layer in enumerate(self.layers):
            layer_past = past_kvs[i] if past_kvs is not None else None
            x, pres_kv = layer(x, past_kv=layer_past)
            new_kvs.append(pres_kv)

        x = self.final_ln(x)
        logits = self.head(x)
        return logits, new_kvs

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=64, temperature=0.8, top_k=50, top_p=0.9):
        curr_ids = prompt_ids
        logits, past_kvs = self.forward(prompt_ids, past_kvs=None, offset=0)
        offset = prompt_ids.shape[1]

        for _ in range(max_new_tokens):
            if offset >= self.max_seq_len:
                break

            last_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                values, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                min_value = values[:, -1].unsqueeze(-1)
                last_logits = torch.where(last_logits < min_value, torch.full_like(last_logits, -float("inf")), last_logits)

            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            logits, past_kvs = self.forward(next_token, past_kvs=past_kvs, offset=offset)
            offset += 1

        return curr_ids
