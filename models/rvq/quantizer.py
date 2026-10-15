import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantize(nn.Module):
    def __init__(self, codebook_size=2048, dim=256, commitment_weight=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_weight = commitment_weight
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, x):
        flat_x = x.reshape(-1, self.dim)
        d = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_x, self.embedding.weight.t())
        )
        indices = torch.argmin(d, dim=1)
        quantized = self.embedding(indices).view_as(x)

        commit_loss = F.mse_loss(quantized.detach(), x) * self.commitment_weight
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = commit_loss + codebook_loss

        quantized = x + (quantized - x).detach()
        indices = indices.view(x.shape[:-1])
        return quantized, indices, loss

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_codebooks=4, codebook_size=2048, dim=256, commitment_weight=0.25):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.dim = dim
        self.layers = nn.ModuleList([
            VectorQuantize(codebook_size, dim, commitment_weight)
            for _ in range(num_codebooks)
        ])

    def forward(self, x):
        residual = x
        quantized_out = 0
        all_indices = []
        total_loss = 0.0

        for layer in self.layers:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            total_loss = total_loss + loss

        indices_tensor = torch.stack(all_indices, dim=-1)
        return quantized_out, indices_tensor, total_loss

    def decode_indices(self, indices):
        b, t, k = indices.shape
        quantized = 0
        for i in range(k):
            codebook = self.layers[i].embedding
            q = codebook(indices[:, :, i])
            quantized = quantized + q
        return quantized

    @staticmethod
    def apply_delay_pattern(tokens, delay_steps=1):
        b, t, k = tokens.shape
        padded = []
        for i in range(k):
            shift = i * delay_steps
            col = tokens[:, :, i]
            if shift > 0:
                pad = torch.zeros((b, shift), dtype=tokens.dtype, device=tokens.device)
                shifted = torch.cat([pad, col], dim=1)
            else:
                shifted = col
            padded.append(shifted)

        max_len = max(p.shape[1] for p in padded)
        aligned = []
        for p in padded:
            if p.shape[1] < max_len:
                p = F.pad(p, (0, max_len - p.shape[1]))
            aligned.append(p)

        return torch.stack(aligned, dim=-1)

    @staticmethod
    def revert_delay_pattern(tokens, orig_length, delay_steps=1):
        b, total_t, k = tokens.shape
        unaligned = []
        for i in range(k):
            shift = i * delay_steps
            col = tokens[:, shift:shift + orig_length, i]
            if col.shape[1] < orig_length:
                col = F.pad(col, (0, orig_length - col.shape[1]))
            unaligned.append(col)
        return torch.stack(unaligned, dim=-1)
