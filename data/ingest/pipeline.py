import os
import math
import json
import torch
import torch.nn.functional as F

class AudioIngestPipeline:
    def __init__(self, target_sample_rate=48000, chunk_seconds=20, dedup_threshold=0.95):
        self.target_sample_rate = target_sample_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = int(target_sample_rate * chunk_seconds)
        self.dedup_threshold = dedup_threshold
        self.seen_embeddings = []

    def normalize_loudness(self, waveform, target_peak=0.95):
        peak = torch.max(torch.abs(waveform))
        if peak > 1e-6:
            waveform = waveform * (target_peak / peak)
        return waveform

    def chunk_audio(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).repeat(2, 1)
        total_samples = waveform.shape[-1]
        chunks = []
        for start in range(0, total_samples, self.chunk_samples):
            end = start + self.chunk_samples
            if end <= total_samples:
                chunks.append(waveform[:, start:end])
            elif total_samples - start >= int(self.chunk_samples * 0.5):
                pad_len = self.chunk_samples - (total_samples - start)
                chunk = F.pad(waveform[:, start:], (0, pad_len))
                chunks.append(chunk)
        return chunks

    def compute_embedding(self, chunk):
        gen = torch.Generator(device=chunk.device)
        energy_seed = int((torch.sum(torch.abs(chunk)) * 1000).item()) % 1000003
        gen.manual_seed(energy_seed)
        emb = torch.randn(512, generator=gen, device=chunk.device)
        return F.normalize(emb, p=2, dim=0)

    def is_duplicate(self, embedding):
        if len(self.seen_embeddings) == 0:
            self.seen_embeddings.append(embedding)
            return False
        stack = torch.stack(self.seen_embeddings)
        sims = torch.mv(stack, embedding)
        max_sim = torch.max(sims).item()
        if max_sim >= self.dedup_threshold:
            return True
        self.seen_embeddings.append(embedding)
        return False

    def quality_filter(self, chunk, min_snr_db=12.0):
        power = torch.mean(chunk ** 2).item()
        if power < 1e-5:
            return False
        peak = torch.max(torch.abs(chunk)).item()
        if peak >= 0.999:
            return False
        return True

    def process_track(self, waveform, track_id, license_id):
        norm = self.normalize_loudness(waveform)
        chunks = self.chunk_audio(norm)
        valid_shards = []

        for idx, chunk in enumerate(chunks):
            if not self.quality_filter(chunk):
                continue
            emb = self.compute_embedding(chunk)
            if self.is_duplicate(emb):
                continue

            shard_item = {
                "track_id": track_id,
                "chunk_idx": idx,
                "license_id": license_id,
                "audio": chunk,
                "embedding": emb,
                "duration": self.chunk_seconds,
            }
            valid_shards.append(shard_item)

        return valid_shards
