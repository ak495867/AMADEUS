import time
import torch
import torch.nn.functional as F

class MusicRetrievalIndex:
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.catalog = []
        self.embeddings = None

    def add_tracks(self, track_records):
        new_embs = []
        for r in track_records:
            emb = r["embedding"]
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            emb = F.normalize(emb, p=2, dim=-1)
            new_embs.append(emb)
            self.catalog.append({
                "track_id": r["track_id"],
                "artist": r.get("artist", "Licensed Artist"),
                "title": r.get("title", "Untitled Catalog Track"),
                "license_id": r.get("license_id", "COMMERCIAL-V1"),
                "genre": r.get("genre", "Soundtrack"),
                "bpm": r.get("bpm", 120.0),
                "key": r.get("key", "C"),
            })

        stacked = torch.cat(new_embs, dim=0)
        if self.embeddings is None:
            self.embeddings = stacked
        else:
            self.embeddings = torch.cat([self.embeddings, stacked], dim=0)

    def search(self, query_emb, k=3):
        if self.embeddings is None or len(self.catalog) == 0:
            return [], torch.zeros((1, 512), device=query_emb.device if hasattr(query_emb, "device") else "cpu")

        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        query_emb = F.normalize(query_emb, p=2, dim=-1)

        sims = torch.mm(query_emb, self.embeddings.t())
        k = min(k, len(self.catalog))
        top_scores, top_indices = torch.topk(sims, k=k, dim=-1)

        matches = []
        for rank, (idx, score) in enumerate(zip(top_indices[0].tolist(), top_scores[0].tolist())):
            record = dict(self.catalog[idx])
            record["score"] = float(round(score, 4))
            record["rank"] = rank + 1
            matches.append(record)

        retrieved_embs = self.embeddings[top_indices[0]]
        weights = F.softmax(top_scores[0], dim=-1).unsqueeze(-1)
        soft_prompt = torch.sum(retrieved_embs * weights, dim=0, keepdim=True)
        if soft_prompt.shape[-1] > 512:
            soft_prompt = soft_prompt[:, :512]
        elif soft_prompt.shape[-1] < 512:
            soft_prompt = F.pad(soft_prompt, (0, 512 - soft_prompt.shape[-1]))

        return matches, soft_prompt

class AttributionEngine:
    def __init__(self, index):
        self.index = index
        self.logs = []

    def log_generation(self, job_id, retrieved_matches, prompt, duration):
        record = {
            "job_id": job_id,
            "timestamp": time.time(),
            "prompt": prompt,
            "duration": duration,
            "attributed_sources": [
                {
                    "track_id": m["track_id"],
                    "artist": m["artist"],
                    "title": m["title"],
                    "license_id": m["license_id"],
                    "similarity": m["score"],
                }
                for m in retrieved_matches
            ],
        }
        self.logs.append(record)
        return record
