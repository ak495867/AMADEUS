import time
import torch

class RealtimeBGMGenerator:
    def __init__(self, pipeline, chunk_duration_s=2.0):
        self.pipeline = pipeline
        self.chunk_duration_s = chunk_duration_s
        self.latent_frames = int(chunk_duration_s * 50)
        self.lookahead_tokens = int(chunk_duration_s * 25)

    def stream_bgm(self, text_prompt="dynamic game background music", total_chunks=10):
        controls = {
            "steps": 4,
            "duration_s": int(self.chunk_duration_s),
            "text_cfg": 2.0,
            "stems": False
        }

        for chunk_idx in range(total_chunks):
            t_start = time.time()
            result = self.pipeline.generate(
                text_prompt=f"{text_prompt} part {chunk_idx + 1}",
                controls=controls
            )
            t_elapsed = (time.time() - t_start) * 1000.0

            yield {
                "chunk_index": chunk_idx,
                "audio": result["audio"],
                "latency_ms": round(t_elapsed, 2),
                "is_last": (chunk_idx == total_chunks - 1)
            }
