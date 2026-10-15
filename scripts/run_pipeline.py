import sys
import torch
from amadeus.pipeline.inference import AMADEUSPipeline
from amadeus.pipeline.longform import LongformMusicGenerator
from amadeus.pipeline.realtime import RealtimeBGMGenerator
from amadeus.eval.metrics import EvaluationSuite

def run_smoke_test():
    print("Initializing AMADEUS Pipeline...")
    pipeline = AMADEUSPipeline(device="cpu")

    print("\n--- Testing Single Generation ---")
    ref_audio = torch.randn(2, 48000 * 3) * 0.15
    res = pipeline.generate(
        reference_song=ref_audio,
        text_prompt="cyberpunk synthwave with driving drums and analog lead",
        controls={"steps": 4, "duration_s": 3, "stems": True}
    )

    print(f"Job ID: {res['job_id']}")
    print(f"Generated Audio Shape: {res['audio'].shape}")
    print(f"Watermark ID: {res['watermark_id']}")
    print(f"Key: {res['analysis']['key']} {res['analysis']['mode']}, BPM: {res['analysis']['bpm']}")
    print(f"Stems: {list(res['stems'].keys())}")
    print(f"Attributed Sources: {len(res['attribution']['attributed_sources'])}")

    print("\n--- Testing Longform Generation ---")
    longform = LongformMusicGenerator(pipeline, window_seconds=4, crossfade_seconds=1)
    lf_res = longform.generate_longform(total_seconds=6, text_prompt="ambient piano meditation")
    print(f"Longform Audio Shape: {lf_res['audio'].shape}, Duration: {lf_res['duration']}s")

    print("\n--- Testing Realtime Streaming ---")
    realtime = RealtimeBGMGenerator(pipeline, chunk_duration_s=1.0)
    for chunk in realtime.stream_bgm("retro arcade stage theme", total_chunks=2):
        print(f"Chunk {chunk['chunk_index']} Latency: {chunk['latency_ms']} ms, Shape: {chunk['audio'].shape}")

    print("\n--- Testing Evaluation Suite ---")
    eval_suite = EvaluationSuite()
    metrics = eval_suite.run_full_evaluation(pipeline, test_samples=2)
    print(f"Mean Latency: {metrics['mean_latency_ms']} ms")
    print(f"Mean STFT Distance: {metrics['mean_stft_distance']}")
    print(f"Mean Style Cosine: {metrics['mean_style_cosine']}")
    print(f"Targets Status: {metrics['targets_met']}")

    print("\nAMADEUS Verification Completed Successfully!")

if __name__ == "__main__":
    run_smoke_test()
