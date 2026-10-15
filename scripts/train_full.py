import argparse
import os
import wave
import torch
from amadeus.data.dataset import download_real_music_dataset, download_demo_fma_dataset
from amadeus.scripts.train_codec import train_codec
from amadeus.scripts.train_dit import train_dit
from amadeus.scripts.train_lm import train_lm
from amadeus.pipeline.inference import AMADEUSPipeline
from amadeus.eval.metrics import EvaluationSuite

def save_audio_wav(filepath, tensor_audio, sample_rate=48000):
    audio_clamped = torch.clamp(tensor_audio.detach(), -1.0, 1.0)
    pcm = (audio_clamped * 32767.0).to(torch.int16).cpu()
    channels = pcm.shape[0]
    interleaved = pcm.t().contiguous().view(-1).numpy().tobytes()
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved)

def main():
    parser = argparse.ArgumentParser(description="AMADEUS Full Model Training Engine")
    parser.add_argument("--data_dir", type=str, default="./real_music_dataset")
    parser.add_argument("--dataset", type=str, default="gtzan", choices=["gtzan", "fma_small", "custom"])
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "codec", "dit", "lm"])
    parser.add_argument("--epochs_codec", type=int, default=10)
    parser.add_argument("--epochs_dit", type=int, default=10)
    parser.add_argument("--epochs_lm", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr_codec", type=float, default=2e-4)
    parser.add_argument("--lr_dit", type=float, default=1e-4)
    parser.add_argument("--lr_lm", type=float, default=1.5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output_audio", type=str, default="full_train_sample.wav")
    args = parser.parse_args()

    print("==================================================================")
    print(f"AMADEUS Production Training Engine | Device: {args.device.upper()}")
    print(f"Target Checkpoints Directory: {os.path.abspath(args.checkpoints_dir)}")
    print("==================================================================")

    os.makedirs(args.checkpoints_dir, exist_ok=True)

    if args.dataset != "custom":
        download_real_music_dataset(args.data_dir, dataset_name=args.dataset)

    if not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0:
        download_demo_fma_dataset(args.data_dir)

    if args.stage in ["all", "codec"]:
        print("\n>>> STAGE 1: Training Neural Audio Codec VAE (HiFi-GAN & SEANet)...")
        train_codec(
            data_dir=args.data_dir,
            epochs=args.epochs_codec,
            batch_size=args.batch_size,
            lr=args.lr_codec,
            save_dir=args.checkpoints_dir,
            device=args.device,
            compile_model=args.compile,
            grad_accum=args.grad_accum,
            resume=args.resume
        )

    if args.stage in ["all", "dit"]:
        print("\n>>> STAGE 2: Training Flow Matching Diffusion Transformer (DiT)...")
        train_dit(
            data_dir=args.data_dir,
            epochs=args.epochs_dit,
            batch_size=args.batch_size,
            lr=args.lr_dit,
            save_dir=args.checkpoints_dir,
            device=args.device,
            compile_model=args.compile,
            grad_accum=args.grad_accum,
            resume=args.resume
        )

    if args.stage in ["all", "lm"]:
        print("\n>>> STAGE 3: Training Autoregressive Semantic Token LM...")
        train_lm(
            data_dir=args.data_dir,
            epochs=args.epochs_lm,
            batch_size=args.batch_size,
            lr=args.lr_lm,
            save_dir=args.checkpoints_dir,
            device=args.device,
            compile_model=args.compile,
            grad_accum=args.grad_accum,
            resume=args.resume
        )

    print("\n>>> STAGE 4: Running Comprehensive Model Evaluation...")
    pipe = AMADEUSPipeline(device=args.device, checkpoint_dir=args.checkpoints_dir)
    suite = EvaluationSuite()
    metrics = suite.run_full_evaluation(pipe, test_samples=3)
    print("Full Model Evaluation Results:")
    print(f"- Mean Latency:       {metrics['mean_latency_ms']} ms")
    print(f"- Mean STFT Distance: {metrics['mean_stft_distance']}")
    print(f"- FAD Score:          {metrics['fad_score']}")
    print(f"- Mean Beat F1:       {metrics['mean_beat_f1']}")
    print(f"- Target Standards:   {metrics['targets_met']}")

    print(f"\n>>> STAGE 5: Synthesizing Full Audio Sample -> {args.output_audio}")
    res = pipe.generate(
        text_prompt="epic orchestral hybrid cinematic soundtrack with punchy percussion and warm analog synth",
        controls={"steps": 8, "duration_s": 8, "stems": True}
    )
    save_audio_wav(args.output_audio, res["audio"], sample_rate=48000)
    print(f"Sample generated successfully! Job ID: {res['job_id']}")
    print("All training stages completed successfully.")

if __name__ == "__main__":
    main()
