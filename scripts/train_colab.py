import argparse
import os
import wave
import torch
from amadeus.data.dataset import download_demo_fma_dataset
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./fma_dataset")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs_codec", type=int, default=1)
    parser.add_argument("--epochs_dit", type=int, default=1)
    parser.add_argument("--epochs_lm", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_audio", type=str, default="generated_output.wav")
    args = parser.parse_args()

    print(f"Running AMADEUS Colab Engine on Device: {args.device}")
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    if not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0:
        print("Preparing demo audio dataset...")
        download_demo_fma_dataset(args.data_dir)

    print("Step 1/3: Training Neural Codec VAE...")
    train_codec(
        data_dir=args.data_dir,
        epochs=args.epochs_codec,
        batch_size=args.batch_size,
        save_dir=args.checkpoints_dir,
        device=args.device
    )

    print("Step 2/3: Training Flow Matching Diffusion Transformer (DiT)...")
    train_dit(
        data_dir=args.data_dir,
        epochs=args.epochs_dit,
        batch_size=args.batch_size,
        save_dir=args.checkpoints_dir,
        device=args.device
    )

    print("Step 3/3: Training Semantic Token LM...")
    train_lm(
        data_dir=args.data_dir,
        epochs=args.epochs_lm,
        batch_size=args.batch_size,
        save_dir=args.checkpoints_dir,
        device=args.device
    )

    print("Running Model Evaluation Suite...")
    pipe = AMADEUSPipeline(device=args.device, checkpoint_dir=args.checkpoints_dir)
    suite = EvaluationSuite()
    eval_res = suite.run_full_evaluation(pipe, test_samples=2)
    print("Evaluation Results:", eval_res["targets_met"])

    print(f"Generating test music track -> {args.output_audio}")
    gen_result = pipe.generate(
        text_prompt="cyberpunk synthwave with driving bass and drums",
        controls={"steps": 4, "duration_s": 4, "stems": True}
    )

    save_audio_wav(args.output_audio, gen_result["audio"], sample_rate=48000)
    print(f"Track generated successfully: Job ID {gen_result['job_id']}")

if __name__ == "__main__":
    main()
