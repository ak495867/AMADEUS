import argparse
import sys
import torch
from amadeus.pipeline.inference import AMADEUSPipeline
from amadeus.eval.metrics import EvaluationSuite

def main():
    parser = argparse.ArgumentParser(description="AMADEUS CLI Engine")
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--text", type=str, required=True)
    gen_parser.add_argument("--duration", type=int, default=10)
    gen_parser.add_argument("--steps", type=int, default=8)
    gen_parser.add_argument("--out", type=str, default="output.raw")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--samples", type=int, default=3)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "generate":
        pipe = AMADEUSPipeline()
        res = pipe.generate(
            text_prompt=args.text,
            controls={"duration_s": args.duration, "steps": args.steps}
        )
        audio = res["audio"].cpu().numpy()
        with open(args.out, "wb") as f:
            f.write(audio.tobytes())
        print(f"Generated track saved to {args.out} (Job ID: {res['job_id']})")
        print(f"Watermark: {res['watermark_id']}")

    elif args.command == "eval":
        pipe = AMADEUSPipeline()
        suite = EvaluationSuite()
        results = suite.run_full_evaluation(pipe, test_samples=args.samples)
        print("AMADEUS Evaluation Suite Report:")
        print(f"Mean Latency: {results['mean_latency_ms']} ms")
        print(f"Mean STFT Distance: {results['mean_stft_distance']}")
        print(f"Mean Style Cosine: {results['mean_style_cosine']}")
        print(f"Targets Met: {results['targets_met']}")

    elif args.command == "serve":
        import uvicorn
        from amadeus.serving.api import app
        uvicorn.run(app, host=args.host, port=args.port)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
