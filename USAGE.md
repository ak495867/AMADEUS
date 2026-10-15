# AMADEUS Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quickstart (Python API)](#quickstart-python-api)
3. [Zero-Shot Style Transfer](#zero-shot-style-transfer)
4. [Multi-Stem Separation](#multi-stem-separation)
5. [Model Training](#model-training)
6. [Evaluation Suite](#evaluation-suite)
7. [HTTP & WebSocket Serving](#http--websocket-serving)
8. [DAW Plugin Bridge](#daw-plugin-bridge)
9. [AudioSeal Watermarking](#audioseal-watermarking)

---

## Installation

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA 11.8+ or 12.0+ (CPU fallback supported)
- FFmpeg (for MP3/FLAC encoding)

### Clone & Install

```bash
git clone https://github.com/ak495867/AMADEUS.git
cd AMADEUS
pip install -e .
```

### PyTorch GPU Check

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Quickstart (Python API)

Generate full 48 kHz stereo music from a natural language text prompt:

```python
import torch
import soundfile as sf
from amadeus.pipeline.inference import AMADEUSPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = AMADEUSPipeline(device=device, checkpoint_dir="./checkpoints")

result = pipeline.generate(
    text_prompt="lo-fi hip hop beat with warm rhodes electric piano and vinyl crackle",
    duration_s=15.0,
    temperature=0.8,
    style_strength=0.7
)

sf.write("output_mix.wav", result["audio"].squeeze(0).cpu().numpy().T, 48000)
```

---

## Zero-Shot Style Transfer

Transfer the acoustic timbre, melodic contour, and energy profile of an existing reference track onto a target text prompt:

```python
import torch
import soundfile as sf
from amadeus.pipeline.inference import AMADEUSPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = AMADEUSPipeline(device=device, checkpoint_dir="./checkpoints")

ref_audio, sr = sf.read("reference_track.wav")
ref_tensor = torch.from_numpy(ref_audio).float()
if ref_tensor.dim() == 1:
    ref_tensor = ref_tensor.unsqueeze(0).repeat(2, 1)
elif ref_tensor.shape[0] != 2:
    ref_tensor = ref_tensor.t()
ref_tensor = ref_tensor.unsqueeze(0).to(device)

result = pipeline.generate(
    reference_song=ref_tensor,
    text_prompt="cyberpunk dark synthwave with heavy analog bass",
    duration_s=20.0,
    style_strength=0.85
)

sf.write("transferred_track.wav", result["audio"].squeeze(0).cpu().numpy().T, 48000)
```

---

## Multi-Stem Separation

Render discrete multi-track stems (`vocals`, `drums`, `bass`, `other`) directly during the generative synthesis pass:

```python
import soundfile as sf
from amadeus.pipeline.inference import AMADEUSPipeline

pipeline = AMADEUSPipeline(device="cuda", checkpoint_dir="./checkpoints")

result = pipeline.generate(
    text_prompt="funk groove with slap bass and tight brass section",
    duration_s=10.0
)

stems = result["stems"]
for stem_name, stem_audio in stems.items():
    audio_np = stem_audio.squeeze(0).cpu().numpy().T
    sf.write(f"stem_{stem_name}.wav", audio_np, 48000)
```

---

## Model Training

AMADEUS provides modular scripts to train each stage individually or perform full end-to-end training.

### 1. Neural Codec Training
Trains the SEANet encoder, HiFi-GAN 4-stem generator, and multi-period discriminators:

```bash
python scripts/train_codec.py --dataset gtzan --batch_size 4 --epochs 10 --lr 0.0002 --checkpoint_dir ./checkpoints
```

Resume codec training from checkpoint:

```bash
python scripts/train_codec.py --dataset gtzan --batch_size 4 --epochs 20 --resume ./checkpoints/codec_epoch_5.pt
```

### 2. Semantic Token LM Training
Trains the causal transformer with T5 cross-attention and RVQ delay patterns:

```bash
python scripts/train_lm.py --dataset gtzan --batch_size 8 --epochs 15 --lr 0.0001 --checkpoint_dir ./checkpoints
```

### 3. Flow Matching DiT Training
Trains the continuous flow matching diffusion transformer on extracted latent codes:

```bash
python scripts/train_dit.py --dataset gtzan --batch_size 8 --epochs 25 --lr 0.0001 --checkpoint_dir ./checkpoints
```

### 4. End-to-End Multi-Stage Training
Sequentially trains Codec, LM, and Flow Matching DiT with automatic dataset downloading, mixed-precision AMP, gradient clipping, and cosine annealing:

```bash
python scripts/train_full.py --dataset gtzan --epochs 5 --batch_size 4 --lr 0.0002 --checkpoint_dir ./checkpoints
```

Train using FMA (Free Music Archive) dataset:

```bash
python scripts/train_full.py --dataset fma --fma_size small --epochs 10 --batch_size 4 --checkpoint_dir ./checkpoints
```

---

## Evaluation Suite

Run the automated evaluation benchmark to measure latency, STFT distance, style cosine alignment, Fréchet Audio Distance (FAD), and beat tracking F1:

```python
import torch
from amadeus.pipeline.inference import AMADEUSPipeline
from amadeus.eval.metrics import EvaluationSuite

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = AMADEUSPipeline(device=device, checkpoint_dir="./checkpoints")
suite = EvaluationSuite()

metrics = suite.run_full_evaluation(pipeline, test_samples=5)

print("Mean Latency (ms):", metrics["mean_latency_ms"])
print("Mean STFT Distance:", metrics["mean_stft_distance"])
print("Mean Style Cosine:", metrics["mean_style_cosine"])
print("FAD Score:", metrics["fad_score"])
print("Beat F1:", metrics["mean_beat_f1"])
print("Target Benchmarks Met:", metrics["targets_met"])
```

---

## HTTP & WebSocket Serving

Deploy AMADEUS as a high-throughput production API service with real-time audio chunk streaming.

### Start the Uvicorn Server

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --workers 1
```

### REST API Generation Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text_prompt": "cinematic ambient piano", "duration_s": 10.0, "style_strength": 0.8}' \
  --output generated.wav
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Attribution Query

```bash
curl http://localhost:8000/attribution/job-001
```

### WebSocket Streaming Client

```python
import asyncio
import websockets
import json

async def stream_audio():
    uri = "ws://localhost:8000/stream"
    async with websockets.connect(uri) as websocket:
        request_payload = {
            "text_prompt": "upbeat electronic disco house",
            "duration_s": 10.0
        }
        await websocket.send(json.dumps(request_payload))
        while True:
            chunk = await websocket.recv()
            if isinstance(chunk, bytes):
                print(f"Received audio packet: {len(chunk)} bytes")
            else:
                event = json.loads(chunk)
                if event.get("event") == "complete":
                    break

asyncio.run(stream_audio())
```

---

## DAW Plugin Bridge

AMADEUS includes a C++ JUCE plugin bridge located in `plugin/` allowing real-time parameter streaming, MIDI sync, and stem bouncing within DAWs (Ableton Live, Logic Pro, FL Studio, Reaper).

The plugin connects via local UNIX domain sockets or IPC named pipes to the local AMADEUS inference engine:

```bash
python sdk/client.py --port 8000 --interactive
```

---

## AudioSeal Watermarking

Embed and detect 16-bit cryptographic watermarks to ensure AI transparency and audio authenticity:

```python
import torch
import soundfile as sf
from amadeus.safety.watermark import AudioSealWatermarker

watermarker = AudioSealWatermarker()

audio_data, sr = sf.read("output_mix.wav")
audio_tensor = torch.from_numpy(audio_data).float().t().unsqueeze(0)

watermarked_audio = watermarker.embed(audio_tensor, payload=42)
sf.write("watermarked.wav", watermarked_audio.squeeze(0).cpu().numpy().T, sr)

test_audio, _ = sf.read("watermarked.wav")
test_tensor = torch.from_numpy(test_audio).float().t().unsqueeze(0)
is_detected, extracted_payload = watermarker.detect(test_tensor)

print(f"Watermark Detected: {is_detected}")
print(f"Extracted Payload ID: {extracted_payload}")
```
