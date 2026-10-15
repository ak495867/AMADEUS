# AMADEUS Model Card

## Model Details
- Architecture: Four-level coarse-to-fine generative cascade
  - Level 0: Structure Planner (150M parameter autoregressive transformer @ 1 Hz)
  - Level 1: Semantic Token LM (1.5B parameter transformer with 4-codebook delay pattern @ 25 Hz)
  - Level 2: Latent Diffusion Refiner (900M parameter DiT with Rectified Flow / Flow Matching @ 50 Hz)
  - Level 3: Neural Codec VAE (SEANet encoder, stride 960) and HiFi-GAN decoder with 4-stem head @ 48 kHz stereo
- Intended Use: Zero-shot style transfer soundtrack generation, text-to-music generation, interactive stems creation, and realtime background music streaming.
- Watermarking: Embedded AudioSeal imperceptible watermark with cryptographic verification.
- Attribution: Soft-prompt retrieval attribution engine with licensed catalog index.
