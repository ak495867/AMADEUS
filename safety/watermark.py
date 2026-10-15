import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioSealWatermarker(nn.Module):
    def __init__(self, key_bits=32, sample_rate=48000):
        super().__init__()
        self.key_bits = key_bits
        self.sample_rate = sample_rate
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=15, padding=7),
            nn.Tanh()
        )
        self.detector = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=31, stride=4, padding=15),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=31, stride=4, padding=15),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, key_bits),
            nn.Sigmoid()
        )

    def generate_watermark_id(self, prompt, timestamp_str):
        h = hashlib.sha256(f"{prompt}_{timestamp_str}".encode()).hexdigest()
        return f"amadeus-seal-{h[:16]}"

    def embed_watermark(self, audio, watermark_scale=0.003):
        if audio.dim() == 2:
            audio_in = audio.unsqueeze(0)
        else:
            audio_in = audio
        noise_carrier = self.encoder(audio_in)
        watermarked = audio_in + watermark_scale * noise_carrier
        watermarked = torch.clamp(watermarked, -1.0, 1.0)
        return watermarked.squeeze(0) if audio.dim() == 2 else watermarked

    def detect_watermark(self, audio):
        if audio.dim() == 2:
            audio_in = audio.unsqueeze(0)
        else:
            audio_in = audio
        preds = self.detector(audio_in)
        detected = (preds > 0.5).int()
        confidence = float(torch.mean(torch.abs(preds - 0.5) * 2.0).item())
        return {
            "watermark_detected": confidence > 0.4,
            "confidence": round(confidence, 4),
            "payload_bits": detected.tolist(),
        }
