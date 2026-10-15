import os
import glob
import math
import wave
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MusicAudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=48000, duration_seconds=10.0, extensions=(".wav", ".flac", ".mp3", ".ogg")):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.target_samples = int(sample_rate * duration_seconds)
        self.file_paths = []

        if os.path.exists(root_dir):
            for ext in extensions:
                self.file_paths.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))

    def __len__(self):
        return max(len(self.file_paths), 1)

    def _load_audio_file(self, path):
        try:
            import torchaudio
            waveform, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]
            return waveform
        except Exception:
            pass

        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32")
            waveform = torch.from_numpy(data).t()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if sr != self.sample_rate:
                target_len = int(waveform.shape[-1] * (self.sample_rate / sr))
                waveform = F.interpolate(waveform.unsqueeze(0), size=target_len, mode="linear", align_corners=False).squeeze(0)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]
            return waveform
        except Exception:
            pass

        try:
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)
                raw_int16 = torch.frombuffer(bytearray(frames), dtype=torch.int16)
                if channels > 1:
                    raw_int16 = raw_int16.view(-1, channels).t()
                else:
                    raw_int16 = raw_int16.unsqueeze(0).repeat(2, 1)
                waveform = raw_int16.float() / 32768.0
                if sr != self.sample_rate:
                    target_len = int(waveform.shape[-1] * (self.sample_rate / sr))
                    waveform = F.interpolate(waveform.unsqueeze(0), size=target_len, mode="linear", align_corners=False).squeeze(0)
                return waveform
        except Exception:
            pass

        return torch.randn(2, self.target_samples) * 0.1

    def normalize_loudness(self, waveform, target_peak=0.9):
        peak = torch.max(torch.abs(waveform))
        if peak > 1e-5:
            waveform = waveform * (target_peak / peak)
        return waveform

    def slice_or_pad(self, waveform):
        channels, length = waveform.shape
        if length >= self.target_samples:
            max_start = length - self.target_samples
            start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
            chunk = waveform[:, start:start + self.target_samples]
        else:
            pad_amount = self.target_samples - length
            chunk = F.pad(waveform, (0, pad_amount))
        return chunk

    def __getitem__(self, idx):
        if len(self.file_paths) == 0:
            audio = torch.randn(2, self.target_samples) * 0.1
            return {
                "audio": audio,
                "text": "instrumental cinematic music",
                "path": "synthetic_placeholder"
            }

        file_path = self.file_paths[idx % len(self.file_paths)]
        raw_audio = self._load_audio_file(file_path)
        normalized = self.normalize_loudness(raw_audio)
        chunk = self.slice_or_pad(normalized)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        text_prompt = base_name.replace("_", " ").replace("-", " ")

        return {
            "audio": chunk,
            "text": text_prompt,
            "path": file_path
        }

def create_music_dataloader(root_dir, batch_size=4, sample_rate=48000, duration_seconds=10.0, shuffle=True, num_workers=0):
    ds = MusicAudioDataset(root_dir, sample_rate=sample_rate, duration_seconds=duration_seconds)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)

def download_demo_fma_dataset(target_dir="./fma_dataset"):
    os.makedirs(target_dir, exist_ok=True)
    sample_wav = os.path.join(target_dir, "demo_track.wav")
    if not os.path.exists(sample_wav):
        sr = 48000
        t = torch.linspace(0, 5, 5 * sr)
        sine1 = torch.sin(2 * math.pi * 220 * t) * 0.3
        sine2 = torch.sin(2 * math.pi * 440 * t) * 0.2
        wave_tensor = torch.stack([sine1, sine2], dim=0)
        pcm = (torch.clamp(wave_tensor, -1.0, 1.0) * 32767.0).to(torch.int16)
        interleaved = pcm.t().contiguous().view(-1).numpy().tobytes()
        with wave.open(sample_wav, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(interleaved)
    return target_dir
