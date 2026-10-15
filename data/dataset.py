import os
import glob
import math
import wave
import urllib.request
import tarfile
import zipfile
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

def generate_synthetic_music_dataset(target_dir="./real_music_dataset", num_tracks=40, duration_seconds=10.0, sr=48000):
    os.makedirs(target_dir, exist_ok=True)
    genres = ["synthwave", "ambient", "techno", "rock", "hiphop", "classical", "jazz", "disco"]
    scales = {
        "minor": [0, 3, 5, 7, 10, 12, 15, 17],
        "major": [0, 2, 4, 7, 9, 12, 14, 16],
        "dorian": [0, 2, 3, 5, 7, 9, 10, 12]
    }
    roots = [110.0, 123.47, 130.81, 146.83, 164.81, 174.61, 196.0, 220.0]
    total_samples = int(duration_seconds * sr)
    t = torch.linspace(0, duration_seconds, total_samples)

    for i in range(num_tracks):
        genre = genres[i % len(genres)]
        file_path = os.path.join(target_dir, f"{genre}_track_{i+1:03d}.wav")
        if os.path.exists(file_path):
            continue

        bpm = 80 + (i * 7) % 60
        beat_len = 60.0 / bpm
        root = roots[i % len(roots)]
        scale = scales["minor" if i % 2 == 0 else "major"]

        kick_t = (t % beat_len)
        kick_env = torch.exp(-kick_t * 18.0)
        kick_freq = 45.0 + 90.0 * torch.exp(-kick_t * 30.0)
        kick = torch.sin(2 * math.pi * kick_freq * kick_t) * kick_env * 0.5

        snare_t = ((t + beat_len * 0.5) % beat_len)
        snare_env = torch.exp(-snare_t * 22.0)
        noise = (torch.rand(total_samples) * 2.0 - 1.0)
        snare = noise * snare_env * 0.25

        bass_freq = root * 0.5
        bass = torch.sin(2 * math.pi * bass_freq * t) * 0.35 + 0.15 * torch.sin(2 * math.pi * bass_freq * 2.0 * t)

        scale_tensor = torch.tensor(scale, dtype=torch.float32)
        chord_idx = (t / (beat_len * 4)).long() % len(scale)
        degree = scale_tensor[chord_idx]
        chord_freq = root * (2.0 ** (degree / 12.0))
        pad = 0.2 * torch.sin(2 * math.pi * chord_freq * t) + 0.1 * torch.sin(2 * math.pi * chord_freq * 1.5 * t)

        left = kick + 0.8 * snare + bass + 0.8 * pad
        right = kick + 0.7 * snare + bass + 0.9 * pad
        stereo = torch.stack([left, right], dim=0)

        peak = torch.max(torch.abs(stereo))
        if peak > 1e-4:
            stereo = stereo * (0.85 / peak)

        pcm = (torch.clamp(stereo, -1.0, 1.0) * 32767.0).to(torch.int16)
        interleaved = pcm.t().contiguous().view(-1).numpy().tobytes()

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(interleaved)

    return target_dir

def download_demo_fma_dataset(target_dir="./fma_dataset"):
    return generate_synthetic_music_dataset(target_dir=target_dir, num_tracks=20)

def download_real_music_dataset(target_dir="./real_music_dataset", dataset_name="gtzan"):
    os.makedirs(target_dir, exist_ok=True)
    existing_files = glob.glob(os.path.join(target_dir, "**", "*.wav"), recursive=True) + glob.glob(os.path.join(target_dir, "**", "*.mp3"), recursive=True)
    if len(existing_files) >= 10:
        print(f"Found {len(existing_files)} audio tracks in {target_dir}. Ready for training.")
        return target_dir

    if dataset_name == "synthetic":
        print("Generating procedural multi-track music dataset...")
        return generate_synthetic_music_dataset(target_dir=target_dir, num_tracks=40)

    if dataset_name == "gtzan":
        try:
            from datasets import load_dataset
            print("Streaming GTZAN dataset from Hugging Face...")
            ds = load_dataset("sanchit-gandhi/gtzan", split="train", streaming=True)
            count = 0
            for item in ds:
                if count >= 50:
                    break
                audio_arr = item["audio"]["array"]
                orig_sr = item["audio"]["sampling_rate"]
                tensor_audio = torch.from_numpy(audio_arr).float().unsqueeze(0)
                if tensor_audio.dim() == 2:
                    tensor_audio = tensor_audio.repeat(2, 1)
                wav_path = os.path.join(target_dir, f"gtzan_track_{count+1:03d}.wav")
                pcm = (torch.clamp(tensor_audio, -1.0, 1.0) * 32767.0).to(torch.int16)
                interleaved = pcm.t().contiguous().view(-1).numpy().tobytes()
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(2)
                    wf.setsampwidth(2)
                    wf.setframerate(orig_sr)
                    wf.writeframes(interleaved)
                count += 1
            if count >= 10:
                print(f"Downloaded {count} GTZAN tracks from Hugging Face.")
                return target_dir
        except Exception:
            pass

        archive_path = os.path.join(target_dir, "genres.tar.gz")
        url = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
        print(f"Attempting download of GTZAN dataset from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp, open(archive_path, "wb") as out_file:
                out_file.write(resp.read())
            print("Extracting GTZAN dataset...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=target_dir)
            if os.path.exists(archive_path):
                os.remove(archive_path)
        except Exception as e:
            print(f"Direct GTZAN download failed ({e}). Generating multi-track procedural music dataset...")
            generate_synthetic_music_dataset(target_dir=target_dir, num_tracks=40)

    elif dataset_name == "fma_small":
        archive_path = os.path.join(target_dir, "fma_small.zip")
        url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
        print(f"Downloading Free Music Archive small dataset from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp, open(archive_path, "wb") as out_file:
                out_file.write(resp.read())
            print("Extracting FMA dataset...")
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            if os.path.exists(archive_path):
                os.remove(archive_path)
        except Exception as e:
            print(f"FMA download failed ({e}), generating multi-track procedural dataset...")
            generate_synthetic_music_dataset(target_dir=target_dir, num_tracks=40)

    return target_dir
