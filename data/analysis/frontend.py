import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioAnalysisFrontend:
    def __init__(self, sample_rate=48000, device="cpu"):
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.major_profile = torch.tensor([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=torch.float32)
        self.minor_profile = torch.tensor([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=torch.float32)
        self.pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.projection_clap = nn.Linear(512, 512).to(self.device)
        self.projection_timbre = nn.Linear(256, 256).to(self.device)

    def compute_chroma(self, waveform):
        if waveform.dim() == 2:
            mono = waveform.mean(dim=0)
        else:
            mono = waveform
        n_fft = 4096
        hop_length = 1024
        window = torch.hann_window(n_fft, device=mono.device)
        stft = torch.stft(mono, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag = torch.abs(stft)
        num_bins = mag.shape[0]
        chroma = torch.zeros((12, mag.shape[1]), device=mono.device)
        freqs = torch.linspace(0, self.sample_rate / 2, num_bins, device=mono.device)
        for i in range(1, num_bins):
            f = freqs[i].item()
            if f < 20.0 or f > 5000.0:
                continue
            midi_pitch = 69 + 12 * math.log2(max(f, 1e-4) / 440.0)
            pc = int(round(midi_pitch)) % 12
            chroma[pc] += mag[i]
        norm = torch.norm(chroma, p=2, dim=0, keepdim=True) + 1e-7
        return chroma / norm

    def estimate_key(self, chroma):
        chroma_sum = chroma.sum(dim=1)
        norm = torch.norm(chroma_sum)
        if norm > 0:
            chroma_sum = chroma_sum / norm
        maj_prof = self.major_profile.to(chroma.device)
        min_prof = self.minor_profile.to(chroma.device)
        maj_prof = maj_prof / torch.norm(maj_prof)
        min_prof = min_prof / torch.norm(min_prof)

        best_score = -1e9
        best_key = "C"
        best_mode = "major"

        for shift in range(12):
            shifted = torch.roll(chroma_sum, -shift)
            score_maj = torch.dot(shifted, maj_prof).item()
            score_min = torch.dot(shifted, min_prof).item()

            if score_maj > best_score:
                best_score = score_maj
                best_key = self.pitch_classes[shift]
                best_mode = "major"
            if score_min > best_score:
                best_score = score_min
                best_key = self.pitch_classes[shift]
                best_mode = "minor"

        return best_key, best_mode

    def estimate_tempo_and_beats(self, waveform):
        if waveform.dim() == 2:
            mono = waveform.mean(dim=0)
        else:
            mono = waveform
        frame_len = 2048
        hop_len = 512
        if mono.numel() < frame_len:
            return 120.0, [0.0], [0.0]

        frames = mono.unfold(0, frame_len, hop_len)
        rms = torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-8)
        onset_env = F.relu(rms[1:] - rms[:-1])
        if onset_env.numel() == 0:
            return 120.0, [0.0, 0.5, 1.0], [0.0, 2.0]

        padded = F.pad(onset_env, (0, onset_env.size(0)))
        fft_env = torch.fft.rfft(padded)
        corr = torch.fft.irfft(fft_env * torch.conj(fft_env))[:onset_env.size(0)]

        min_lag = int((self.sample_rate / hop_len) * 60 / 220)
        max_lag = int((self.sample_rate / hop_len) * 60 / 60)
        max_lag = min(max_lag, corr.size(0) - 1)

        if max_lag > min_lag:
            lag = torch.argmax(corr[min_lag:max_lag]).item() + min_lag
            bpm = (60.0 * (self.sample_rate / hop_len)) / max(lag, 1)
        else:
            bpm = 120.0

        bpm = max(60.0, min(bpm, 200.0))
        beat_interval = 60.0 / bpm
        duration = mono.shape[0] / self.sample_rate
        beat_grid = torch.arange(0.0, duration, beat_interval, device=mono.device).tolist()
        downbeats = beat_grid[::4]
        return round(bpm, 1), beat_grid, downbeats

    def extract_f0(self, waveform, frame_rate=25):
        if waveform.dim() == 2:
            mono = waveform.mean(dim=0)
        else:
            mono = waveform

        hop_length = int(self.sample_rate / frame_rate)
        frame_size = hop_length * 2
        total_frames = max(1, mono.shape[-1] // hop_length)

        f0_list = []
        min_lag = int(self.sample_rate / 600.0)
        max_lag = int(self.sample_rate / 50.0)

        for i in range(total_frames):
            start = i * hop_length
            end = min(start + frame_size, mono.shape[-1])
            frame = mono[start:end]
            if frame.shape[0] < max_lag * 2:
                f0_list.append(0.0)
                continue

            frame = frame - frame.mean()
            pad = frame.shape[0]
            fft_frame = torch.fft.rfft(F.pad(frame, (0, pad)))
            autocorr = torch.fft.irfft(fft_frame * torch.conj(fft_frame))[:frame.shape[0]]

            search_region = autocorr[min_lag:min(max_lag, autocorr.shape[0])]
            if search_region.numel() > 0:
                best_lag = torch.argmax(search_region).item() + min_lag
                energy = autocorr[0].item() + 1e-7
                peak_val = autocorr[best_lag].item()
                if peak_val / energy > 0.3:
                    f0_list.append(self.sample_rate / best_lag)
                else:
                    f0_list.append(0.0)
            else:
                f0_list.append(0.0)

        return torch.tensor(f0_list, device=waveform.device, dtype=torch.float32).unsqueeze(0)

    def estimate_structure(self, chroma, duration):
        num_frames = chroma.shape[1]
        if num_frames < 4:
            return [{"section": "verse", "start": 0.0, "end": duration}]

        norm_chroma = F.normalize(chroma, p=2, dim=0)
        ssm = torch.mm(norm_chroma.t(), norm_chroma)
        kernel_size = min(16, max(3, num_frames // 4))
        if kernel_size % 2 == 0:
            kernel_size += 1

        checker = torch.ones((kernel_size, kernel_size), device=chroma.device)
        half = kernel_size // 2
        checker[:half, :half] = 1
        checker[half:, half:] = 1
        checker[:half, half:] = -1
        checker[half:, :half] = -1

        novelty = []
        for i in range(half, num_frames - half):
            sub = ssm[i - half:i + half + 1, i - half:i + half + 1]
            val = (sub * checker).sum().item()
            novelty.append((i, val))

        peaks = []
        for idx in range(1, len(novelty) - 1):
            if novelty[idx][1] > novelty[idx - 1][1] and novelty[idx][1] > novelty[idx + 1][1]:
                peaks.append(novelty[idx][0])

        time_per_frame = duration / max(num_frames, 1)
        labels = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "outro"]
        sections = []
        prev_time = 0.0
        for i, p in enumerate(peaks[:6]):
            boundary_time = p * time_per_frame
            if boundary_time - prev_time > 8.0:
                sections.append({
                    "section": labels[i % len(labels)],
                    "start": round(prev_time, 2),
                    "end": round(boundary_time, 2)
                })
                prev_time = boundary_time

        sections.append({
            "section": labels[len(sections) % len(labels)],
            "start": round(prev_time, 2),
            "end": round(duration, 2)
        })
        return sections

    def estimate_lufs(self, waveform):
        if waveform.dim() == 2:
            mono = waveform.mean(dim=0)
        else:
            mono = waveform
        power = torch.mean(mono ** 2).item()
        if power <= 1e-12:
            return -70.0
        lufs = -0.691 + 10.0 * math.log10(power)
        return float(round(lufs, 2))

    def extract_clap_embedding(self, waveform):
        if waveform.dim() == 2:
            mono = waveform.mean(dim=0)
        else:
            mono = waveform

        n_fft = 2048
        hop_length = 512
        window = torch.hann_window(n_fft, device=mono.device)
        stft = torch.stft(mono, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag = torch.abs(stft)
        spec_summary = mag.mean(dim=-1)
        if spec_summary.shape[0] > 512:
            spec_summary = F.adaptive_avg_pool1d(spec_summary.unsqueeze(0), 512).squeeze(0)
        elif spec_summary.shape[0] < 512:
            spec_summary = F.pad(spec_summary, (0, 512 - spec_summary.shape[0]))

        projected = self.projection_clap(spec_summary)
        return F.normalize(projected, p=2, dim=-1)

    def extract_timbre_embedding(self, waveform):
        if waveform.dim() == 2:
            mono = waveform.mean(dim=0)
        else:
            mono = waveform

        n_fft = 1024
        hop_length = 256
        window = torch.hann_window(n_fft, device=mono.device)
        stft = torch.stft(mono, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        mag = torch.abs(stft)
        timbre_spec = mag.std(dim=-1)
        if timbre_spec.shape[0] > 256:
            timbre_spec = F.adaptive_avg_pool1d(timbre_spec.unsqueeze(0), 256).squeeze(0)
        elif timbre_spec.shape[0] < 256:
            timbre_spec = F.pad(timbre_spec, (0, 256 - timbre_spec.shape[0]))

        projected = self.projection_timbre(timbre_spec)
        return F.normalize(projected, p=2, dim=-1)

    def analyze(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).repeat(2, 1)
        duration = waveform.shape[-1] / self.sample_rate
        chroma = self.compute_chroma(waveform)
        key, mode = self.estimate_key(chroma)
        bpm, beat_grid, downbeats = self.estimate_tempo_and_beats(waveform)
        f0 = self.extract_f0(waveform)
        structure = self.estimate_structure(chroma, duration)
        lufs = self.estimate_lufs(waveform)
        clap = self.extract_clap_embedding(waveform)
        timbre = self.extract_timbre_embedding(waveform)

        return {
            "key": key,
            "mode": mode,
            "bpm": bpm,
            "time_sig": "4/4",
            "structure": structure,
            "clap": clap,
            "timbre": timbre,
            "f0": f0,
            "chroma": chroma,
            "lufs": lufs,
            "beat_grid": beat_grid,
            "downbeats": downbeats,
            "duration": duration,
        }
