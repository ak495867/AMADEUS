import math
import torch
import torch.nn.functional as F

class LongformMusicGenerator:
    def __init__(self, pipeline, window_seconds=20, crossfade_seconds=2):
        self.pipeline = pipeline
        self.sample_rate = 48000
        self.window_seconds = window_seconds
        self.crossfade_seconds = crossfade_seconds
        self.latent_rate = 50
        self.window_latent_frames = window_seconds * self.latent_rate
        self.crossfade_latent_frames = crossfade_seconds * self.latent_rate

    def generate_longform(self, total_seconds=120, text_prompt="epic orchestral piece", reference_song=None, controls=None):
        if controls is None:
            controls = {}

        num_windows = math.ceil(total_seconds / (self.window_seconds - self.crossfade_seconds))
        combined_audio = None
        combined_stems = {"drums": None, "bass": None, "harmony": None, "lead": None}

        fade_samples = int(self.crossfade_seconds * self.sample_rate)
        fade_in = torch.linspace(0.0, 1.0, fade_samples)
        fade_out = torch.linspace(1.0, 0.0, fade_samples)

        for win_idx in range(num_windows):
            win_controls = dict(controls)
            win_controls["duration_s"] = self.window_seconds

            result = self.pipeline.generate(
                reference_song=reference_song,
                text_prompt=f"{text_prompt} section {win_idx + 1}",
                controls=win_controls
            )

            chunk_audio = result["audio"]
            chunk_stems = result["stems"]

            if combined_audio is None:
                combined_audio = chunk_audio
                for s in combined_stems:
                    if s in chunk_stems:
                        combined_stems[s] = chunk_stems[s][0]
            else:
                c_len = combined_audio.shape[-1]
                overlap_start = c_len - fade_samples

                dev = combined_audio.device
                fi = fade_in.to(dev).view(1, -1)
                fo = fade_out.to(dev).view(1, -1)

                overlap_prev = combined_audio[:, overlap_start:] * fo
                overlap_curr = chunk_audio[:, :fade_samples] * fi
                blended = overlap_prev + overlap_curr

                combined_audio = torch.cat([combined_audio[:, :overlap_start], blended, chunk_audio[:, fade_samples:]], dim=-1)

                for s in combined_stems:
                    if s in chunk_stems and combined_stems[s] is not None:
                        s_prev = combined_stems[s][:, overlap_start:] * fo
                        s_curr = chunk_stems[s][0][:, :fade_samples] * fi
                        s_blend = s_prev + s_curr
                        combined_stems[s] = torch.cat([combined_stems[s][:, :overlap_start], s_blend, chunk_stems[s][0][:, fade_samples:]], dim=-1)

        target_total_samples = int(total_seconds * self.sample_rate)
        combined_audio = combined_audio[:, :target_total_samples]
        for s in combined_stems:
            if combined_stems[s] is not None:
                combined_stems[s] = combined_stems[s][:, :target_total_samples]

        return {
            "audio": combined_audio,
            "stems": combined_stems,
            "duration": total_seconds,
            "total_windows": num_windows,
        }
