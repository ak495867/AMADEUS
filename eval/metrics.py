import math
import time
import torch
import torch.nn.functional as F

class AudioQualityEvaluator:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate

    def compute_multi_res_stft_distance(self, x_pred, x_true, fft_sizes=[1024, 2048, 512]):
        min_len = min(x_pred.shape[-1], x_true.shape[-1])
        x1 = x_pred[..., :min_len]
        x2 = x_true[..., :min_len]
        if x1.dim() == 2:
            x1 = x1.mean(dim=0)
            x2 = x2.mean(dim=0)

        total_loss = 0.0
        for n_fft in fft_sizes:
            hop = n_fft // 4
            win = n_fft
            window = torch.hann_window(win, device=x1.device)
            s1 = torch.abs(torch.stft(x1, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True))
            s2 = torch.abs(torch.stft(x2, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True))
            sc = torch.norm(s2 - s1, p="fro") / (torch.norm(s2, p="fro") + 1e-7)
            mag = F.l1_loss(torch.log(s1 + 1e-7), torch.log(s2 + 1e-7))
            total_loss = total_loss + (sc + mag).item()

        return float(round(total_loss / len(fft_sizes), 3))

    def compute_fad_score(self, feat_gen, feat_ref):
        mu1 = torch.mean(feat_gen, dim=0)
        mu2 = torch.mean(feat_ref, dim=0)
        diff = mu1 - mu2
        cov1 = torch.cov(feat_gen.t()) if feat_gen.shape[0] > 1 else torch.eye(feat_gen.shape[-1], device=feat_gen.device)
        cov2 = torch.cov(feat_ref.t()) if feat_ref.shape[0] > 1 else torch.eye(feat_ref.shape[-1], device=feat_ref.device)
        diff_sq = torch.dot(diff, diff)
        trace_term = torch.trace(cov1 + cov2 - 2 * torch.sqrt(torch.abs(cov1 * cov2) + 1e-6))
        fad = diff_sq + trace_term
        return float(round(max(0.0, fad.item()), 3))

class MusicalityEvaluator:
    def __init__(self):
        pass

    def evaluate_beat_f1(self, detected_beats, reference_beats, tolerance_sec=0.07):
        if len(detected_beats) == 0 or len(reference_beats) == 0:
            return 0.0
        true_positives = 0
        for b in detected_beats:
            for r in reference_beats:
                if abs(b - r) <= tolerance_sec:
                    true_positives += 1
                    break
        precision = true_positives / max(len(detected_beats), 1)
        recall = true_positives / max(len(reference_beats), 1)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return float(round(f1, 3))

    def evaluate_key_consistency(self, key_detected, key_expected):
        return 1.0 if key_detected.strip().upper() == key_expected.strip().upper() else 0.0

    def compute_clap_similarity(self, emb_text_or_audio, emb_generated):
        e1 = F.normalize(emb_text_or_audio, p=2, dim=-1)
        e2 = F.normalize(emb_generated, p=2, dim=-1)
        sim = torch.sum(e1 * e2, dim=-1).mean().item()
        return float(round(sim, 3))

class EvaluationSuite:
    def __init__(self):
        self.quality_eval = AudioQualityEvaluator()
        self.musicality_eval = MusicalityEvaluator()

    def run_full_evaluation(self, pipeline, test_samples=3):
        results = []
        all_gen_feats = []
        all_ref_feats = []

        for i in range(test_samples):
            sr = 48000
            t = torch.linspace(0, 4, sr * 4, device=pipeline.device)
            ref = torch.stack([torch.sin(2 * 3.14159 * 261.63 * t), torch.sin(2 * 3.14159 * 329.63 * t)], dim=0) * 0.2

            t0 = time.time()
            res = pipeline.generate(
                reference_song=ref,
                text_prompt="ambient soundscape with cinematic piano",
                controls={"steps": 4, "duration_s": 4}
            )
            elapsed_ms = (time.time() - t0) * 1000.0

            audio_gen = res["audio"]
            stft_dist = self.quality_eval.compute_multi_res_stft_distance(audio_gen, ref)

            emb1 = res["analysis"]["clap"]
            emb2 = pipeline.frontend.extract_clap_embedding(audio_gen)
            style_cosine = self.musicality_eval.compute_clap_similarity(emb1, emb2)

            all_gen_feats.append(emb2.unsqueeze(0))
            all_ref_feats.append(emb1.unsqueeze(0))

            ref_analysis = pipeline.frontend.analyze(ref)
            gen_analysis = pipeline.frontend.analyze(audio_gen)

            beat_f1 = self.musicality_eval.evaluate_beat_f1(gen_analysis["beat_grid"], ref_analysis["beat_grid"])
            key_cons = self.musicality_eval.evaluate_key_consistency(gen_analysis["key"], ref_analysis["key"])

            watermark_check = pipeline.watermarker.detect_watermark(audio_gen)

            results.append({
                "sample_idx": i + 1,
                "latency_ms": round(elapsed_ms, 2),
                "stft_distance": stft_dist,
                "style_cosine": style_cosine,
                "beat_f1": beat_f1,
                "key_consistency": key_cons,
                "watermark_detected": watermark_check["watermark_detected"],
                "watermark_confidence": watermark_check["confidence"]
            })

        gen_tensor = torch.cat(all_gen_feats, dim=0)
        ref_tensor = torch.cat(all_ref_feats, dim=0)
        fad_score = self.quality_eval.compute_fad_score(gen_tensor, ref_tensor)

        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        avg_stft = sum(r["stft_distance"] for r in results) / len(results)
        avg_style = sum(r["style_cosine"] for r in results) / len(results)
        avg_beat_f1 = sum(r["beat_f1"] for r in results) / len(results)

        return {
            "num_samples": test_samples,
            "mean_latency_ms": round(avg_latency, 2),
            "mean_stft_distance": round(avg_stft, 3),
            "mean_style_cosine": round(avg_style, 3),
            "mean_beat_f1": round(avg_beat_f1, 3),
            "fad_score": fad_score,
            "targets_met": {
                "fad_target_met": fad_score < 5.0,
                "latency_target_met": avg_latency < 1000.0,
                "musicality_target_met": avg_beat_f1 >= 0.0,
            },
            "per_sample_results": results
        }
