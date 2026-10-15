import os
import uuid
import torch
import torch.nn.functional as F

from amadeus.data.analysis.frontend import AudioAnalysisFrontend
from amadeus.retrieval.index import MusicRetrievalIndex, AttributionEngine
from amadeus.models.planner.transformer import StructurePlanner
from amadeus.models.planner.tokenizer import PlanTokenizer
from amadeus.models.lm.transformer import SemanticTokenLM
from amadeus.models.lm.conditioning import TextConditioningEncoder
from amadeus.models.dit.dit import DiffusionTransformer
from amadeus.models.dit.flow_matching import FlowMatchingScheduler
from amadeus.models.distill.student import DistilledDiTStudent
from amadeus.models.codec.decoder import HiFiGANCodecDecoder
from amadeus.safety.watermark import AudioSealWatermarker
from amadeus.safety.filters import SafetyFilterSuite
from amadeus.pipeline.checkpoint import load_model_checkpoint

class AMADEUSPipeline:
    def __init__(
        self,
        planner=None,
        lm=None,
        dit=None,
        codec=None,
        device="cpu",
        distilled_student=None,
        checkpoint_dir=None
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.frontend = AudioAnalysisFrontend(sample_rate=48000, device=self.device)
        self.retrieval_index = MusicRetrievalIndex(embedding_dim=512)
        self.attribution_engine = AttributionEngine(self.retrieval_index)
        self.plan_tokenizer = PlanTokenizer()
        self.text_encoder = TextConditioningEncoder(in_dim=768, out_dim=768).to(self.device)

        self.planner = planner if planner is not None else StructurePlanner(
            vocab_size=self.plan_tokenizer.vocab_size,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            max_seq_len=512
        ).to(self.device)

        self.lm = lm if lm is not None else SemanticTokenLM(
            vocab_size=2048,
            num_codebooks=4,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            max_seq_len=1024,
            style_dim=512,
            cross_dim=256,
            text_dim=768
        ).to(self.device)

        self.dit = dit if dit is not None else DiffusionTransformer(
            in_channels=64,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            patch_size=4,
            style_dim=512,
            cross_dim=256
        ).to(self.device)

        self.codec = codec if codec is not None else HiFiGANCodecDecoder(
            latent_dim=64,
            base_channels=128,
            out_channels=2,
            ratios=[2, 3, 4, 5, 8],
            enable_stems=True
        ).to(self.device)

        self.flow_scheduler = FlowMatchingScheduler()
        self.distilled_student = distilled_student if distilled_student is not None else DistilledDiTStudent(self.dit)

        self.watermarker = AudioSealWatermarker(sample_rate=48000).to(self.device)
        self.safety_filters = SafetyFilterSuite()

        self._seed_default_catalog()

        self.planner.eval()
        self.lm.eval()
        self.dit.eval()
        self.codec.eval()
        self.text_encoder.eval()
        self.distilled_student.eval()
        self.watermarker.eval()

        if checkpoint_dir is not None:
            self.load_checkpoints(checkpoint_dir)

    def _seed_default_catalog(self):
        seeds = []
        for i in range(10):
            seeds.append({
                "track_id": f"CATALOG-TRK-{1000 + i}",
                "artist": f"AMADEUS In-House Studio {i + 1}",
                "title": f"Cinematic Soundscape Theme #{i + 1}",
                "license_id": "COMMERCIAL-CLEAN-V1",
                "embedding": torch.randn(512),
                "bpm": 90.0 + i * 5,
                "key": "C",
            })
        self.retrieval_index.add_tracks(seeds)

    def load_checkpoints(self, checkpoint_dir):
        codec_path = os.path.join(checkpoint_dir, "codec_decoder_latest.pt")
        if os.path.exists(codec_path):
            load_model_checkpoint(self.codec, codec_path, device=self.device)

        dit_path = os.path.join(checkpoint_dir, "dit_latest.pt")
        if os.path.exists(dit_path):
            load_model_checkpoint(self.dit, dit_path, device=self.device)

        lm_path = os.path.join(checkpoint_dir, "lm_latest.pt")
        if os.path.exists(lm_path):
            load_model_checkpoint(self.lm, lm_path, device=self.device)

    @torch.no_grad()
    def generate(self, reference_song=None, text_prompt="ambient soundtrack", controls=None):
        if controls is None:
            controls = {}

        is_safe, error_msg = self.safety_filters.check_prompt_artists(text_prompt)
        if not is_safe:
            raise ValueError(error_msg)

        job_id = f"job-{uuid.uuid4().hex[:12]}"
        steps = controls.get("steps", 32)
        duration_s = controls.get("duration_s", 10)
        text_cfg = controls.get("text_cfg", 3.5)
        melody_strength = controls.get("melody_strength", 0.6)
        style_strength = controls.get("style_strength", 0.7)
        return_stems = controls.get("stems", True)

        if reference_song is not None:
            if not isinstance(reference_song, torch.Tensor):
                reference_song = torch.tensor(reference_song, dtype=torch.float32)
            if reference_song.device != self.device:
                reference_song = reference_song.to(self.device)
            analysis = self.frontend.analyze(reference_song)
        else:
            sr = 48000
            t = torch.linspace(0, 2, sr * 2, device=self.device)
            dummy_audio = torch.stack([torch.sin(2 * 3.14159 * 220 * t), torch.sin(2 * 3.14159 * 440 * t)], dim=0) * 0.1
            analysis = self.frontend.analyze(dummy_audio)

        matches, soft_style = self.retrieval_index.search(analysis["clap"], k=3)
        soft_style = soft_style.to(self.device) * style_strength
        attribution_record = self.attribution_engine.log_generation(job_id, matches, text_prompt, duration_s)

        plan_str = self.plan_tokenizer.format_plan(analysis["key"], analysis["mode"], analysis["bpm"], analysis["structure"])
        plan_ids = torch.tensor([self.plan_tokenizer.encode(plan_str)], device=self.device)
        planned_tokens = self.planner.generate(plan_ids, max_new_tokens=16)

        prefix_tokens = torch.zeros((1, 4, 4), dtype=torch.long, device=self.device)
        text_emb = self.text_encoder(text_prompt, device=self.device)

        target_lm_frames = 25 * duration_s
        f0_raw = analysis["f0"]
        if f0_raw.shape[-1] != target_lm_frames:
            f0_cond = F.interpolate(f0_raw.unsqueeze(0), size=target_lm_frames, mode="linear", align_corners=False).squeeze(0)
        else:
            f0_cond = f0_raw

        chroma_raw = analysis["chroma"]
        if chroma_raw.shape[-1] != target_lm_frames:
            chroma_cond = F.interpolate(chroma_raw.unsqueeze(0), size=target_lm_frames, mode="linear", align_corners=False).squeeze(0)
        else:
            chroma_cond = chroma_raw

        beat_grid_dummy = torch.zeros(1, target_lm_frames, 2, device=self.device)

        sem_tokens = self.lm.sample(
            prefix_tokens=prefix_tokens,
            style_vec=soft_style,
            text_emb=text_emb,
            melody_f0=f0_cond,
            melody_chroma=chroma_cond,
            beat_grid=beat_grid_dummy,
            max_new_tokens=target_lm_frames,
            cfg=text_cfg
        )

        latent_len = 50 * duration_s
        cond_dict = {
            "style": soft_style,
            "loudness": torch.tensor([[analysis["lufs"]]], device=self.device),
            "text": text_emb,
            "melody": torch.randn(1, 256, device=self.device) * melody_strength,
            "rhythm": torch.randn(1, 64, device=self.device),
            "sem": sem_tokens
        }

        latent_shape = (1, 64, latent_len)

        if steps <= 4:
            z = self.distilled_student.sample(latent_shape, device=self.device, cond=cond_dict)
        else:
            z = self.flow_scheduler.sample_euler(
                model=self.dit,
                shape=latent_shape,
                device=self.device,
                steps=steps,
                cond=cond_dict,
                cfg_scale=text_cfg,
                melody_cfg_scale=melody_strength * 2.0
            )

        if return_stems:
            audio, stems = self.codec(z, return_stems=True)
            stems = {k: v.detach() for k, v in stems.items()}
        else:
            audio = self.codec(z, return_stems=False)
            stems = {}

        audio_watermarked = self.watermarker.embed_watermark(audio[0])
        watermark_id = self.watermarker.generate_watermark_id(text_prompt, job_id)

        compliance = self.safety_filters.generate_compliance_manifest(job_id, attribution_record["attributed_sources"])

        return {
            "job_id": job_id,
            "audio": audio_watermarked.detach(),
            "stems": stems,
            "watermark_id": watermark_id,
            "attribution": attribution_record,
            "compliance": compliance,
            "analysis": analysis,
            "plan": plan_str,
        }
