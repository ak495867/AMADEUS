import base64
import io
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from amadeus.pipeline.inference import AMADEUSPipeline
from amadeus.pipeline.realtime import RealtimeBGMGenerator

def create_app(pipeline=None):
    if pipeline is None:
        pipeline = AMADEUSPipeline()

    app = FastAPI(
        title="AMADEUS Generative Music API",
        version="1.0.0",
        description="Adaptive Music & Audio Diffusion Engine for Universal Soundtracks"
    )

    app.state.pipeline = pipeline
    app.state.realtime = RealtimeBGMGenerator(pipeline)

    class GenerationRequest(BaseModel):
        text: str = Field(..., example="warm lo-fi hip hop with jazz piano, rainy night")
        reference_audio_base64: Optional[str] = None
        duration_s: int = Field(default=10, ge=1, le=300)
        controls: Optional[Dict[str, Any]] = Field(
            default_factory=lambda: {"text_cfg": 4.5, "melody_strength": 0.5, "style_strength": 0.7, "steps": 8, "stems": True}
        )

    class AnalysisRequest(BaseModel):
        audio_base64: str

    class VerifyWatermarkRequest(BaseModel):
        audio_base64: str

    @app.get("/v1/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": "AMADEUS Inference Engine",
            "version": "1.0.0",
            "device": str(app.state.pipeline.device)
        }

    @app.post("/v1/generate")
    async def generate(req: GenerationRequest):
        try:
            ref_tensor = None
            if req.reference_audio_base64:
                raw_bytes = base64.b64decode(req.reference_audio_base64)
                ref_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32).view(2, -1)

            controls = dict(req.controls or {})
            controls["duration_s"] = req.duration_s

            res = app.state.pipeline.generate(
                reference_song=ref_tensor,
                text_prompt=req.text,
                controls=controls
            )

            audio_data = res["audio"].cpu().numpy().tobytes()
            b64_audio = base64.b64encode(audio_data).decode("utf-8")

            stems_b64 = {}
            if "stems" in res and res["stems"]:
                for s_name, s_tensor in res["stems"].items():
                    s_bytes = s_tensor.cpu().numpy().tobytes()
                    stems_b64[s_name] = base64.b64encode(s_bytes).decode("utf-8")

            return {
                "job_id": res["job_id"],
                "audio_base64": b64_audio,
                "stems_base64": stems_b64,
                "watermark_id": res["watermark_id"],
                "compliance": res["compliance"],
                "attribution": res["attribution"],
                "plan": res["plan"],
            }
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/analyze")
    async def analyze(req: AnalysisRequest):
        try:
            raw_bytes = base64.b64decode(req.audio_base64)
            audio_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32).view(2, -1)
            analysis = app.state.pipeline.frontend.analyze(audio_tensor)
            return {
                "key": analysis["key"],
                "mode": analysis["mode"],
                "bpm": analysis["bpm"],
                "time_sig": analysis["time_sig"],
                "structure": analysis["structure"],
                "lufs": analysis["lufs"],
                "duration": analysis["duration"],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/attribution/{job_id}")
    async def get_attribution(job_id: str):
        for log in app.state.pipeline.attribution_engine.logs:
            if log["job_id"] == job_id:
                return log
        raise HTTPException(status_code=404, detail="Attribution record not found")

    @app.post("/v1/verify_watermark")
    async def verify_watermark(req: VerifyWatermarkRequest):
        try:
            raw_bytes = base64.b64decode(req.audio_base64)
            audio_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32).view(2, -1)
            det = app.state.pipeline.watermarker.detect_watermark(audio_tensor)
            return det
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/v1/stream")
    async def stream_audio(websocket: WebSocket):
        await websocket.accept()
        try:
            init_data = await websocket.receive_json()
            prompt = init_data.get("text", "ambient BGM")
            chunks = init_data.get("chunks", 5)

            for chunk_res in app.state.realtime.stream_bgm(prompt, total_chunks=chunks):
                raw_bytes = chunk_res["audio"].cpu().numpy().tobytes()
                b64_chunk = base64.b64encode(raw_bytes).decode("utf-8")
                await websocket.send_json({
                    "chunk_index": chunk_res["chunk_index"],
                    "latency_ms": chunk_res["latency_ms"],
                    "audio_base64": b64_chunk,
                    "is_last": chunk_res["is_last"]
                })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            await websocket.close(code=1011)

    return app

app = create_app()
