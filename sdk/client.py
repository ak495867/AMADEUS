import base64
import json
import urllib.request
import urllib.error

class AmadeusClient:
    def __init__(self, endpoint="http://localhost:8000"):
        self.endpoint = endpoint.rstrip("/")

    def health(self):
        req = urllib.request.Request(f"{self.endpoint}/v1/health")
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())

    def generate(self, text, reference_audio_bytes=None, duration_s=10, controls=None):
        payload = {
            "text": text,
            "duration_s": duration_s,
            "controls": controls or {"steps": 8, "stems": True}
        }
        if reference_audio_bytes is not None:
            payload["reference_audio_base64"] = base64.b64encode(reference_audio_bytes).decode()

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint}/v1/generate",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            res_json = json.loads(resp.read().decode())
            raw_audio = base64.b64decode(res_json["audio_base64"])
            return {
                "job_id": res_json["job_id"],
                "audio_bytes": raw_audio,
                "watermark_id": res_json["watermark_id"],
                "attribution": res_json["attribution"],
                "plan": res_json["plan"],
            }

    def analyze(self, audio_bytes):
        payload = {"audio_base64": base64.b64encode(audio_bytes).decode()}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint}/v1/analyze",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())

    def verify_watermark(self, audio_bytes):
        payload = {"audio_base64": base64.b64encode(audio_bytes).decode()}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint}/v1/verify_watermark",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
