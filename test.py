import requests

payload = {
    "text": "dreamy synthwave sunset drive",
    "duration_s": 10,
    "controls": {
        "steps": 4,
        "stems": True
    }
}

response = requests.post("http://localhost:8000/v1/generate", json=payload)
data = response.json()
print("Job ID:", data["job_id"])
print("Watermark ID:", data["watermark_id"])
print("Attribution:", data["attribution"]["attributed_sources"])