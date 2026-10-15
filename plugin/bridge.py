import json
import socket
import threading
import torch

class DAWPluginBridge:
    def __init__(self, pipeline, port=9099):
        self.pipeline = pipeline
        self.port = port
        self.is_running = False
        self.host_tempo = 120.0
        self.time_sig = (4, 4)
        self.is_playing = False
        self.stems_enabled = True

    def update_host_state(self, tempo, time_sig_num, time_sig_denom, is_playing):
        self.host_tempo = float(tempo)
        self.time_sig = (int(time_sig_num), int(time_sig_denom))
        self.is_playing = bool(is_playing)

    def process_midi_event(self, note_number, velocity, timestamp):
        return {
            "note": note_number,
            "velocity": velocity,
            "timestamp": timestamp,
            "processed": True
        }

    def render_block(self, prompt, duration_seconds=4, num_steps=4):
        controls = {
            "steps": num_steps,
            "duration_s": duration_seconds,
            "text_cfg": 2.5,
            "stems": self.stems_enabled
        }
        res = self.pipeline.generate(text_prompt=prompt, controls=controls)
        return {
            "job_id": res["job_id"],
            "waveform_shape": list(res["audio"].shape),
            "stems_available": list(res["stems"].keys()) if "stems" in res else []
        }

    def start_ipc_server(self):
        self.is_running = True
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", self.port))
        server.listen(1)

        def listen_loop():
            while self.is_running:
                try:
                    conn, _ = server.accept()
                    data = conn.recv(4096).decode("utf-8")
                    if not data:
                        conn.close()
                        continue
                    req = json.loads(data)
                    cmd = req.get("command")

                    if cmd == "sync":
                        self.update_host_state(
                            req.get("tempo", 120.0),
                            req.get("time_sig_num", 4),
                            req.get("time_sig_denom", 4),
                            req.get("is_playing", False)
                        )
                        resp = {"status": "synced", "tempo": self.host_tempo}
                    elif cmd == "render":
                        resp = self.render_block(req.get("prompt", "lo-fi beat"))
                    else:
                        resp = {"status": "unknown_command"}

                    conn.sendall(json.dumps(resp).encode("utf-8"))
                    conn.close()
                except Exception:
                    pass

        t = threading.Thread(target=listen_loop, daemon=True)
        t.start()

    def stop(self):
        self.is_running = False
