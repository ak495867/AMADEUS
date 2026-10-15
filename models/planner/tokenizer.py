class PlanTokenizer:
    def __init__(self):
        special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "|", "[4/4]", "[3/4]", "[6/8]"]
        keys = [f"[KEY={p}{m}]" for p in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] for m in ["", "m"]]
        bpms = [f"[BPM={b}]" for b in range(50, 220)]
        sections = [f"[{s.upper()}" for s in ["intro", "verse", "chorus", "bridge", "outro", "drop", "build", "solo"]]
        bars = [f"{b} bars" for b in [2, 4, 8, 12, 16, 24, 32]]
        instruments = [
            "drums", "bass", "piano", "pad", "guitar", "strings", "synth", "lead",
            "vocal-space", "full kit", "fade", "drop drums", "brass", "arpeggio", "percussion"
        ]
        dynamics = ["+3 dB energy", "+6 dB energy", "-3 dB energy", "-6 dB energy", "breakdown", "crescendo"]

        self.vocab = special_tokens + keys + bpms + sections + bars + instruments + dynamics
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for idx, token in enumerate(self.vocab)}
        self.pad_token_id = self.token2id["[PAD]"]
        self.bos_token_id = self.token2id["[BOS]"]
        self.eos_token_id = self.token2id["[EOS]"]
        self.vocab_size = len(self.vocab)

    def encode(self, plan_str):
        raw_tokens = plan_str.replace("]", "] ").replace("[", " [").replace("|", " | ").split()
        ids = [self.bos_token_id]
        for t in raw_tokens:
            if t in self.token2id:
                ids.append(self.token2id[t])
            else:
                ids.append(self.token2id["[UNK]"])
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                continue
            tokens.append(self.id2token.get(i, "[UNK]"))
        return " ".join(tokens)

    def format_plan(self, key, mode, bpm, sections):
        mode_str = "m" if mode == "minor" else ""
        header = f"[BPM={int(bpm)}][KEY={key}{mode_str}][4/4]"
        lines = [header]
        for sec in sections:
            name = sec.get("section", "verse").upper()
            duration = sec.get("end", 0.0) - sec.get("start", 0.0)
            bars = max(4, int(round((duration / 60.0 * bpm) / 4.0)) * 4)
            inst = sec.get("instrumentation", "drums, bass")
            lines.append(f"[{name} {bars} bars | {inst}]")
        return "\n".join(lines)
