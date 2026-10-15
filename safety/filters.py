import re

class SafetyFilterSuite:
    def __init__(self, max_melody_ngram=8):
        self.max_melody_ngram = max_melody_ngram
        self.blocked_artists = [
            "taylor swift", "drake", "beyonce", "ed sheeran", "billie eilish",
            "the weeknd", "bad bunny", "ariana grande", "justin bieber", "adele",
            "kendrick lamar", "post malone", "dua lipa", "harry styles", "rihanna",
            "travis scott", "bruno mars", "kanye west", "eminem", "olivia rodrigo"
        ]

    def check_prompt_artists(self, prompt_text):
        lower = prompt_text.lower()
        for artist in self.blocked_artists:
            pattern = rf"\b(style of|sounds like|sound like|like|imitate)\s+{re.escape(artist)}\b"
            if re.search(pattern, lower) or f"by {artist}" in lower or artist in lower:
                return False, f"Prompt contains restricted living artist reference: '{artist}'. Please use generic style, genre, or mood descriptors."
        return True, ""

    def extract_pitch_ngrams(self, notes, n):
        if len(notes) < n:
            return set()
        return set(tuple(notes[i:i + n]) for i in range(len(notes) - n + 1))

    def check_melody_copying(self, generated_notes, reference_notes):
        if not reference_notes or not generated_notes:
            return True, 0
        ref_ngrams = self.extract_pitch_ngrams(reference_notes, self.max_melody_ngram)
        gen_ngrams = self.extract_pitch_ngrams(generated_notes, self.max_melody_ngram)
        overlap = ref_ngrams.intersection(gen_ngrams)
        if len(overlap) > 0:
            return False, len(overlap)
        return True, 0

    def generate_compliance_manifest(self, job_id, attribution_records):
        return {
            "job_id": job_id,
            "eu_ai_act_compliance": {
                "article_52_disclosure": "This audio was synthetically generated using AMADEUS AI engine.",
                "watermark_standard": "AudioSeal v1.0",
                "machine_readable": True,
            },
            "attribution_count": len(attribution_records),
            "attributions": attribution_records,
            "timestamp": "ISO-8601-verified",
        }
