import json
import time

class OptOutRegistry:
    def __init__(self):
        self.opted_out_artists = set()
        self.opted_out_isrcs = set()

    def register_opt_out(self, artist_name=None, isrc=None):
        if artist_name:
            self.opted_out_artists.add(artist_name.strip().lower())
        if isrc:
            self.opted_out_isrcs.add(isrc.strip().upper())

    def is_opted_out(self, artist_name=None, isrc=None):
        if artist_name and artist_name.strip().lower() in self.opted_out_artists:
            return True
        if isrc and isrc.strip().upper() in self.opted_out_isrcs:
            return True
        return False

class LicenseRegistry:
    def __init__(self):
        self.licenses = {}
        self.track_licenses = {}
        self.opt_out = OptOutRegistry()

    def add_license(self, license_id, license_type, licensor, commercial_allowed=True, attribution_required=True):
        self.licenses[license_id] = {
            "license_id": license_id,
            "type": license_type,
            "licensor": licensor,
            "commercial_allowed": commercial_allowed,
            "attribution_required": attribution_required,
            "registered_at": time.time(),
        }

    def register_track(self, track_id, license_id, artist_name, title, isrc=None):
        if self.opt_out.is_opted_out(artist_name=artist_name, isrc=isrc):
            raise ValueError(f"Track {track_id} is opted out by rights holder.")
        if license_id not in self.licenses:
            raise ValueError(f"Unknown license ID: {license_id}")

        self.track_licenses[track_id] = {
            "track_id": track_id,
            "license_id": license_id,
            "artist": artist_name,
            "title": title,
            "isrc": isrc,
        }

    def get_track_metadata(self, track_id):
        meta = self.track_licenses.get(track_id)
        if not meta:
            return None
        lic = self.licenses.get(meta["license_id"], {})
        return {**meta, "license_info": lic}

    def get_attribution(self, track_ids):
        attributions = []
        for tid in track_ids:
            meta = self.get_track_metadata(tid)
            if meta:
                attributions.append({
                    "track_id": tid,
                    "artist": meta["artist"],
                    "title": meta["title"],
                    "licensor": meta["license_info"].get("licensor", "Unknown"),
                    "license_type": meta["license_info"].get("type", "Custom"),
                })
        return attributions
