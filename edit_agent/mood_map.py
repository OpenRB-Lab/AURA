"""Mood → painting-category mapping and reference-image selection.

The painting dataset has 4 mood categories; suno chunks get a mood-matched image by
mapping their description.mood keywords onto a category. Painting chunks use one of
their actually-paired images (from data/annotations.jsonl).
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAINTING_DIR = PROJECT_ROOT / "data/image_music/painting_dataset"
ANNOTATIONS = PROJECT_ROOT / "data/annotations.jsonl"

CATEGORIES = ["安静平和", "悲伤孤傲", "活泼欢快", "激昂肆意"]

CATEGORY_EN = {
    "安静平和": "calm, peaceful, serene",
    "悲伤孤傲": "sad, lonely, melancholic",
    "活泼欢快": "lively, cheerful, upbeat",
    "激昂肆意": "passionate, intense, epic",
}

# keyword → category; matched against lowercase mood strings
_KEYWORDS = {
    "安静平和": ["calm", "peaceful", "serene", "gentle", "soft", "ambient", "relax",
                 "tranquil", "meditat", "soothing", "dreamy", "ethereal", "chill",
                 "mellow", "quiet", "warm", "atmospheric", "floating"],
    "悲伤孤傲": ["sad", "melanchol", "lonely", "somber", "mournful", "wistful",
                 "nostalg", "dark", "gloomy", "sorrow", "bitters", "haunting",
                 "longing", "emotional", "reflective", "pensive", "moody"],
    "活泼欢快": ["happy", "cheerful", "upbeat", "lively", "playful", "joyful", "fun",
                 "bright", "bouncy", "energetic", "danc", "groovy", "sunny",
                 "festive", "light", "optimistic", "funky", "catchy"],
    "激昂肆意": ["epic", "intense", "passionate", "powerful", "aggressive", "driving",
                 "triumphant", "heroic", "dramatic", "fierce", "bold", "anthemic",
                 "climactic", "explosive", "hard", "battle", "grand", "cinematic"],
}


def mood_to_category(moods: list[str], energy: str = "medium") -> str:
    """Map a list of mood words to the best painting category by keyword votes."""
    text = " ".join(moods).lower()
    scores = {c: sum(1 for kw in kws if kw in text) for c, kws in _KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:  # no keyword hit → fall back on energy
        return {"low": "安静平和", "high": "激昂肆意"}.get(energy, "活泼欢快")
    return best


class ImagePool:
    def __init__(self):
        self.by_category = {
            c: sorted(str(p.relative_to(PROJECT_ROOT))
                      for p in (PAINTING_DIR / c).iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            for c in CATEGORIES
        }
        # painting music stem -> list of truly-paired image paths (repo-relative)
        self.paired: dict[str, list[str]] = {}
        with open(ANNOTATIONS) as f:
            for line in f:
                row = json.loads(line)
                if row.get("source") != "painting" or not row.get("image_path"):
                    continue
                stem = Path(row["music_path"]).stem
                img = str(Path(row["image_path"]).resolve().relative_to(PROJECT_ROOT))
                self.paired.setdefault(stem, []).append(img)

    def pick(self, category: str, rng: random.Random) -> str:
        return rng.choice(self.by_category[category])

    def paired_image(self, music_stem: str, rng: random.Random) -> str | None:
        imgs = self.paired.get(music_stem)
        return rng.choice(imgs) if imgs else None

    def category_of(self, image_path: str) -> str | None:
        for c in CATEGORIES:
            if f"/{c}/" in image_path:
                return c
        return None
