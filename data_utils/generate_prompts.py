"""Generate music-style text prompts using served Gemma model.

- For painting dataset: captions images → music generation prompts
- For Suno dataset: rewrites tags into dense user-style prompts

Saves prompts into the cached index.jsonl manifest.

Usage:
  python src/data_utils/generate_prompts.py --config src/configs/image_cond.yaml
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_GEMMA_URL = "http://localhost:9002/v1/chat/completions"
GEMMA_MODEL = "google/gemma-4-E4B-it"

def _set_gemma_url(url):
    global _GEMMA_URL
    _GEMMA_URL = url

IMAGE_TO_MUSIC_PROMPT = """Look at this painting and write a short music generation prompt (1-2 sentences) describing what instrumental music would match it. Include genre, mood, tempo, and key instruments. Be concise and direct — output ONLY the prompt, no explanation."""

TAGS_TO_PROMPT = """Rewrite the following music tags into a concise music generation prompt (1-2 sentences) as if a user is requesting a song. Keep the key details (genre, mood, tempo, instruments) but make it sound natural. Output ONLY the prompt.

Tags: {tags}"""

CATEGORY_HINTS = {
    "安静平和": "calm, peaceful, serene",
    "悲伤孤傲": "sad, lonely, melancholic",
    "活泼欢快": "lively, happy, upbeat",
    "激昂肆意": "passionate, intense, powerful",
}


SYSTEM_PROMPT = "You are a concise music prompt writer. Output ONLY the music generation prompt — no thinking, no explanation, no bullet points. Just 1-2 sentences."


def query_gemma(messages: list, max_tokens: int = 150, retries: int = 3) -> str:
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    for attempt in range(retries):
        try:
            resp = requests.post(_GEMMA_URL, json={
                "model": GEMMA_MODEL,
                "messages": full_messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }, timeout=60)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip thinking preamble — find the last sentence-like chunk
            lines = [l.strip() for l in text.replace("\n\n", "\n").split("\n") if l.strip()]
            skip_prefixes = ("The user", "*", "**", "Analysis", "Constraint", "I need",
                             "Image analysis", "Mood:", "Genre:", "Tempo:", "Instrument",
                             "Drafting", "Key", "Production", "Style", "#")
            candidates = [l for l in lines if not any(l.startswith(x) for x in skip_prefixes)]
            if candidates:
                return candidates[-1]
            return lines[-1] if lines else text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise e


def caption_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    ext = Path(image_path).suffix.lstrip(".")
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            {"type": "text", "text": IMAGE_TO_MUSIC_PROMPT},
        ]
    }]
    return query_gemma(messages)


def rewrite_tags(tags: str) -> str:
    messages = [{
        "role": "user",
        "content": TAGS_TO_PROMPT.format(tags=tags),
    }]
    return query_gemma(messages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/image_cond.yaml")
    parser.add_argument("--gemma-url", type=str, default=_GEMMA_URL)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip entries that already have a prompt")
    args = parser.parse_args()

    _set_gemma_url(args.gemma_url)

    import yaml
    with open(args.config) as f:
        raw = f.read()
    raw = raw.replace("${dataset_root}", str(PROJECT_ROOT / "data" / "image_music"))
    raw = raw.replace("${diffrhythm_root}", str(PROJECT_ROOT / "src" / "DiffRhythm"))
    raw = raw.replace("${image_encoder.embed_dim}", "768")
    cfg = yaml.safe_load(raw)

    cache_dir_raw = cfg["cache_dir"]
    if not Path(cache_dir_raw).is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir_raw
    else:
        cache_dir = Path(cache_dir_raw)
    manifest_path = cache_dir / "index.jsonl"

    entries = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    print(f"Total entries: {len(entries)}")

    updated = 0
    for entry in tqdm(entries, desc="Generating prompts"):
        if args.skip_existing and entry.get("text_prompt"):
            continue

        try:
            source = entry.get("source", "painting")

            if source == "suno":
                tags = entry.get("tags", "")
                if not tags:
                    continue
                prompt = rewrite_tags(tags)
            else:
                image_path = entry.get("image_path", "")
                if not image_path or not Path(image_path).exists():
                    category = entry.get("category", "")
                    hint = CATEGORY_HINTS.get(category, category)
                    prompt = f"Generate {hint} instrumental music"
                else:
                    prompt = caption_image(image_path)

            entry["text_prompt"] = prompt
            updated += 1

        except Exception as e:
            print(f"Error [{entry.get('idx', '?')}]: {e}")
            category = entry.get("category", "")
            hint = CATEGORY_HINTS.get(category, category)
            entry["text_prompt"] = f"Generate {hint} instrumental music"

    with open(manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Save annotation CSV to data/ for easy inspection
    dataset_dir = PROJECT_ROOT / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = dataset_dir / "annotations.jsonl"
    with open(annotation_path, "w") as f:
        for entry in entries:
            row = {
                "idx": entry.get("idx"),
                "source": entry.get("source", "painting"),
                "category": entry.get("category", ""),
                "tags": entry.get("tags", ""),
                "text_prompt": entry.get("text_prompt", ""),
                "latent_frames": entry.get("latent_frames"),
                "image_path": entry.get("image_path", ""),
                "music_path": entry.get("music_path", ""),
                "suno_id": entry.get("suno_id", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Updated {updated} entries with text prompts")
    print(f"Annotation saved to {annotation_path}")


if __name__ == "__main__":
    main()
