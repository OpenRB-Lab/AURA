"""Generate music prompts via Gemma. Writes after every batch for crash safety."""

import base64
import json
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEMMA_URL = "http://localhost:9002/v1/chat/completions"
GEMMA_MODEL = "google/gemma-4-E4B-it"

IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAQABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAFBABAAAAAAAAAAAAAAAAAAAAcP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AKYH/9k="}},
            {"type": "text", "text": "Write a music generation prompt for this image. Your response must start with PROMPT: followed by the prompt."}
        ]
    },
    {
        "role": "assistant",
        "content": "PROMPT: Create a serene ambient piece with gentle piano arpeggios, soft string pads, and delicate wind chimes at a slow tempo, evoking a peaceful and contemplative atmosphere."
    },
]

TAG_MESSAGES = [
    {
        "role": "user",
        "content": "Rewrite these music tags as a music generation prompt. Your response must start with PROMPT: followed by the prompt.\n\nTags: Lo-fi hip hop, mellow, 85 BPM, vinyl crackle, jazzy Rhodes piano, soft drum machine, rainy day vibes, nostalgic"
    },
    {
        "role": "assistant",
        "content": "PROMPT: Create a mellow lo-fi hip hop track at 85 BPM with jazzy Rhodes piano chords, a soft drum machine beat, and vinyl crackle texture, evoking the nostalgic warmth of a rainy afternoon."
    },
    {
        "role": "user",
        "content": "Rewrite these music tags as a music generation prompt. Your response must start with PROMPT: followed by the prompt.\n\nTags: Epic orchestral cinematic, 140 BPM, powerful brass, sweeping strings, thunderous percussion, heroic theme, battle scene"
    },
    {
        "role": "assistant",
        "content": "PROMPT: Compose an epic orchestral cinematic piece at 140 BPM with powerful brass fanfares, sweeping string melodies, and thunderous percussion building to a heroic climax, suitable for an intense battle scene."
    },
]

CATEGORY_HINTS = {
    "安静平和": "calm, peaceful, serene",
    "悲伤孤傲": "sad, lonely, melancholic",
    "活泼欢快": "lively, happy, upbeat",
    "激昂肆意": "passionate, intense, powerful",
}


def query_gemma(messages, max_tokens=300, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.post(GEMMA_URL, json={
                "model": GEMMA_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }, timeout=60)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return clean_response(text)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return None


def clean_response(text: str) -> str:
    # Extract everything after PROMPT: delimiter
    match = re.search(r'PROMPT:\s*(.+)', text, re.DOTALL)
    if match:
        prompt = match.group(1).strip()
        # Take only the first paragraph after PROMPT:
        prompt = re.split(r'\n\s*\n', prompt)[0]
        return prompt.strip()

    # Fallback: find action verb starts
    music_start = re.search(
        r'(Create |Compose |Generate |Produce |Make )',
        text, re.IGNORECASE
    )
    if music_start:
        result = text[music_start.start():]
        return re.split(r'\n\s*\n', result)[0].strip()

    return text.strip()


def caption_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = Path(image_path).suffix.lstrip(".")
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"

    messages = IMAGE_MESSAGES + [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": "Write a music generation prompt for this image. Include genre, mood, tempo, instruments, and atmosphere. Your response must start with PROMPT: followed by the prompt."},
        ]
    }]
    return query_gemma(messages)


def rewrite_tags(tags: str) -> str:
    messages = TAG_MESSAGES + [{
        "role": "user",
        "content": f"Rewrite these music tags as a music generation prompt. Your response must start with PROMPT: followed by the prompt.\n\nTags: {tags}",
    }]
    return query_gemma(messages)


def process_entry(entry):
    source = entry.get("source", "painting")

    if source == "suno":
        tags = entry.get("tags", "")
        if not tags:
            return "Generate instrumental music"
        return rewrite_tags(tags)

    image_path = entry.get("image_path", "")
    if image_path and Path(image_path).exists():
        return caption_image(image_path)

    cat = entry.get("category", "")
    hint = CATEGORY_HINTS.get(cat, cat)
    return f"Generate {hint} instrumental music"


def main():
    manifest_path = PROJECT_ROOT / "data/image_music/cached_latents/index.jsonl"
    annotation_path = PROJECT_ROOT / "data/annotations.jsonl"

    with open(manifest_path) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    print(f"Total entries: {len(entries)}")

    # Clear old prompts to regenerate all
    for e in entries:
        e.pop("text_prompt", None)

    save_interval = 50
    updated = 0

    for i, entry in enumerate(tqdm(entries, desc="Generating")):
        if entry.get("text_prompt"):
            continue

        prompt = process_entry(entry)
        if prompt:
            entry["text_prompt"] = prompt
        else:
            cat = entry.get("category", "")
            hint = CATEGORY_HINTS.get(cat, cat)
            entry["text_prompt"] = f"Generate {hint} instrumental music"
        updated += 1

        if updated > 0 and updated % save_interval == 0:
            with open(manifest_path, "w") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            tqdm.write(f"  Saved checkpoint at {updated} updates")

    # Final save
    with open(manifest_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Save annotations
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    with open(annotation_path, "w") as f:
        for e in entries:
            row = {
                "idx": e.get("idx"),
                "source": e.get("source", "painting"),
                "category": e.get("category", ""),
                "tags": e.get("tags", ""),
                "text_prompt": e.get("text_prompt", ""),
                "latent_frames": e.get("latent_frames"),
                "image_path": e.get("image_path", ""),
                "music_path": e.get("music_path", ""),
                "suno_id": e.get("suno_id", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Updated {updated} entries")
    print(f"Manifest: {manifest_path}")
    print(f"Annotations: {annotation_path}")


if __name__ == "__main__":
    main()
