"""Fix 131 painting entries that got empty prompts after cleanup.

These had thinking-leak text_prompts but no tags to fall back on.
Strategy: extract mood/instruments/tempo from the reasoning text + category,
then construct a usable music generation prompt.
"""

import json
import re
from pathlib import Path

CATEGORY_TEMPLATES = {
    "安静平和": "Generate a serene, contemplative piece of traditional East Asian ambient music. The mood should be {moods}. Use a {tempo} tempo. Feature {instruments}. The atmosphere should feel vast, misty, and ancient.",
    "悲伤孤傲": "Generate a melancholic, introspective piece of cinematic instrumental music. The mood should be {moods}. Use a {tempo} tempo. Feature {instruments}. The atmosphere should feel lonely, emotional, and deeply expressive.",
    "活泼欢快": "Generate a lively, cheerful piece of instrumental music. The mood should be {moods}. Use a {tempo} tempo. Feature {instruments}. The atmosphere should feel bright, festive, and energetic.",
    "激昂肆意": "Generate an intense, passionate piece of dramatic orchestral music. The mood should be {moods}. Use a {tempo} tempo. Feature {instruments}. The atmosphere should feel powerful, bold, and triumphant.",
}

CATEGORY_DEFAULTS = {
    "安静平和": {
        "moods": "tranquil, meditative, and majestic",
        "tempo": "slow",
        "instruments": "Guzheng, Shakuhachi flute, soft strings, and ambient pads",
    },
    "悲伤孤傲": {
        "moods": "sorrowful, contemplative, and deeply emotional",
        "tempo": "slow",
        "instruments": "solo Erhu, piano, deep cello, and atmospheric strings",
    },
    "活泼欢快": {
        "moods": "joyful, playful, and uplifting",
        "tempo": "moderate to fast",
        "instruments": "bright flute, plucked strings, light percussion, and cheerful woodwinds",
    },
    "激昂肆意": {
        "moods": "dramatic, fierce, and exhilarating",
        "tempo": "fast",
        "instruments": "bold brass, driving drums, powerful strings, and epic orchestral elements",
    },
}


def extract_from_thinking(text):
    instruments = list(set(re.findall(
        r'(Guzheng|Erhu|Shakuhachi|Dizi|Koto|flute|piano|strings|drums|percussion|guitar|violin|cello|harp|brass|trumpet|bamboo|gong|taiko|pipa|xiao|zither)',
        text, re.IGNORECASE)))

    moods = list(set(m.lower() for m in re.findall(
        r'(serene|tranquil|melancholic|dramatic|energetic|joyful|peaceful|contemplative|majestic|ethereal|somber|dark|bright|upbeat|solemn|ancient|meditative|passionate|sorrowful|lonely|playful|fierce)',
        text, re.IGNORECASE)))

    tempo = "slow"
    if re.search(r'\bfast\b', text, re.IGNORECASE):
        tempo = "fast"
    elif re.search(r'\bmoderate\b', text, re.IGNORECASE):
        tempo = "moderate"

    return instruments[:4], moods[:4], tempo


def build_prompt(category, text):
    template = CATEGORY_TEMPLATES.get(category)
    defaults = CATEGORY_DEFAULTS.get(category)
    if not template or not defaults:
        return f"Generate instrumental music in a {category} style."

    instruments, moods, tempo = extract_from_thinking(text)

    mood_str = ", ".join(moods) if moods else defaults["moods"]
    tempo_str = tempo or defaults["tempo"]
    instr_str = ", ".join(instruments) if instruments else defaults["instruments"]

    return template.format(moods=mood_str, tempo=tempo_str, instruments=instr_str)


def main():
    index_path = Path("data/image_music/cached_latents/index.jsonl")
    backup_path = index_path.with_suffix(".jsonl.bak")

    with open(index_path) as f:
        entries = [json.loads(l) for l in f]

    with open(backup_path) as f:
        originals = {json.loads(l)["idx"]: json.loads(l) for l in f}

    fixed = 0
    for entry in entries:
        if entry.get("text_prompt", "").strip():
            continue

        idx = entry["idx"]
        orig = originals.get(idx, {})
        orig_text = orig.get("text_prompt", "")
        category = entry.get("category", "")

        if orig_text and category:
            prompt = build_prompt(category, orig_text)
            entry["text_prompt"] = prompt
            entry["_prompt_cleaned"] = "rebuilt_from_thinking"
            fixed += 1

    with open(index_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Fixed {fixed} empty prompts")

    # Verify
    with open(index_path) as f:
        entries = [json.loads(l) for l in f]
    still_empty = sum(1 for e in entries if not e.get("text_prompt", "").strip())
    print(f"Still empty: {still_empty}")

    # Show examples
    print("\n--- Examples of rebuilt prompts ---")
    count = 0
    for e in entries:
        if e.get("_prompt_cleaned") == "rebuilt_from_thinking":
            print(f"  idx={e['idx']} ({e.get('category','?')}):")
            print(f"    {e['text_prompt'][:200]}")
            print()
            count += 1
            if count >= 4:
                break


if __name__ == "__main__":
    main()
