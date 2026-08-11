"""Precompute MuLan text embeddings from text_prompt field and add to cached .pt files."""

import json
import sys
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFRHYTHM_ROOT = PROJECT_ROOT / "src" / "DiffRhythm"
sys.path.insert(0, str(DIFFRHYTHM_ROOT))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest_path = PROJECT_ROOT / "data/image_music/cached_latents/index.jsonl"

    with open(manifest_path) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    print(f"Total entries: {len(entries)}")
    has_text = sum(1 for e in entries if e.get("text_prompt"))
    print(f"With text_prompt: {has_text}")

    # Check how many already have z_mulan_text cached
    first_pt = torch.load(entries[0]["cache_path"], map_location="cpu", weights_only=True)
    if "z_mulan_text" in first_pt:
        print("z_mulan_text already exists in cached files. Skipping.")
        return

    print("Loading MuQ-MuLan...")
    from muq import MuQMuLan
    pretrained_cache = str(PROJECT_ROOT / "weights")
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large",
                                      cache_dir=pretrained_cache).to(device).eval()

    updated = 0
    for entry in tqdm(entries, desc="Computing text embeddings"):
        cache_path = entry["cache_path"]
        text_prompt = entry.get("text_prompt", "")

        data = torch.load(cache_path, map_location="cpu", weights_only=True)

        if text_prompt:
            with torch.no_grad():
                z_text = mulan(texts=text_prompt)  # [1, 512]
            data["z_mulan_text"] = z_text.squeeze(0).cpu().half()
        else:
            data["z_mulan_text"] = torch.zeros(512, dtype=torch.float16)

        torch.save(data, cache_path)
        updated += 1

    print(f"Updated {updated} cached files with z_mulan_text")


if __name__ == "__main__":
    main()
