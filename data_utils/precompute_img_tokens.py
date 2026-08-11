"""Add SigLIP image patch tokens (z_img_tokens) to cached .pt files for cross-attention."""

import json
import sys
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

    first_pt = torch.load(entries[0]["cache_path"], map_location="cpu", weights_only=True)
    if "z_img_tokens" in first_pt:
        print("z_img_tokens already exists. Skipping.")
        return

    print("Loading SigLIP...")
    from transformers import SiglipVisionModel, SiglipImageProcessor
    vision_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-256").to(device).eval()
    processor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-256")
    for p in vision_model.parameters():
        p.requires_grad_(False)

    updated = 0
    for entry in tqdm(entries, desc="Computing image tokens"):
        cache_path = entry["cache_path"]
        data = torch.load(cache_path, map_location="cpu", weights_only=True)

        image_path = entry.get("image_path", "")
        if image_path and Path(image_path).exists():
            with torch.no_grad():
                img = Image.open(image_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(device)
                out = vision_model(**inputs)
                data["z_img_tokens"] = out.last_hidden_state.squeeze(0).cpu().half()  # [256, 768]
        else:
            # Suno entries: use the pooled embedding repeated as a single token
            z_img = data["z_img"]  # [768]
            data["z_img_tokens"] = z_img.unsqueeze(0).half()  # [1, 768]

        torch.save(data, cache_path)
        updated += 1

    print(f"Updated {updated} cached files with z_img_tokens")


if __name__ == "__main__":
    main()
