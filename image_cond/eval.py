"""Evaluation: retrieval metrics + caption-bridge baseline comparison.

Metrics:
  1. Retrieval R@{1,5,10}: projector(image) vs MuLan audio embeddings
  2. Caption-bridge baseline: image → CLIP text → compare to MuLan audio

Usage:
  python src/image_cond/eval.py \
      --cache-dir data/image_music/cached_latents \
      --adapter-dir ckpts/final \
      --config src/configs/image_cond.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFFRHYTHM_ROOT = PROJECT_ROOT / "src" / "DiffRhythm"
sys.path.insert(0, str(DIFFRHYTHM_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        raw = f.read()
    raw = raw.replace("${dataset_root}", str(PROJECT_ROOT / "data" / "image_music"))
    raw = raw.replace("${diffrhythm_root}", str(DIFFRHYTHM_ROOT))
    raw = raw.replace("${image_encoder.embed_dim}", "768")
    return yaml.safe_load(raw)


def compute_recall_at_k(query_embs, gallery_embs, ks=(1, 5, 10)):
    """Compute R@K retrieval. query[i] should match gallery[i]."""
    query_embs = F.normalize(query_embs, dim=-1)
    gallery_embs = F.normalize(gallery_embs, dim=-1)
    sim = query_embs @ gallery_embs.T  # [N, N]
    ranks = sim.argsort(dim=-1, descending=True)
    results = {}
    for k in ks:
        correct = 0
        for i in range(len(query_embs)):
            if i in ranks[i, :k]:
                correct += 1
        results[f"R@{k}"] = correct / len(query_embs)
    return results


@torch.no_grad()
def evaluate(cfg: dict, adapter_dir: str = None, device_str: str = "cuda"):
    device = torch.device(device_str)
    cache_dir = Path(cfg["cache_dir"])

    manifest = []
    with open(cache_dir / "index.jsonl") as f:
        for line in f:
            manifest.append(json.loads(line))

    print(f"Loading {len(manifest)} cached samples...")
    z_imgs = []
    z_mulan_audios = []
    for entry in tqdm(manifest, desc="Loading cache"):
        data = torch.load(entry["cache_path"], map_location="cpu")
        z_imgs.append(data["z_img"].float())
        z_mulan_audios.append(data["z_mulan_audio"].float())

    z_imgs = torch.stack(z_imgs).to(device)
    z_mulan_audios = torch.stack(z_mulan_audios).to(device)

    results = {}

    # Raw image embedding retrieval (no projector) — only if dims match
    if z_imgs.shape[-1] == z_mulan_audios.shape[-1]:
        raw_recall = compute_recall_at_k(z_imgs, z_mulan_audios)
        results["raw_clip_vs_mulan"] = raw_recall
        print(f"\nRaw CLIP → MuLan retrieval (no training): {raw_recall}")
    else:
        print(f"\nSkipping raw retrieval: dim mismatch ({z_imgs.shape[-1]} vs {z_mulan_audios.shape[-1]})")

    # Trained projector retrieval
    if adapter_dir:
        from src.image_cond.projector import ImageToMuLanProjector
        proj_cfg = cfg["projector"]
        projector = ImageToMuLanProjector(
            d_img=proj_cfg["d_img"], d_mulan=proj_cfg["d_mulan"],
            hidden_dim=proj_cfg["hidden_dim"], num_layers=proj_cfg["num_layers"],
            dropout=proj_cfg["dropout"], l2_normalize=proj_cfg["l2_normalize"],
        ).to(device)
        proj_state = torch.load(Path(adapter_dir) / "projector.pt", map_location=device)
        projector.load_state_dict(proj_state)
        projector.eval()

        z_proj = projector(z_imgs)  # [N, 512]
        proj_recall = compute_recall_at_k(z_proj, z_mulan_audios)
        results["trained_projector"] = proj_recall
        print(f"Trained projector → MuLan retrieval: {proj_recall}")

        cos_sim = F.cosine_similarity(z_proj, z_mulan_audios, dim=-1)
        results["mean_cosine_sim"] = cos_sim.mean().item()
        print(f"Mean cosine similarity (projected vs MuLan): {cos_sim.mean().item():.4f}")

    # Per-category breakdown
    categories = {}
    for i, entry in enumerate(manifest):
        cat = entry.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(i)

    print(f"\n{'Category':<15} {'Count':<8} {'Mean cos sim':<15}")
    print("-" * 40)
    for cat, indices in sorted(categories.items()):
        idx = torch.tensor(indices, device=device)
        if adapter_dir:
            cat_proj = z_proj[idx]
        else:
            cat_proj = z_imgs[idx]
        cat_mulan = z_mulan_audios[idx]
        cat_sim = F.cosine_similarity(cat_proj, cat_mulan, dim=-1).mean().item()
        print(f"{cat:<15} {len(indices):<8} {cat_sim:<15.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/image_cond.yaml")
    parser.add_argument("--adapter-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, adapter_dir=args.adapter_dir, device_str=args.device)


if __name__ == "__main__":
    main()
