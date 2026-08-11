"""Dataset + collation for bridge training over the precomputed caches."""

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CACHE = PROJECT_ROOT / "data/edit_dataset/bridge_cache"
DIALOGUES = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"

FPS = 44100 / 2048
MAX_FRAMES = 2048


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


class BridgeDataset(Dataset):
    def __init__(self, split: str, melodyflow_cap: float = 0.35,
                 limit: int | None = None):
        segments = {}
        for l in open(DIALOGUES):
            r = json.loads(l)
            if r.get("segment"):
                segments[r["id"]] = r["segment"]

        rows = []
        for l in open(CACHE / "examples.jsonl"):
            e = json.loads(l)
            if e["split"] != split:
                continue
            if not (CACHE / "hidden" / f"{e['example_id']}.pt").exists():
                continue
            if not (CACHE / "latents" / f"{md5(e['src_path'])}.pt").exists():
                continue
            if not (CACHE / "latents" / f"{md5(e['tgt_path'])}.pt").exists():
                continue
            if not (CACHE / "mulan" / f"{md5(e['tgt_path'])}.pt").exists():
                continue
            e["segment"] = segments.get(e["dialogue_id"])
            rows.append(e)

        # keep the exact tiers dominant: cap creative (MelodyFlow-tier) examples
        creative = [r for r in rows if r["conv_type"].startswith(("text", "image", "game"))]
        exact = [r for r in rows if not r["conv_type"].startswith(("text", "image", "game"))]
        cap = int(len(exact) * melodyflow_cap / (1 - melodyflow_cap))
        if len(creative) > cap:
            rng = torch.Generator().manual_seed(0)
            idx = torch.randperm(len(creative), generator=rng)[:cap].tolist()
            creative = [creative[i] for i in idx]
        self.rows = exact + creative
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        e = self.rows[i]
        src = torch.load(CACHE / "latents" / f"{md5(e['src_path'])}.pt", weights_only=True)
        tgt = torch.load(CACHE / "latents" / f"{md5(e['tgt_path'])}.pt", weights_only=True)
        h = torch.load(CACHE / "hidden" / f"{e['example_id']}.pt", weights_only=True)["h"]
        mulan = torch.load(CACHE / "mulan" / f"{md5(e['tgt_path'])}.pt", weights_only=True)
        t = min(src["latent"].shape[0], tgt["latent"].shape[0])
        seg = None
        if e.get("segment"):
            s, en = e["segment"]
            seg = (int(s * FPS), min(int(en * FPS), t))
        return {"src": src["latent"][:t], "tgt": tgt["latent"][:t], "h": h,
                "mulan": mulan, "n_frames": t, "segment": seg}


def collate(batch):
    b = len(batch)
    t_max = max(x["n_frames"] for x in batch)
    src = torch.zeros(b, t_max, 64, dtype=torch.half)
    tgt = torch.zeros(b, t_max, 64, dtype=torch.half)
    pred_mask = torch.zeros(b, t_max, dtype=torch.bool)
    lens = torch.zeros(b, dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["n_frames"]
        src[i, :n] = x["src"]
        tgt[i, :n] = x["tgt"]
        lens[i] = n
        if x["segment"] is not None:
            s, e = x["segment"]
            pred_mask[i, s:e] = True
        else:
            pred_mask[i, :n] = True
    h = torch.stack([x["h"] for x in batch])       # [B, 9, 3584] fp16
    mulan = torch.stack([x["mulan"] for x in batch])
    return {"src": src, "tgt": tgt, "h": h, "mulan": mulan,
            "lens": lens, "pred_mask": pred_mask}
