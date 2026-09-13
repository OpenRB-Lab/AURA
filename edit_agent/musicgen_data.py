"""Dataset for the MusicGen edit bridge over the EnCodec + hidden caches."""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CACHE = PROJECT_ROOT / "data/edit_dataset/bridge_cache"


class MusicGenBridgeDataset(Dataset):
    def __init__(self, split: str, melodyflow_cap: float = 0.35,
                 limit: int | None = None, id_filter: str | None = None,
                 latent: bool = False, silence_filter: bool = False,
                 silence_max: float = 0.30, stem_mode_mix: float = 0.0):
        """stem_mode_mix: target max fraction of stem-mode rows (tgt = isolated
        stem, stem_row memory on). 0 disables; -1 = stem rows ONLY (val)."""
        self.latent = latent
        sil = {}
        if silence_filter and (CACHE / "silence_ratio.json").exists():
            sil = json.load(open(CACHE / "silence_ratio.json"))
        n_sil = 0
        rows = []
        for l in open(CACHE / "examples.jsonl"):
            e = json.loads(l)
            if e["split"] != split:
                continue
            if id_filter and id_filter not in e["example_id"].lower():
                continue
            if silence_filter and sil.get(e["example_id"], 0.0) > silence_max:
                n_sil += 1
                continue
            if latent and not (CACHE / "latent32" / f"{e['example_id']}.pt").exists():
                continue
            if not (CACHE / "encodec" / f"{e['example_id']}.pt").exists():
                continue
            if not (CACHE / "hidden" / f"{e['example_id']}.pt").exists():
                continue
            rows.append(e)

        creative = [r for r in rows if r["conv_type"].startswith(("text", "image", "game"))]
        exact = [r for r in rows if not r["conv_type"].startswith(("text", "image", "game"))]
        cap = int(len(exact) * melodyflow_cap / (1 - melodyflow_cap))
        if len(creative) > cap:
            rng = torch.Generator().manual_seed(0)
            idx = torch.randperm(len(creative), generator=rng)[:cap].tolist()
            creative = [creative[i] for i in idx]
        self.rows = exact + creative

        if stem_mode_mix != 0.0 and (CACHE / "stem_mode_ids.json").exists():
            ids = json.load(open(CACHE / "stem_mode_ids.json"))
            by_id = {r["example_id"]: r for r in rows}
            stem_rows = []
            for eid in ids["add_stem"]:
                if eid in by_id and (CACHE / "stem_encodec" / f"{eid}.pt").exists():
                    stem_rows.append({**by_id[eid], "_stem": "add"})
            for eid in ids["isolate"]:
                if eid in by_id:
                    stem_rows.append({**by_id[eid], "_stem": "isolate"})
            if stem_mode_mix < 0:
                self.rows = stem_rows
            else:
                cap = int(len(self.rows) * stem_mode_mix / (1 - stem_mode_mix))
                if len(stem_rows) > cap:
                    rng = torch.Generator().manual_seed(1)
                    idx = torch.randperm(len(stem_rows), generator=rng)[:cap].tolist()
                    stem_rows = [stem_rows[i] for i in idx]
                self.rows = self.rows + stem_rows
            print(f"[data] {split}: {len(stem_rows)} stem-mode rows "
                  f"({len(self.rows)} total)", flush=True)

        if limit:
            self.rows = self.rows[:limit]
        if silence_filter and n_sil:
            print(f"[data] silence filter dropped {n_sil} rows (> {silence_max:.0%})",
                  flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        e = self.rows[i]
        enc = torch.load(CACHE / "encodec" / f"{e['example_id']}.pt", weights_only=True)
        h = torch.load(CACHE / "hidden" / f"{e['example_id']}.pt", weights_only=True)["h"]
        out = {"h": h.float(), "src": enc["src"].long(), "tgt": enc["tgt"].long(),
               "n_frames": enc["n_frames"], "stem": "_stem" in e}
        if e.get("_stem") == "add":
            st = torch.load(CACHE / "stem_encodec" / f"{e['example_id']}.pt",
                            weights_only=True)
            out["tgt"] = st["tgt_stem"].long()
            out["n_frames"] = st["n_frames"]
        if self.latent:
            lat = torch.load(CACHE / "latent32" / f"{e['example_id']}.pt",
                             weights_only=True)
            out["z_tgt"] = lat["tgt"].float()
        return out


def collate(batch):
    out = {"h": torch.stack([b["h"] for b in batch]),
           "src": torch.stack([b["src"] for b in batch]),
           "tgt": torch.stack([b["tgt"] for b in batch]),
           "n_frames": torch.tensor([b["n_frames"] for b in batch]),
           "stem_mode": torch.tensor([b.get("stem", False) for b in batch])}
    if "z_tgt" in batch[0]:
        out["z_tgt"] = torch.stack([b["z_tgt"] for b in batch])
    return out
