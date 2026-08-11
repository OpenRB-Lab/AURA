"""Dataset for text-instruction MusicGen editing (no MuLan, no hidden states)."""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CACHE = PROJECT_ROOT / "data/edit_dataset/bridge_cache"
DIALOGUES = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"


def _instruction_for(e, dialogues) -> str | None:
    if e.get("template"):
        return e["template"].get("instruction")
    d = dialogues.get(e["dialogue_id"])
    if d is None:
        return None
    if e.get("step_index") is not None and d.get("steps"):
        return d["steps"][e["step_index"]].get("instruction")
    return d.get("edit_instruction")


class TextEditDataset(Dataset):
    def __init__(self, split: str, melodyflow_cap: float = 0.35,
                 limit: int | None = None):
        dialogues = {}
        for l in open(DIALOGUES):
            r = json.loads(l)
            dialogues[r["id"]] = r
        rows = []
        for l in open(CACHE / "examples.jsonl"):
            e = json.loads(l)
            if e["split"] != split:
                continue
            if not (CACHE / "encodec" / f"{e['example_id']}.pt").exists():
                continue
            instr = _instruction_for(e, dialogues)
            if not instr:
                continue
            e["instruction"] = instr
            rows.append(e)
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
        enc = torch.load(CACHE / "encodec" / f"{e['example_id']}.pt", weights_only=True)
        return {"instruction": e["instruction"], "src": enc["src"].long(),
                "tgt": enc["tgt"].long(), "n_frames": enc["n_frames"]}


def make_collate(tokenizer, max_len: int = 64):
    def collate(batch):
        tok = tokenizer([b["instruction"] for b in batch], padding=True,
                        truncation=True, max_length=max_len, return_tensors="pt")
        return {"text_ids": tok["input_ids"], "text_mask": tok["attention_mask"],
                "src": torch.stack([b["src"] for b in batch]),
                "tgt": torch.stack([b["tgt"] for b in batch]),
                "n_frames": torch.tensor([b["n_frames"] for b in batch])}
    return collate
