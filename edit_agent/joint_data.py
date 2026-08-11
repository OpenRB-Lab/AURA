"""Dataset for JOINT thinker + MusicGen-bridge training.

Two row kinds:
- edit rows: bridge example (dialogue or tpl_slakh template context) with EnCodec
  codes -> LM loss on assistant tokens + MusicGen CE through the live hidden states
- qa rows: no-edit dialogues (music Q&A / chat) -> LM loss only, keeps the
  assistant able to talk about the music without emitting edits
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.tokens import typed_block  # noqa: E402

CACHE = PROJECT_ROOT / "data/edit_dataset/bridge_cache"
DIALOGUES = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"


class JointDataset(Dataset):
    def __init__(self, split: str, melodyflow_cap: float = 0.35,
                 limit: int | None = None):
        self.dialogues = {}
        qa_rows = []
        for l in open(DIALOGUES):
            r = json.loads(l)
            self.dialogues[r["id"]] = r
            if r["split"] == split and not r["has_edit"]:
                qa_rows.append({"kind": "qa", "dialogue_id": r["id"],
                                "conv_type": r["conv_type"]})

        rows = []
        for l in open(CACHE / "examples.jsonl"):
            e = json.loads(l)
            if e["split"] != split:
                continue
            if not (CACHE / "encodec" / f"{e['example_id']}.pt").exists():
                continue
            e["kind"] = "edit"
            rows.append(e)
        creative = [r for r in rows if r["conv_type"].startswith(("text", "image", "game"))]
        exact = [r for r in rows if not r["conv_type"].startswith(("text", "image", "game"))]
        cap = int(len(exact) * melodyflow_cap / (1 - melodyflow_cap))
        if len(creative) > cap:
            rng = torch.Generator().manual_seed(0)
            idx = torch.randperm(len(creative), generator=rng)[:cap].tolist()
            creative = [creative[i] for i in idx]
        self.rows = exact + creative + qa_rows
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        e = self.rows[i]
        if e["kind"] == "qa":
            d = self.dialogues[e["dialogue_id"]]
            return {"kind": "qa", "messages": d["messages"],
                    "chunk_path": d["chunk_path"], "image_path": d.get("image_path"),
                    "enc": None}
        if e.get("template"):
            tpl = e["template"]
            msgs = [
                {"role": "user", "content": [
                    {"type": "audio", "audio": e["chunk_path"]},
                    {"type": "text", "text": tpl["instruction"]}]},
                {"role": "assistant", "content": [
                    {"type": "text",
                     "text": f"I've applied that edit. {typed_block(tpl['op'])}"}]},
            ]
            chunk = e["chunk_path"]
        else:
            d = self.dialogues[e["dialogue_id"]]
            msgs = d["messages"]
            if e["step_index"] is not None:
                msgs = msgs[: 2 * e["step_index"] + 2]  # through assistant turn i
            chunk = d["chunk_path"]
        enc = torch.load(CACHE / "encodec" / f"{e['example_id']}.pt", weights_only=True)
        z_tgt = None
        lat_p = CACHE / "latent32" / f"{e['example_id']}.pt"
        if lat_p.exists():
            z_tgt = torch.load(lat_p, weights_only=True)["tgt"].float()
        return {"kind": "edit", "messages": msgs, "chunk_path": chunk,
                "image_path": e.get("image_path"), "enc": enc, "z_tgt": z_tgt}
