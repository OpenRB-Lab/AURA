"""Dataset + collation for SFT of the Omni thinker on edit dialogues.

Each dialogue is rendered through the processor's chat template with its chunk audio
(and optional reference image). Labels are -100 everywhere except assistant-turn
content, which is located by scanning PROCESSED token ids for the
`<|im_start|>assistant` marker — robust to audio/image placeholder expansion because
assistant turns are pure text.
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.qwen_wrapper import load_audio_16k  # noqa: E402

SYSTEM_PROMPT = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                 "Group, capable of perceiving auditory and visual inputs, as well as "
                 "generating text and speech.")


class DialogueDataset(Dataset):
    def __init__(self, dialogues_path: Path, split: str, limit: int | None = None):
        self.rows = []
        with open(dialogues_path) as f:
            for line in f:
                r = json.loads(line)
                if r["split"] == split:
                    self.rows.append(r)
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        messages += r["messages"]
        audio = load_audio_16k(r["chunk_path"])
        image = None
        if r.get("image_path"):
            from PIL import Image
            image = Image.open(PROJECT_ROOT / r["image_path"]).convert("RGB")
        return {"messages": messages, "audio": audio, "image": image, "id": r["id"]}


def _assistant_label_mask(input_ids: torch.Tensor, tok) -> torch.Tensor:
    """True on tokens belonging to assistant responses (content + trailing im_end)."""
    marker = tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    ids = input_ids.tolist()
    m = len(marker)
    i = 0
    while i <= len(ids) - m:
        if ids[i:i + m] == marker:
            j = i + m
            while j < len(ids) and ids[j] != im_end:
                j += 1
            mask[i + m: min(j + 1, len(ids))] = True  # content + im_end
            i = j
        i += 1
    return mask


def make_collate(processor):
    tok = processor.tokenizer

    def collate(batch):
        texts, audios, images = [], [], []
        for ex in batch:
            texts.append(processor.apply_chat_template(ex["messages"], tokenize=False,
                                                       add_generation_prompt=False))
            audios.append(ex["audio"])
            if ex["image"] is not None:
                images.append(ex["image"])
        inputs = processor(text=texts, audio=audios, images=images or None,
                           return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()
        for b in range(labels.shape[0]):
            keep = _assistant_label_mask(inputs["input_ids"][b], tok)
            keep &= inputs["attention_mask"][b].bool()
            labels[b][~keep] = -100
        inputs["labels"] = labels
        return inputs

    return collate
