"""EnCodec-code cache for the MusicGen bridge (Instruct-MusicGen style).

Per bridge example: a 10 s window (segment-centered for localized inp_* pairs,
center-of-file otherwise), applied identically to src and tgt audio, encoded with
MusicGen's EnCodec (32 kHz, 4 codebooks, 50 Hz) -> int16 codes [4, 500] each.

Output: data/edit_dataset/bridge_cache/encodec/<example_id>.pt
  {"src": int16 [4,500], "tgt": int16 [4,500], "n_frames": int (valid, <=500),
   "win": [start_s, end_s], "seg_frames": [s,e] | None (segment within window)}

Resumable; run:  conda run -n llama python -u src/edit_agent/precompute_encodec.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

torch.backends.cudnn.enabled = False

CACHE = PROJECT_ROOT / "data/edit_dataset/bridge_cache"
OUT = CACHE / "encodec"
DIALOGUES = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"

SR = 32000
WIN_S = 10.0
FRAME_RATE = 50
N_FRAMES = int(WIN_S * FRAME_RATE)  # 500


class ClipDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import librosa
        e = self.rows[i]
        try:
            dur = librosa.get_duration(path=str(PROJECT_ROOT / e["src_path"]))
            seg = e.get("segment")
            if seg:
                mid = (seg[0] + seg[1]) / 2
                start = min(max(mid - WIN_S / 2, 0), max(dur - WIN_S, 0))
            else:
                start = max((dur - WIN_S) / 2, 0)
            clips = []
            for p in (e["src_path"], e["tgt_path"]):
                y, _ = librosa.load(PROJECT_ROOT / p, sr=SR, mono=True,
                                    offset=start, duration=WIN_S)
                n = len(y)
                if n < int(WIN_S * SR):
                    y = np.pad(y, (0, int(WIN_S * SR) - n))
                clips.append(torch.from_numpy(y.astype("float32")))
            n_frames = min(N_FRAMES, max(1, int(n / SR * FRAME_RATE)))
            seg_frames = None
            if seg:
                s = max(0, int((seg[0] - start) * FRAME_RATE))
                en = min(N_FRAMES, int((seg[1] - start) * FRAME_RATE))
                if en > s:
                    seg_frames = [s, en]
            return {"id": e["example_id"], "src": clips[0], "tgt": clips[1],
                    "n_frames": n_frames, "win": [start, start + WIN_S],
                    "seg_frames": seg_frames, "ok": True}
        except Exception as exc:  # noqa: BLE001
            return {"id": e["example_id"], "err": str(exc), "ok": False}


def collate(batch):
    return batch


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    segments = {}
    for l in open(DIALOGUES):
        r = json.loads(l)
        if r.get("segment"):
            segments[r["id"]] = r["segment"]

    rows = []
    for l in open(CACHE / "examples.jsonl"):
        e = json.loads(l)
        if (OUT / f"{e['example_id']}.pt").exists():
            continue
        if not (CACHE / "hidden" / f"{e['example_id']}.pt").exists():
            continue
        e["segment"] = segments.get(e["dialogue_id"])
        rows.append(e)
    print(f"encodec: {len(rows)} examples pending", flush=True)
    if not rows:
        return

    from transformers import MusicgenForConditionalGeneration
    enc = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-medium", cache_dir=str(PROJECT_ROOT / "weights"),
        torch_dtype=torch.float32).audio_encoder.to("cuda").eval()

    loader = DataLoader(ClipDataset(rows), batch_size=16, num_workers=12,
                        collate_fn=collate)
    done = 0
    with torch.no_grad():
        for batch in loader:
            good = [b for b in batch if b["ok"]]
            for b in batch:
                if not b["ok"]:
                    print(f"[error] {b['id']}: {b['err']}", flush=True)
            if good:
                wavs = torch.stack([b["src"] for b in good] +
                                   [b["tgt"] for b in good]).unsqueeze(1).to("cuda")
                codes = enc.encode(wavs, bandwidth=None).audio_codes  # [1,2N,4,T] or [chunks,...]
                codes = codes[0] if codes.dim() == 4 else codes
                n = len(good)
                for j, b in enumerate(good):
                    torch.save({"src": codes[j].short().cpu(),
                                "tgt": codes[n + j].short().cpu(),
                                "n_frames": b["n_frames"], "win": b["win"],
                                "seg_frames": b["seg_frames"]},
                               OUT / f"{b['id']}.pt")
            done += len(batch)
            if done % 1600 < 16:
                print(f"encodec {done}/{len(rows)}", flush=True)
    print("encodec done", flush=True)


if __name__ == "__main__":
    main()
