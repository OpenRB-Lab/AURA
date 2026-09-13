"""Per-stem target cache for stem-mode bridge training.

For slakh `add` pairs the render is linear with no gains and a shared scale,
so the added-stems-only audio is EXACTLY target - input (both lossless FLAC;
the 44.1k->32k resample is linear and commutes with the subtraction). We load
the same 10 s window precompute_encodec.py used, subtract, and EnCodec-encode.

Output: data/edit_dataset/bridge_cache/stem_encodec/<example_id>.pt
  {"tgt_stem": int16 [4,500], "n_frames": int}
plus stem_mode_ids.json listing usable example ids:
  {"add_stem": [...ids with a stem_encodec cache...],
   "isolate": [...ids whose existing tgt is already stem-only...]}

Pairs whose target FLAC clips (|x| >= 0.999 on >0.1% of samples) are skipped —
clipping breaks the linearity identity.

Usage:
  conda run -n llama python -u src/edit_agent/build_stem_targets.py [--verify 5]
"""

import argparse
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
OUT = CACHE / "stem_encodec"
PAIRS = PROJECT_ROOT / "data/edit_dataset/slakh/slakh_pairs.jsonl"

SR = 32000
WIN_S = 10.0
FRAME_RATE = 50
N_FRAMES = int(WIN_S * FRAME_RATE)  # 500
CLIP_FRAC = 1e-3


def pair_id_of(e: dict) -> str | None:
    p = Path(e["tgt_path"])
    if "slakh/pairs" not in e["tgt_path"] or not p.stem.endswith("_out"):
        return None
    return p.stem[: -len("_out")]


class DiffDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import librosa
        e = self.rows[i]
        try:
            dur = librosa.get_duration(path=str(PROJECT_ROOT / e["src_path"]))
            start = max((dur - WIN_S) / 2, 0)  # slakh rows have no segment
            clips = []
            for p in (e["src_path"], e["tgt_path"]):
                y, _ = librosa.load(PROJECT_ROOT / p, sr=SR, mono=True,
                                    offset=start, duration=WIN_S)
                n = len(y)
                if n < int(WIN_S * SR):
                    y = np.pad(y, (0, int(WIN_S * SR) - n))
                clips.append(y.astype("float32"))
            src, tgt = clips
            if np.mean(np.abs(tgt) >= 0.999) > CLIP_FRAC:
                return {"id": e["example_id"], "err": "clipped", "ok": False}
            diff = tgt - src
            if np.sqrt((diff ** 2).mean()) < 1e-4:
                return {"id": e["example_id"], "err": "silent diff", "ok": False}
            n_frames = min(N_FRAMES, max(1, int(n / SR * FRAME_RATE)))
            return {"id": e["example_id"], "diff": torch.from_numpy(diff),
                    "n_frames": n_frames, "ok": True}
        except Exception as exc:  # noqa: BLE001
            return {"id": e["example_id"], "err": str(exc), "ok": False}


def verify(rows, n):
    """Raw-stem render check: diff(target,input) == render of the added stems."""
    import random

    import librosa
    import soundfile as sf
    import yaml

    from dataset_generation.slakh_edits import render, stem_display_names

    pairs = {}
    for l in open(PAIRS):
        r = json.loads(l)
        pairs[r["pair_id"]] = r
    roots = list((PROJECT_ROOT / "data/edit_dataset/slakh_raw"
                  / "slakh2100_flac_redux").glob("*"))
    rng = random.Random(0)
    for e in rng.sample(rows, min(n, len(rows))):
        pid = pair_id_of(e)
        r = pairs[pid]
        track_dir = next(rt / r["track"] for rt in roots
                         if (rt / r["track"] / "metadata.yaml").exists())
        meta = yaml.safe_load(open(track_dir / "metadata.yaml"))
        names = stem_display_names(meta)
        name_to_sid = {v: k for k, v in names.items()}
        added = [name_to_sid[x] for x in r["target_stems"]
                 if x not in set(r["input_stems"])]
        stems_audio, sr = {}, None
        for sid in added + [s for s in meta["stems"]]:
            for ext in (".flac", ".wav"):
                p = track_dir / "stems" / f"{sid}{ext}"
                if p.exists():
                    y, sr = sf.read(p, dtype="float32")
                    if y.ndim > 1:
                        y = y.mean(axis=1)
                    stems_audio[sid] = y
                    break
        full = np.sum([v[: min(len(x) for x in stems_audio.values())]
                       for v in stems_audio.values()], axis=0)
        s0, s1 = int(r["start_s"] * sr), int(r["end_s"] * sr)
        scale = 0.9 / np.abs(full[s0:s1]).max()
        ref = render(stems_audio, added, {}, scale, s0, s1)
        inp, _ = sf.read(PROJECT_ROOT / r["input_path"], dtype="float32")
        tgt, _ = sf.read(PROJECT_ROOT / r["target_path"], dtype="float32")
        diff = (tgt - inp).mean(axis=1)
        L = min(len(ref), len(diff))
        err = np.abs(ref[:L] - diff[:L]).max()
        print(f"[verify] {pid}: added={len(added)} stems, "
              f"max|render-diff|={err:.2e} {'OK' if err < 1e-3 else 'FAIL'}",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", type=int, default=0,
                    help="raw-stem render check on N random add pairs, then exit")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    op_of = {}
    for l in open(PAIRS):
        r = json.loads(l)
        op_of[r["pair_id"]] = r["op"]

    add_rows, isolate_ids = [], []
    for l in open(CACHE / "examples.jsonl"):
        e = json.loads(l)
        pid = pair_id_of(e)
        if pid is None or pid not in op_of:
            continue
        if op_of[pid] == "add":
            add_rows.append(e)
        elif op_of[pid] == "isolate":
            isolate_ids.append(e["example_id"])
    print(f"stem targets: {len(add_rows)} add examples, "
          f"{len(isolate_ids)} isolate examples", flush=True)

    if args.verify:
        verify(add_rows, args.verify)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    pending = [e for e in add_rows
               if not (OUT / f"{e['example_id']}.pt").exists()
               and (CACHE / "encodec" / f"{e['example_id']}.pt").exists()]
    print(f"{len(pending)} pending", flush=True)

    if pending:
        from transformers import MusicgenForConditionalGeneration
        enc = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-medium", cache_dir=str(PROJECT_ROOT / "weights"),
            torch_dtype=torch.float32).audio_encoder.to(args.device).eval()

        loader = DataLoader(DiffDataset(pending), batch_size=16, num_workers=12,
                            collate_fn=lambda b: b)
        done = skipped = 0
        with torch.no_grad():
            for batch in loader:
                good = [b for b in batch if b["ok"]]
                skipped += len(batch) - len(good)
                if good:
                    wavs = torch.stack([b["diff"] for b in good]) \
                        .unsqueeze(1).to(args.device)
                    codes = enc.encode(wavs, bandwidth=None).audio_codes
                    codes = codes[0] if codes.dim() == 4 else codes
                    for j, b in enumerate(good):
                        torch.save({"tgt_stem": codes[j].short().cpu(),
                                    "n_frames": b["n_frames"]},
                                   OUT / f"{b['id']}.pt")
                done += len(batch)
                if done % 1600 < 16:
                    print(f"stem_encodec {done}/{len(pending)} "
                          f"({skipped} skipped)", flush=True)
        print(f"stem_encodec done ({skipped} skipped)", flush=True)

    cached = {p.stem for p in OUT.glob("*.pt")}
    ids = {"add_stem": sorted(e["example_id"] for e in add_rows
                              if e["example_id"] in cached),
           "isolate": sorted(isolate_ids)}
    json.dump(ids, open(CACHE / "stem_mode_ids.json", "w"))
    print(f"stem_mode_ids.json: {len(ids['add_stem'])} add_stem, "
          f"{len(ids['isolate'])} isolate", flush=True)


if __name__ == "__main__":
    main()
