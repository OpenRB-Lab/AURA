"""Silence ratio of the SOURCE 10 s window for every Slakh bridge example.

silent frame = RMS < -40 dBFS over 50 ms hops; ratio = silent/total.
Writes data/edit_dataset/bridge_cache/silence_ratio.json {example_id: ratio}.
Training drops rows with ratio > 0.30 via --silence-filter.

Usage: conda run -n llama python -u src/edit_agent/filter_silence.py [--procs 12]
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CACHE = PROJECT_ROOT / "data/edit_dataset/bridge_cache"
OUT = CACHE / "silence_ratio.json"

SR = 24000
HOP = int(0.05 * SR)
DB_FLOOR = -40.0


def ratio_for(item):
    example_id, src_path, win = item
    import librosa
    try:
        y, _ = librosa.load(PROJECT_ROOT / src_path, sr=SR, mono=True,
                            offset=win[0], duration=win[1] - win[0])
        if len(y) < HOP:
            return example_id, 1.0
        n = len(y) // HOP
        fr = y[: n * HOP].reshape(n, HOP)
        rms = np.sqrt((fr ** 2).mean(axis=1))
        peak = np.abs(y).max()
        if peak < 1e-4:
            return example_id, 1.0
        db = 20 * np.log10(rms / peak + 1e-10)
        return example_id, float((db < DB_FLOOR).mean())
    except Exception:  # noqa: BLE001
        return example_id, 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()

    done = json.load(open(OUT)) if OUT.exists() else {}
    items = []
    for line in open(CACHE / "examples.jsonl"):
        e = json.loads(line)
        if "slakh" not in e["example_id"].lower() or e["example_id"] in done:
            continue
        enc_p = CACHE / "encodec" / f"{e['example_id']}.pt"
        if not enc_p.exists():
            continue
        enc = torch.load(enc_p, weights_only=True)
        items.append((e["example_id"], e["src_path"], enc["win"]))
    print(f"{len(items)} slakh examples to score ({len(done)} cached)", flush=True)

    with ProcessPoolExecutor(max_workers=args.procs) as pool:
        futs = [pool.submit(ratio_for, it) for it in items]
        for i, fut in enumerate(as_completed(futs)):
            eid, r = fut.result()
            done[eid] = r
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(items)}", flush=True)
                json.dump(done, open(OUT, "w"))
    json.dump(done, open(OUT, "w"))
    ratios = np.array(list(done.values()))
    print(f"total scored {len(done)}; >0.30 silent: {(ratios > 0.30).sum()} "
          f"({(ratios > 0.30).mean() * 100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
