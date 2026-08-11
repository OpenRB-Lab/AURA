"""Instruct-MusicGen-style benchmark from the Slakh2100 TEST split.

Replicates the slakh_datamodule.py protocol of arXiv 2405.18386:
- 4-stem taxonomy (Drums/Bass/Piano/Guitar) via metadata `inst_class`
- tasks add / remove / extract with their exact set logic
- instruction templates "Music piece. Instruct: Add {x}." / "No {x}." / "Only {x}."
- 5 s windows, 32 kHz mono, peak-normalized (shared scale), quality filters:
  input peak >= 0.1, target-difference silence <= 70%, 10 retries

Output: results/impg_bench/{task}/{input,ground_truth,instruction}/{i:03d}.*
        + manifest.json

Usage: conda run -n llama python -u src/edit_agent/build_impg_benchmark.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TEST_ROOT = PROJECT_ROOT / "data/edit_dataset/slakh_raw/slakh2100_flac_redux/test"
OUT = PROJECT_ROOT / "results/impg_bench"

SR = 32000
WIN_S = 5.0
N_PER_TASK = 100
TARGET_CLASSES = ["Drums", "Bass", "Piano", "Guitar"]
ACTIVE_RMS = 2e-3
TASKS = ["add", "remove", "extract"]
TEMPLATE = {"add": "Music piece. Instruct: Add {x}.",
            "remove": "Music piece. Instruct: No {x}.",
            "extract": "Music piece. Instruct: Only {x}."}


def load_track(track_dir: Path):
    """-> (class_audio: {inst_class: mono float32 @32k}, dur_s) or None."""
    import librosa
    meta = yaml.safe_load(open(track_dir / "metadata.yaml"))
    per_class: dict[str, list[np.ndarray]] = {}
    sr = None
    for sid, info in meta["stems"].items():
        if not info.get("audio_rendered", True):
            continue
        p = track_dir / "stems" / f"{sid}.flac"
        if not p.exists():
            continue
        y, sr = sf.read(p, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        cls = (info.get("inst_class") or "other").strip().title()
        per_class.setdefault(cls, []).append(y)
    if not per_class:
        return None
    n = min(len(y) for ys in per_class.values() for y in ys)
    class_audio = {}
    for cls, ys in per_class.items():
        s = np.zeros(n, dtype=np.float32)
        for y in ys:
            s += y[:n]
        class_audio[cls] = s
    if sr != SR:
        class_audio = {c: librosa.resample(y, orig_sr=sr, target_sr=SR)
                       for c, y in class_audio.items()}
    n32 = min(len(y) for y in class_audio.values())
    class_audio = {c: y[:n32] for c, y in class_audio.items()}
    return class_audio, n32 / SR


def silence_frac(y, hop=1600):
    n = len(y) // hop
    if n == 0:
        return 1.0
    fr = y[: n * hop].reshape(n, hop)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    peak = np.abs(y).max()
    if peak < 1e-5:
        return 1.0
    return float((rms < 10 ** (-40 / 20) * peak).mean())


def sample_example(class_audio, dur, task, rng):
    win = int(WIN_S * SR)
    total = int(dur * SR)
    if total <= win:
        return None
    for _ in range(10):
        s0 = rng.randrange(0, total - win)
        seg = {c: y[s0:s0 + win] for c, y in class_audio.items()}
        active = [c for c, y in seg.items()
                  if np.sqrt((y ** 2).mean()) > ACTIVE_RMS]
        targets = [c for c in TARGET_CLASSES if c in active]
        others = [c for c in active if c not in TARGET_CLASSES] + \
                 [c for c in TARGET_CLASSES if c in active]
        if not targets:
            continue
        tgt_cls = rng.choice(targets)
        rest = [c for c in active if c != tgt_cls]
        if not rest:
            continue
        tgt = seg[tgt_cls]
        rest_mix = np.sum([seg[c] for c in rest], axis=0)
        if task == "add":
            inp, gt = rest_mix, rest_mix + tgt
        elif task == "remove":
            inp, gt = rest_mix + tgt, rest_mix
        else:  # extract
            inp, gt = rest_mix + tgt, tgt
        if np.abs(inp).max() < 0.1:
            continue
        if silence_frac(gt - inp if task == "add" else inp - gt) > 0.70:
            continue
        scale = 0.9 / max(np.abs(inp).max(), np.abs(gt).max())
        return {"input": np.clip(inp * scale, -1, 1),
                "gt": np.clip(gt * scale, -1, 1),
                "target_class": tgt_cls, "offset_s": s0 / SR}
    return None


def main():
    rng = random.Random(42)
    tracks = sorted(p for p in TEST_ROOT.glob("Track*")
                    if (p / "metadata.yaml").exists())
    print(f"{len(tracks)} test tracks", flush=True)
    counts = {t: 0 for t in TASKS}
    cls_counts: dict[str, int] = {}
    manifest = []
    for task in TASKS:
        for sub in ["input", "ground_truth", "instruction"]:
            (OUT / task / sub).mkdir(parents=True, exist_ok=True)

    ti = 0
    loaded: dict[str, tuple] = {}
    attempts = 0
    while any(counts[t] < N_PER_TASK for t in TASKS) and attempts < 3000:
        attempts += 1
        track = tracks[ti % len(tracks)]
        ti += 1
        task = min(TASKS, key=lambda t: counts[t])
        if counts[task] >= N_PER_TASK:
            continue
        if track.name not in loaded:
            try:
                loaded[track.name] = load_track(track)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {track.name}: {exc}", flush=True)
                loaded[track.name] = None
            if len(loaded) % 30 == 0:
                print(f"...{len(loaded)} tracks loaded, counts {counts}", flush=True)
        if loaded[track.name] is None:
            continue
        class_audio, dur = loaded[track.name]
        ex = sample_example(class_audio, dur, task, rng)
        if ex is None:
            continue
        i = counts[task]
        instr = TEMPLATE[task].format(x=ex["target_class"].lower())
        sf.write(OUT / task / "input" / f"{i:03d}.wav", ex["input"], SR,
                 subtype="PCM_16")
        sf.write(OUT / task / "ground_truth" / f"{i:03d}.wav", ex["gt"], SR,
                 subtype="PCM_16")
        (OUT / task / "instruction" / f"{i:03d}.txt").write_text(
            f"{instr}\ntarget: {ex['target_class']}\ntrack: {track.name}\n"
            f"offset_s: {ex['offset_s']:.2f}\n")
        manifest.append({"task": task, "idx": i, "instruction": instr,
                         "target_class": ex["target_class"],
                         "track": track.name, "offset_s": ex["offset_s"]})
        counts[task] += 1
        cls_counts[ex["target_class"]] = cls_counts.get(ex["target_class"], 0) + 1

    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    print(f"done: {counts} | target classes {cls_counts} | "
          f"{len(loaded)} tracks touched", flush=True)


if __name__ == "__main__":
    main()
