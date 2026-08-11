"""Construct EXACT music-edit pairs from Slakh multi-stem tracks.

Unlike the MelodyFlow-regenerated targets, these pairs are built by mixing stems, so
the target preserves everything the edit doesn't touch — including compositional edits
(add/remove several instruments at once) and multi-step buildup trajectories with an
exact intermediate target at every step.

Operations: add (1-3 stems), remove (1-3), isolate (keep 1-2), rebalance (gain +-6-10dB
on 1-2 stems), swap (remove one, add another). Plus 3-step buildup trajectories.

Output: 44.1 kHz stereo FLAC pairs under data/edit_dataset/slakh/pairs/ + manifest
data/edit_dataset/slakh/slakh_pairs.jsonl.

Usage:
  python src/edit_agent/slakh_edits.py --slakh-root data/edit_dataset/slakh_raw/babyslakh_16k --limit-tracks 3   # smoke
  python src/edit_agent/slakh_edits.py --slakh-root data/edit_dataset/slakh_raw/slakh2100_flac_redux/train --max-tracks 600
"""

import argparse
import hashlib
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils.chunk_audio import plan_chunks  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "data/edit_dataset/slakh"
PAIRS_DIR = OUT_ROOT / "pairs"
MANIFEST = OUT_ROOT / "slakh_pairs.jsonl"

TARGET_SR = 44100          # DiffRhythm-native; MelodyFlow/Qwen paths resample anyway
ANALYSIS_SR = 22050
ACTIVE_RMS = 2e-3          # stem considered active in a chunk above this RMS

OPS = ["add", "remove", "isolate", "rebalance", "swap"]
OP_WEIGHTS = [0.30, 0.25, 0.15, 0.20, 0.10]


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def stem_display_names(meta: dict) -> dict[str, str]:
    """stem_id -> readable instrument name, disambiguated when a name repeats."""
    raw = {sid: (info.get("midi_program_name") or info.get("inst_class") or sid).lower()
           for sid, info in meta["stems"].items()}
    totals: dict[str, int] = {}
    for n in raw.values():
        totals[n] = totals.get(n, 0) + 1
    seen: dict[str, int] = {}
    names = {}
    for sid in sorted(raw):
        n = raw[sid]
        if totals[n] > 1:
            seen[n] = seen.get(n, 0) + 1
            names[sid] = f"{n} {seen[n]}"
        else:
            names[sid] = n
    return names


def describe(names: dict, sids: list[str]) -> str:
    return ", ".join(names[s] for s in sorted(sids))


def sample_edits(active: list[str], names: dict, rng: random.Random, n_edits: int,
                 ops: list[str] | None = None):
    """Yield edit dicts {op, input_stems, target_stems, gains, instruction_seed}."""
    use_ops = ops or OPS
    use_w = [OP_WEIGHTS[OPS.index(o)] for o in use_ops]
    edits = []
    tries = 0
    while len(edits) < n_edits and tries < n_edits * 8:
        tries += 1
        op = rng.choices(use_ops, weights=use_w)[0]
        A = set(active)
        if op == "add" and len(active) >= 2:
            k = rng.randint(1, min(3, len(active) - 1))
            add = set(rng.sample(active, k))
            base = A - add
            if not base:
                continue
            edits.append(dict(op="add", input_stems=sorted(base), target_stems=sorted(A),
                              gains={}, instruction_seed=f"add {describe(names, list(add))}"))
        elif op == "remove" and len(active) >= 2:
            k = rng.randint(1, min(3, len(active) - 1))
            rem = set(rng.sample(active, k))
            edits.append(dict(op="remove", input_stems=sorted(A), target_stems=sorted(A - rem),
                              gains={}, instruction_seed=f"remove the {describe(names, list(rem))}"))
        elif op == "isolate" and len(active) >= 3:
            k = rng.randint(1, 2)
            keep = set(rng.sample(active, k))
            edits.append(dict(op="isolate", input_stems=sorted(A), target_stems=sorted(keep),
                              gains={}, instruction_seed=f"keep only the {describe(names, list(keep))}"))
        elif op == "rebalance" and len(active) >= 2:
            k = rng.randint(1, 2)
            targets = rng.sample(active, k)
            gains = {}
            words = []
            for s in targets:
                db = rng.choice([-9, -6, 6, 9])
                gains[s] = db
                words.append(f"{'boost' if db > 0 else 'lower'} the {names[s]}")
            edits.append(dict(op="rebalance", input_stems=sorted(A), target_stems=sorted(A),
                              gains=gains, instruction_seed=" and ".join(words)))
        elif op == "swap" and len(active) >= 3:
            out_s, in_s = rng.sample(active, 2)
            base = A - {out_s, in_s}
            edits.append(dict(op="swap", input_stems=sorted(base | {out_s}),
                              target_stems=sorted(base | {in_s}), gains={},
                              instruction_seed=f"replace the {names[out_s]} with {names[in_s]}"))
    # dedupe identical (input,target,gains)
    seen, out = set(), []
    for e in edits:
        key = (tuple(e["input_stems"]), tuple(e["target_stems"]), tuple(sorted(e["gains"].items())))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def sample_trajectory(active: list[str], names: dict, rng: random.Random):
    """3-step buildup: start small, add stems until (near-)full."""
    if len(active) < 4:
        return None
    order = active[:]
    rng.shuffle(order)
    start = order[: rng.randint(1, 2)]
    remaining = [s for s in order if s not in start]
    steps, state = [], set(start)
    for _ in range(3):
        if not remaining:
            break
        k = max(1, min(len(remaining) - 0, rng.randint(1, 2)))
        add = remaining[:k]
        remaining = remaining[k:]
        new_state = state | set(add)
        steps.append(dict(op="add", input_stems=sorted(state), target_stems=sorted(new_state),
                          gains={}, instruction_seed=f"add {describe(names, add)}"))
        state = new_state
    return dict(start=sorted(start), steps=steps) if len(steps) >= 2 else None


def render(stems_audio: dict, sids: list[str], gains: dict, scale: float,
           s0: int, s1: int) -> np.ndarray:
    mix = None
    for sid in sids:
        seg = stems_audio[sid][s0:s1].astype(np.float32)
        g = 10 ** (gains.get(sid, 0) / 20)
        mix = seg * g if mix is None else mix + seg * g
    mix = mix * scale
    np.clip(mix, -1.0, 1.0, out=mix)
    return mix


def save_flac(y: np.ndarray, sr: int, path: Path):
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    stereo = np.stack([y, y], axis=1) if y.ndim == 1 else y
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, stereo, TARGET_SR, format="FLAC")


def process_track(track_dir: str, pairs_per_chunk: int, traj_per_track: int,
                  force_split: str | None = None,
                  ops: list[str] | None = None) -> list[dict]:
    track_dir = Path(track_dir)
    track = track_dir.name
    meta = yaml.safe_load(open(track_dir / "metadata.yaml"))
    names = stem_display_names(meta)
    rng = random.Random(track)

    stems_audio, sr = {}, None
    for sid in meta["stems"]:
        for ext in (".flac", ".wav"):
            p = track_dir / "stems" / f"{sid}{ext}"
            if p.exists():
                y, sr = sf.read(p, dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                stems_audio[sid] = y
                break
    if len(stems_audio) < 3:
        return []
    n = min(len(y) for y in stems_audio.values())
    stems_audio = {k: v[:n] for k, v in stems_audio.items()}
    full = np.sum(list(stems_audio.values()), axis=0)
    dur = n / sr
    if dur < 30:
        return []

    mono22 = librosa.resample(full, orig_sr=sr, target_sr=ANALYSIS_SR)
    _, beats = librosa.beat.beat_track(y=mono22, sr=ANALYSIS_SR)
    beat_times = librosa.frames_to_time(beats, sr=ANALYSIS_SR)
    chunks = plan_chunks(np.asarray(beat_times), dur)
    rng.shuffle(chunks := list(enumerate(chunks)))

    rows, traj_done = [], 0
    split = force_split or ("val" if stable_hash(track) % 20 == 0 else "train")
    for ci, (s, e) in chunks:
        s0, s1 = int(s * sr), int(e * sr)
        active = [sid for sid, y in stems_audio.items()
                  if np.sqrt((y[s0:s1] ** 2).mean()) > ACTIVE_RMS]
        if len(active) < 2:
            continue
        peak = np.abs(full[s0:s1]).max()
        if peak < 1e-3:
            continue
        scale = 0.9 / peak

        jobs = []
        for j, edit in enumerate(sample_edits(active, names, rng, pairs_per_chunk,
                                              ops=ops)):
            jobs.append((f"{track}_c{ci}_e{j}", edit, None, None))
        if traj_done < traj_per_track:
            traj = sample_trajectory(active, names, rng)
            if traj:
                traj_done += 1
                tid = f"{track}_c{ci}_traj"
                for si, step in enumerate(traj["steps"]):
                    jobs.append((f"{tid}_s{si}", step, tid, (si, len(traj["steps"]))))

        for pair_id, edit, traj_id, step_info in jobs:
            in_path = PAIRS_DIR / track / f"{pair_id}_in.flac"
            out_path = PAIRS_DIR / track / f"{pair_id}_out.flac"
            if not out_path.exists():
                save_flac(render(stems_audio, edit["input_stems"], {}, scale, s0, s1), sr, in_path)
                save_flac(render(stems_audio, edit["target_stems"], edit["gains"], scale, s0, s1), sr, out_path)
            rows.append({
                "pair_id": pair_id, "track": track, "chunk_index": ci,
                "start_s": round(s, 3), "end_s": round(e, 3),
                "duration_s": round(e - s, 3),
                "op": edit["op"],
                "input_stems": [names[x] for x in edit["input_stems"]],
                "target_stems": [names[x] for x in edit["target_stems"]],
                "gains_db": {names[k]: v for k, v in edit["gains"].items()},
                "instruction_seed": edit["instruction_seed"],
                "input_path": str(in_path.relative_to(PROJECT_ROOT)),
                "target_path": str(out_path.relative_to(PROJECT_ROOT)),
                "trajectory_id": traj_id,
                "step_index": step_info[0] if step_info else None,
                "n_steps": step_info[1] if step_info else None,
                "split": split,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slakh-root", type=str, required=True)
    parser.add_argument("--max-tracks", type=int, default=600)
    parser.add_argument("--limit-tracks", type=int, default=None, help="smoke-test cap")
    parser.add_argument("--pairs-per-chunk", type=int, default=2)
    parser.add_argument("--traj-per-track", type=int, default=1)
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--force-split", type=str, default=None,
                        help="override split for all pairs (e.g. test)")
    parser.add_argument("--ops", type=str, default=None,
                        help="comma list restricting ops (e.g. isolate)")
    args = parser.parse_args()

    root = Path(args.slakh_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    tracks = sorted(str(p) for p in root.glob("Track*") if (p / "metadata.yaml").exists())
    rng = random.Random(0)
    if len(tracks) > args.max_tracks:
        tracks = sorted(rng.sample(tracks, args.max_tracks))
    if args.limit_tracks:
        tracks = tracks[: args.limit_tracks]

    done_tracks = set()
    if MANIFEST.exists():
        done_tracks = {json.loads(l)["track"] for l in open(MANIFEST)}
        print(f"resuming: {len(done_tracks)} tracks already in manifest")
    pending = [t for t in tracks if Path(t).name not in done_tracks]
    print(f"{len(pending)} tracks to process (of {len(tracks)} selected)")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    ops_list = args.ops.split(",") if args.ops else None
    with open(MANIFEST, "a") as mf, ProcessPoolExecutor(max_workers=args.procs) as pool:
        futs = {pool.submit(process_track, t, args.pairs_per_chunk, args.traj_per_track,
                            args.force_split, ops_list): t
                for t in pending}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="slakh tracks"):
            try:
                rows = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"\n[error] {Path(futs[fut]).name}: {exc}")
                continue
            for r in rows:
                mf.write(json.dumps(r, ensure_ascii=False) + "\n")
            mf.flush()
            n_pairs += len(rows)
    print(f"done: +{n_pairs} pairs -> {MANIFEST.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
