"""Stage 1 of the edit-dataset pipeline: beat-aligned audio chunking + feature extraction.

Cuts each source song into <30 s chunks at beat boundaries, extracts per-chunk
librosa features (BPM, key, energy, spectral stats, onset density), and writes
48 kHz stereo PCM16 wavs (MelodyFlow's native format) plus a chunks.jsonl manifest.

Sources:
  - suno mp3s from data/image_music/cached_latents/suno_audio/ (metadata in index_suno.jsonl)
  - painting-paired wavs from data/image_music/music_dataset/ (metadata in annotations.jsonl)

Usage:
  python src/data_utils/chunk_audio.py                    # everything
  python src/data_utils/chunk_audio.py --limit 3          # smoke test
  python src/data_utils/chunk_audio.py --sources suno     # suno only
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUNO_INDEX = PROJECT_ROOT / "data/image_music/cached_latents/index_suno.jsonl"
SUNO_AUDIO_DIR = PROJECT_ROOT / "data/image_music/cached_latents/suno_audio"
ANNOTATIONS = PROJECT_ROOT / "data/annotations.jsonl"

OUT_ROOT = PROJECT_ROOT / "data/edit_dataset"
CHUNKS_DIR = OUT_ROOT / "chunks"
MANIFEST_DIR = OUT_ROOT / "manifests"
CHUNKS_MANIFEST = MANIFEST_DIR / "chunks.jsonl"
EXCLUDED_MANIFEST = MANIFEST_DIR / "excluded_songs.jsonl"

TARGET_SR = 48000          # MelodyFlow t24 native rate
ANALYSIS_SR = 22050
MIN_SONG_S = 10.0
MIN_CHUNK_S = 10.0
MAX_CHUNK_S = 29.9
TARGET_CHUNK_S = 25.0

# Krumhansl-Schmuckler key profiles
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class SongEntry:
    song_id: str
    source: str                 # "suno" | "painting"
    audio_path: Path
    tags: str = ""
    text_prompt: str = ""
    category: str = ""
    single_chunk: bool = False  # painting wavs are already <30 s
    extra: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Source loading
# ──────────────────────────────────────────────────────────────

def load_sources(which: set[str]) -> tuple[list[SongEntry], list[dict]]:
    entries: list[SongEntry] = []
    excluded: list[dict] = []

    if "suno" in which:
        seen = set()
        with open(SUNO_INDEX) as f:
            for line in f:
                row = json.loads(line)
                sid = row["suno_id"]
                if sid in seen:
                    continue
                seen.add(sid)
                path = SUNO_AUDIO_DIR / f"{sid}.mp3"
                if not path.exists():
                    excluded.append({"song_id": f"suno_{sid}", "reason": "audio_missing"})
                    continue
                dur = row.get("duration")
                if dur is not None and dur < MIN_SONG_S:
                    excluded.append({"song_id": f"suno_{sid}", "reason": f"too_short ({dur:.1f}s)"})
                    continue
                entries.append(SongEntry(
                    song_id=f"suno_{sid}",
                    source="suno",
                    audio_path=path,
                    tags=row.get("tags", ""),
                    text_prompt=row.get("text_prompt", ""),
                ))

    if "painting" in which:
        seen = set()
        with open(ANNOTATIONS) as f:
            for line in f:
                row = json.loads(line)
                if row.get("source") != "painting":
                    continue
                mp = row.get("music_path", "")
                if not mp or mp in seen:
                    continue
                seen.add(mp)
                path = Path(mp)
                if not path.exists():
                    excluded.append({"song_id": f"paint_{row['idx']}", "reason": "audio_missing"})
                    continue
                entries.append(SongEntry(
                    song_id=f"paint_{row['idx']}_{path.stem}",
                    source="painting",
                    audio_path=path,
                    text_prompt=row.get("text_prompt", ""),
                    category=row.get("category", ""),
                    single_chunk=True,
                ))

    return entries, excluded


# ──────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────

def estimate_key(chroma_mean: np.ndarray) -> tuple[str, float]:
    """Krumhansl-Schmuckler template correlation over all 24 rotations."""
    best_key, best_corr = "unknown", -2.0
    for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
        for shift in range(12):
            corr = float(np.corrcoef(np.roll(profile, shift), chroma_mean)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_key = f"{_PITCHES[shift]} {mode}"
    return best_key, round(best_corr, 3)


def extract_features(y: np.ndarray, sr: int) -> dict:
    dur = len(y) / sr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key, key_conf = estimate_key(chroma_mean)
    top3 = [_PITCHES[i] for i in np.argsort(chroma_mean)[::-1][:3]]

    rms = librosa.feature.rms(y=y)[0]
    onsets = librosa.onset.onset_detect(y=y, sr=sr)

    return {
        "bpm": round(tempo, 1),
        "beat_count": int(len(beats)),
        "key": key,
        "key_confidence": key_conf,
        "rms_mean": round(float(rms.mean()), 5),
        "rms_std": round(float(rms.std()), 5),
        "spectral_centroid_mean": round(float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()), 1),
        "spectral_rolloff_mean": round(float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean()), 1),
        "spectral_flatness_mean": round(float(librosa.feature.spectral_flatness(y=y).mean()), 5),
        "onset_density_per_s": round(len(onsets) / max(dur, 1e-6), 2),
        "chroma_top3": top3,
    }


def energy_bucket(rms_mean: float, terciles: tuple[float, float]) -> str:
    lo, hi = terciles
    if rms_mean < lo:
        return "low"
    if rms_mean < hi:
        return "mid"
    return "high"


def position_label(idx: int, n: int) -> str:
    if n == 1:
        return "full"
    if idx == 0:
        return "intro"
    if idx == n - 1:
        return "outro"
    frac = idx / (n - 1)
    if frac < 0.34:
        return "early"
    if frac < 0.67:
        return "middle"
    return "late"


# ──────────────────────────────────────────────────────────────
# Chunk planning
# ──────────────────────────────────────────────────────────────

def plan_chunks(beat_times: np.ndarray, total_dur: float,
                target: float = TARGET_CHUNK_S,
                min_len: float = MIN_CHUNK_S,
                max_len: float = MAX_CHUNK_S) -> list[tuple[float, float]]:
    """Greedy beat-aligned segmentation tiling [0, total_dur].

    From each chunk start, cut at the beat closest to start+target within
    [start+min_len, start+max_len]; hard-cut at start+max_len when no beat lands
    in that window. A final remainder <min_len is merged into the previous chunk
    when the merge stays <=max_len, otherwise dropped.
    """
    if total_dur <= max_len:
        return [(0.0, round(total_dur, 3))]

    chunks: list[tuple[float, float]] = []
    start = 0.0
    while total_dur - start > max_len:
        lo, hi = start + min_len, start + max_len
        candidates = beat_times[(beat_times >= lo) & (beat_times <= hi)]
        if len(candidates):
            end = float(candidates[np.argmin(np.abs(candidates - (start + target)))])
        else:
            end = hi
        chunks.append((round(start, 3), round(end, 3)))
        start = end

    tail = total_dur - start
    if tail >= min_len:
        chunks.append((round(start, 3), round(total_dur, 3)))
    elif chunks and (total_dur - chunks[-1][0]) <= max_len:
        chunks[-1] = (chunks[-1][0], round(total_dur, 3))
    # else: drop the tail
    return chunks


# ──────────────────────────────────────────────────────────────
# Audio IO
# ──────────────────────────────────────────────────────────────

def save_chunk(y_full: np.ndarray, sr: int, start: float, end: float, out_path: Path) -> None:
    """Slice [start, end) seconds from full-quality audio, resample to 48 kHz stereo PCM16."""
    seg = y_full[:, int(start * sr):int(end * sr)]
    if sr != TARGET_SR:
        seg = librosa.resample(seg, orig_sr=sr, target_sr=TARGET_SR)
    if seg.shape[0] == 1:                       # mono → stereo
        seg = np.repeat(seg, 2, axis=0)
    peak = np.abs(seg).max()
    if peak > 1.0:
        seg = seg / peak
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, seg.T, TARGET_SR, subtype="PCM_16")


def process_song(entry: SongEntry, done_ids: set[str]) -> list[dict]:
    """Chunk one song; returns manifest rows (skips chunks already done)."""
    # Analysis pass: mono @ 22050
    y_mono, _ = librosa.load(entry.audio_path, sr=ANALYSIS_SR, mono=True)
    total_dur = len(y_mono) / ANALYSIS_SR
    if total_dur < MIN_SONG_S:
        raise ValueError(f"too_short ({total_dur:.1f}s)")

    if entry.single_chunk:
        chunks = [(0.0, round(min(total_dur, MAX_CHUNK_S), 3))]
    else:
        _, beats = librosa.beat.beat_track(y=y_mono, sr=ANALYSIS_SR)
        beat_times = librosa.frames_to_time(beats, sr=ANALYSIS_SR)
        chunks = plan_chunks(np.asarray(beat_times), total_dur)

    todo = [(k, s, e) for k, (s, e) in enumerate(chunks)
            if f"{entry.song_id}_c{k}" not in done_ids]
    if not todo:
        return []

    # Full-quality pass for writing chunks
    y_full, sr_full = librosa.load(entry.audio_path, sr=None, mono=False)
    y_full = np.atleast_2d(y_full)

    rows = []
    for k, s, e in todo:
        chunk_id = f"{entry.song_id}_c{k}"
        out_path = CHUNKS_DIR / entry.song_id / f"{chunk_id}.wav"
        save_chunk(y_full, sr_full, s, e, out_path)

        feats = extract_features(y_mono[int(s * ANALYSIS_SR):int(e * ANALYSIS_SR)], ANALYSIS_SR)
        rows.append({
            "chunk_id": chunk_id,
            "song_id": entry.song_id,
            "source": entry.source,
            "src_audio_path": str(entry.audio_path.relative_to(PROJECT_ROOT)),
            "chunk_path": str(out_path.relative_to(PROJECT_ROOT)),
            "start_s": s,
            "end_s": e,
            "duration_s": round(e - s, 3),
            "chunk_index": k,
            "n_chunks": len(chunks),
            "position": position_label(k, len(chunks)),
            "song_tags": entry.tags,
            "song_text_prompt": entry.text_prompt,
            "category": entry.category,
            "features": feats,
        })
    return rows


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def read_done(manifest: Path, overwrite: bool) -> tuple[set[str], set[str]]:
    """Returns (done chunk_ids with existing wavs, song_ids fully recorded)."""
    if overwrite or not manifest.exists():
        return set(), set()
    done, songs = set(), {}
    with open(manifest) as f:
        for line in f:
            row = json.loads(line)
            if (PROJECT_ROOT / row["chunk_path"]).exists():
                done.add(row["chunk_id"])
                songs.setdefault(row["song_id"], set()).add(row["chunk_index"])
    complete_songs = {s for s, idxs in songs.items()
                      if len(idxs) >= max(idxs) + 1}
    return done, complete_songs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=str, default="suno,painting")
    parser.add_argument("--limit", type=int, default=None, help="max songs to process (smoke tests)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--procs", type=int, default=8, help="worker processes for analysis")
    args = parser.parse_args()

    which = set(args.sources.split(","))
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    entries, excluded = load_sources(which)
    print(f"{len(entries)} songs to consider, {len(excluded)} excluded up-front")
    with open(EXCLUDED_MANIFEST, "w") as f:
        for row in excluded:
            f.write(json.dumps(row) + "\n")

    done_ids, done_songs = read_done(CHUNKS_MANIFEST, args.overwrite)
    if done_ids:
        print(f"resuming: {len(done_ids)} chunks from {len(done_songs)} songs already done")

    pending = [e for e in entries if e.song_id not in done_songs]
    if args.limit:
        pending = pending[:args.limit]

    # RMS terciles for the energy bucket come from the rows as we go; we do a
    # cheap two-pass instead: collect rows first, bucket at the end.
    mode = "w" if args.overwrite else "a"
    with open(CHUNKS_MANIFEST, mode) as mf, \
         ProcessPoolExecutor(max_workers=args.procs) as pool:
        futures = {pool.submit(process_song, entry, done_ids): entry for entry in pending}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="chunking songs"):
            entry = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:  # noqa: BLE001 — log and continue over bad files
                with open(EXCLUDED_MANIFEST, "a") as f:
                    f.write(json.dumps({"song_id": entry.song_id, "reason": f"error: {exc}"}) + "\n")
                continue
            for row in rows:
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            mf.flush()

    # Assign energy buckets across the whole manifest (rewrite in place)
    with open(CHUNKS_MANIFEST) as f:
        rows = [json.loads(l) for l in f]
    if rows:
        rms = sorted(r["features"]["rms_mean"] for r in rows)
        terciles = (rms[len(rms) // 3], rms[2 * len(rms) // 3])
        for r in rows:
            r["features"]["energy_bucket"] = energy_bucket(r["features"]["rms_mean"], terciles)
        tmp = CHUNKS_MANIFEST.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(CHUNKS_MANIFEST)

    n_songs = len({r["song_id"] for r in rows})
    durs = [r["duration_s"] for r in rows]
    print(f"\nmanifest: {len(rows)} chunks from {n_songs} songs")
    if durs:
        print(f"chunk duration: mean {np.mean(durs):.1f}s  min {min(durs):.1f}s  max {max(durs):.1f}s")


if __name__ == "__main__":
    main()
