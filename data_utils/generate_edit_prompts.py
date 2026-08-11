"""Stage 2 of the edit-dataset pipeline: Qwen-generated chunk descriptions + edit prompts.

For each chunk from chunks.jsonl, one guided-JSON call to the local vLLM Qwen server
produces a structured description (instruments, mood, genre, type beat, BPM, energy,
effects) grounded in the librosa features + song tags, plus 2-3 diverse edit
instructions with matching MelodyFlow target prompts. Optional --verify runs a second
feasibility-check pass per edit.

Usage:
  python src/data_utils/generate_edit_prompts.py --workers 16
  python src/data_utils/generate_edit_prompts.py --limit 5 --workers 2   # smoke test
"""

import argparse
import json
import re
import threading
from pathlib import Path

from llm_client import LLM_MODEL, chat_json, map_concurrent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = PROJECT_ROOT / "data/edit_dataset/manifests"
CHUNKS_MANIFEST = MANIFEST_DIR / "chunks.jsonl"
PROMPTS_MANIFEST = MANIFEST_DIR / "edit_prompts.jsonl"

CATEGORY_HINTS = {
    "安静平和": "calm, peaceful, serene traditional East Asian instrumental",
    "悲伤孤傲": "sad, lonely, melancholic traditional East Asian instrumental",
    "活泼欢快": "lively, cheerful, upbeat traditional East Asian instrumental",
    "激昂肆意": "passionate, intense, unrestrained traditional East Asian instrumental",
}

EDIT_TYPES = ["instrument_add", "instrument_remove", "instrument_swap",
              "mood_shift", "energy_change", "effect_add", "genre_transfer"]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "object",
            "properties": {
                "instruments": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "mood": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "genre": {"type": "string"},
                "type_beat": {"type": "string"},
                "bpm": {"type": "integer"},
                "energy": {"type": "string", "enum": ["low", "medium", "high"]},
                "effects": {"type": "array", "items": {"type": "string"}},
                "vocals": {"type": "boolean"},
                "caption": {"type": "string"},
            },
            "required": ["instruments", "mood", "genre", "type_beat", "bpm",
                         "energy", "effects", "vocals", "caption"],
        },
        "edits": {
            "type": "array",
            "minItems": 2,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "edit_type": {"type": "string", "enum": EDIT_TYPES},
                    "edit_instruction": {"type": "string"},
                    "melodyflow_prompt": {"type": "string"},
                },
                "required": ["edit_type", "edit_instruction", "melodyflow_prompt"],
            },
        },
    },
    "required": ["description", "edits"],
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "feasible": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["feasible", "reason"],
}

SYSTEM_PROMPT = """\
You are a professional music-production annotator building a dataset for a music-editing AI.
You receive objective audio features extracted with librosa from ONE chunk of a song, plus
the song-level tags/description and the chunk's position in the song.

Task 1 — describe the chunk. Ground every claim in the evidence: the features describe THIS
chunk, the tags describe the WHOLE song, so when they conflict trust the features (e.g. a
quiet intro of a loud song). Do not invent instruments the evidence does not support. Set
vocals=true only if the tags clearly indicate vocals/singing. "type_beat" is a short
YouTube-style descriptor like "chill lofi type beat" or "epic orchestral type beat".
"caption" is one sentence describing the chunk like a stock-music catalog entry.

Task 2 — propose diverse edits a user might request on this chunk, each of a different
edit_type. Hard constraints:
- Each edit must be PLAUSIBLE for this chunk: never remove, mute, or soften something that
  is not present; never mention vocals when vocals=false.
- Edits must preserve song coherence: keep the tempo (~the measured BPM), key, and overall
  structure recognizable, so the edited chunk still fits when placed back in the song.
  Prefer targeted changes (one or two elements) over full rewrites.
- "edit_instruction" is casual natural user language, e.g. "make the drums punchier and add
  a bit of reverb".
- "melodyflow_prompt" describes the DESIRED OUTPUT audio as a standalone caption — genre,
  instruments, mood, tempo in BPM, texture/production — not the change itself. One or two
  sentences, stock-music-catalog style with concrete instrument detail. It must stay
  consistent with everything the edit does not change."""


def render_user_message(row: dict) -> str:
    f = row["features"]
    lines = [
        "AUDIO FEATURES (librosa, this chunk only):",
        f"- tempo: {f['bpm']} BPM ({f['beat_count']} beats)",
        f"- key: {f['key']} (confidence {f['key_confidence']}), strongest pitches: {', '.join(f['chroma_top3'])}",
        f"- energy: {f.get('energy_bucket', 'mid')} (RMS mean {f['rms_mean']}, std {f['rms_std']})",
        f"- brightness: spectral centroid {f['spectral_centroid_mean']} Hz, rolloff {f['spectral_rolloff_mean']} Hz",
        f"- texture: spectral flatness {f['spectral_flatness_mean']} (0=tonal, 1=noisy)",
        f"- onset density: {f['onset_density_per_s']} events/s",
        "",
    ]
    if row["source"] == "suno":
        lines.append(f"SONG TAGS (whole song): {row['song_tags']}")
    else:
        hint = CATEGORY_HINTS.get(row["category"], row["category"])
        lines.append(f"SONG STYLE: {hint}")
        if row.get("song_text_prompt"):
            lines.append(f"SONG PROMPT: {row['song_text_prompt']}")
    lines.append("")
    lines.append(f"CHUNK: {row['chunk_index'] + 1}/{row['n_chunks']} "
                 f"({row['position']} of song), {row['duration_s']:.1f} s long.")
    return "\n".join(lines)


# QA heuristic in the spirit of src/scripts/clean_prompts.py::classify_prompt
_META_RE = re.compile(r"^(here is|here's|sure|i will|i'll|as an ai|the user)", re.IGNORECASE)


def classify_prompt(prompt: str) -> str:
    p = prompt.strip()
    if len(p) < 30:
        return "too_short"
    if len(p) > 600:
        return "too_long"
    if _META_RE.match(p):
        return "meta_text"
    return "ok"


def process_chunk(row: dict, verify: bool) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_message(row)},
    ]
    result = chat_json(messages, RESPONSE_SCHEMA, schema_name="chunk_edits")
    desc = result["description"]

    out = []
    for j, edit in enumerate(result["edits"]):
        feasibility = "ok"
        fmt = classify_prompt(edit["melodyflow_prompt"])
        if fmt != "ok":
            feasibility = f"format_{fmt}"
        elif verify:
            v = chat_json(
                [{"role": "system",
                  "content": "You are a strict music-production reviewer."},
                 {"role": "user",
                  "content": (
                      "Chunk description:\n" + json.dumps(desc, ensure_ascii=False) +
                      f"\n\nProposed edit instruction: \"{edit['edit_instruction']}\"\n\n"
                      "Is this instruction performable on this exact audio without "
                      "contradicting it (nothing referenced that is absent, tempo/key/"
                      "structure preserved)? Answer strictly."
                  )}],
                VERIFY_SCHEMA, schema_name="feasibility", max_tokens=512,
            )
            if not v["feasible"]:
                feasibility = f"flagged: {v['reason'][:200]}"

        out.append({
            "edit_id": f"{row['chunk_id']}_e{j}",
            "chunk_id": row["chunk_id"],
            "description": desc,
            "edit_type": edit["edit_type"],
            "edit_instruction": edit["edit_instruction"],
            "melodyflow_prompt": edit["melodyflow_prompt"],
            "src_prompt": desc["caption"],
            "feasibility": feasibility,
            "llm_model": LLM_MODEL,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="max chunks (smoke tests)")
    parser.add_argument("--verify", action="store_true", help="second feasibility pass per edit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(CHUNKS_MANIFEST) as f:
        chunks = [json.loads(l) for l in f]

    done: set[str] = set()
    if PROMPTS_MANIFEST.exists() and not args.overwrite:
        with open(PROMPTS_MANIFEST) as f:
            done = {json.loads(l)["chunk_id"] for l in f}
        print(f"resuming: {len(done)} chunks already have prompts")

    pending = [c for c in chunks if c["chunk_id"] not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"{len(pending)} chunks to process (of {len(chunks)} total)")

    lock = threading.Lock()
    n_ok = n_err = 0
    mode = "w" if args.overwrite else "a"
    with open(PROMPTS_MANIFEST, mode) as mf:
        for row, result, err in map_concurrent(
                lambda r: process_chunk(r, args.verify), pending,
                workers=args.workers, desc="edit prompts"):
            if err is not None:
                n_err += 1
                print(f"\n[error] {row['chunk_id']}: {err}")
                continue
            with lock:
                for rec in result:
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()
            n_ok += 1

    print(f"\ndone: {n_ok} chunks ok, {n_err} failed"
          + (" (rerun to retry failures)" if n_err else ""))


if __name__ == "__main__":
    main()
