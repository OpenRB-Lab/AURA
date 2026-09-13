"""LLM agent that plans segment-level inpainting edits (Stage: SA3 inpaint data).

For each source chunk, the vLLM-served Qwen decides — from the chunk's verified
metadata (description/mood/instruments for music chunks; exact stem list for Slakh
mixes) — a time SEGMENT inside the chunk and an edit on it:
  add_stem | delete_stem | replace_stem | change_mood
and writes the Stable Audio 3 inpainting prompt (what the segment should sound like
AFTER the edit) plus a casual user instruction.

Output: data/edit_dataset/inpaint/inpaint_plan.jsonl

Usage:
  python src/edit_agent/plan_inpaint_edits.py --limit 20 --workers 4   # dry run
  python src/edit_agent/plan_inpaint_edits.py --n-music 3000 --n-slakh 1500 --workers 16
"""

import argparse
import hashlib
import json
import random
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils.llm_client import LLM_MODEL, LLMError, chat_json, map_concurrent  # noqa: E402

DATASET = PROJECT_ROOT / "data/edit_dataset/manifests/dataset.jsonl"
SLAKH_PAIRS = PROJECT_ROOT / "data/edit_dataset/slakh/slakh_pairs.jsonl"
OUT_DIR = PROJECT_ROOT / "data/edit_dataset/inpaint"
PLAN_PATH = OUT_DIR / "inpaint_plan.jsonl"

OPS = ["add_stem", "delete_stem", "replace_stem", "change_mood"]

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "segment_start_s": {"type": "number"},
        "segment_end_s": {"type": "number"},
        "op": {"type": "string", "enum": OPS},
        "target": {"type": "string"},
        "edit_instruction": {"type": "string"},
        "inpaint_prompt": {"type": "string", "minLength": 40},
    },
    "required": ["segment_start_s", "segment_end_s", "op", "target",
                 "edit_instruction", "inpaint_prompt"],
}

SYSTEM_PROMPT = """\
You plan localized audio edits for a music-editing dataset. You receive verified
metadata about one music chunk and must design ONE segment-level edit: pick a time
segment INSIDE the chunk (5-12 seconds long, fully within the chunk, not starting at
0.0 — leave at least 2s of context before and after when possible) and one operation:

- add_stem: a new instrument enters during that segment only
- delete_stem: one listed instrument drops out during that segment
- replace_stem: one listed instrument is replaced by another during that segment
- change_mood: the segment shifts to a different mood/intensity, same instrumentation

Hard rules:
- Ground strictly in the metadata: only delete/replace instruments that are listed;
  the added/replacement instrument must fit the genre.
- "inpaint_prompt" describes the FULL audio content of the segment AFTER the edit —
  genre, all instruments playing there (unchanged ones included), tempo in BPM, mood,
  production feel. Stock-music-caption style, 1-2 sentences. It must be consistent with
  the surrounding music so the segment blends in.
- "edit_instruction" is what a user would casually say, mentioning WHERE (e.g. "around
  the middle", "in the second half", "from about 8 to 15 seconds").
- "target" names the stem/mood being changed (e.g. "electric guitar", "dreamy")."""


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def load_music_sources(n: int) -> list[dict]:
    """One record per chunk from dataset.jsonl (description is chunk-level enough)."""
    seen, out = set(), []
    with open(DATASET) as f:
        for line in f:
            r = json.loads(line)
            if r["chunk_id"] in seen or not r["qa"]["passed"] or r["duration_s"] < 15:
                continue
            seen.add(r["chunk_id"])
            d = r["description"]
            out.append({
                "source_id": r["chunk_id"], "kind": "music",
                "audio_path": r["chunk_path"], "duration_s": r["duration_s"],
                "meta": (f"genre: {d['genre']} | instruments: {', '.join(d['instruments'])} | "
                         f"mood: {', '.join(d['mood'])} | tempo: {d['bpm']} BPM | "
                         f"caption: {d['caption']}"),
                "song_id": r["song_id"],
            })
    rng = random.Random(11)
    rng.shuffle(out)
    return out[:n]


def load_slakh_sources(n: int) -> list[dict]:
    """Full-mix inputs of slakh 'remove' pairs (input = all active stems)."""
    seen, out = set(), []
    with open(SLAKH_PAIRS) as f:
        for line in f:
            r = json.loads(line)
            if r["op"] != "remove" or r["trajectory_id"]:
                continue
            key = (r["track"], r["chunk_index"])
            if key in seen or r["duration_s"] < 15:
                continue
            seen.add(key)
            out.append({
                "source_id": f"slakhmix_{r['track']}_c{r['chunk_index']}", "kind": "slakh",
                "audio_path": r["input_path"], "duration_s": r["duration_s"],
                "meta": f"instruments (exact stem list): {', '.join(r['input_stems'])}",
                "song_id": f"slakh_{r['track']}",
            })
    rng = random.Random(12)
    rng.shuffle(out)
    return out[:n]


def process(item: tuple[dict, str]) -> dict:
    src, forced_op = item
    user = (f"CHUNK METADATA: {src['meta']}\n"
            f"Chunk duration: {src['duration_s']:.1f} seconds.\n"
            f"Required operation: {forced_op}.\n"
            f"Design the segment edit now.")
    last = None
    for attempt in range(2):
        try:
            plan = chat_json(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user}],
                PLAN_SCHEMA, schema_name="inpaint_plan", max_tokens=700,
                temperature=0.8 + 0.1 * attempt)
        except LLMError as exc:
            last = str(exc)
            continue
        s, e = plan["segment_start_s"], plan["segment_end_s"]
        dur = src["duration_s"]
        if not (0 <= s < e <= dur and 3.0 <= e - s <= 14.0):
            last = f"bad_segment {s}-{e} in {dur}"
            continue
        if plan["op"] != forced_op:
            last = f"op_mismatch {plan['op']}"
            continue
        return {
            "plan_id": f"inp_{src['source_id']}_{forced_op}",
            "source_id": src["source_id"], "kind": src["kind"],
            "song_id": src["song_id"],
            "audio_path": src["audio_path"], "duration_s": src["duration_s"],
            "segment_start_s": round(float(s), 2), "segment_end_s": round(float(e), 2),
            "op": plan["op"], "target": plan["target"],
            "edit_instruction": plan["edit_instruction"],
            "inpaint_prompt": plan["inpaint_prompt"],
            "split": "val" if stable_hash(src["song_id"]) % 20 == 0 else "train",
            "llm_model": LLM_MODEL,
        }
    raise LLMError(f"invalid plan after retries: {last}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-music", type=int, default=3000)
    parser.add_argument("--n-slakh", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sources = load_music_sources(args.n_music) + load_slakh_sources(args.n_slakh)
    # dedup on (source_id, op); legacy rows have plan_id inp_<source_id> (no op suffix)
    done_pairs = set()
    if PLAN_PATH.exists():
        for l in open(PLAN_PATH):
            r = json.loads(l)
            done_pairs.add((r["source_id"], r["op"]))
    pending = [(s, op) for s in sources for op in OPS
               if (s["source_id"], op) not in done_pairs]
    if args.limit:
        pending = pending[:args.limit]
    print(f"{len(pending)} plans to generate ({len(sources)} sources x {len(OPS)} ops, "
          f"{len(done_pairs)} existing)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    n_ok = n_err = 0
    with open(PLAN_PATH, "a") as out:
        for item, result, err in map_concurrent(process, pending, workers=args.workers,
                                                desc="inpaint plans"):
            if err is not None:
                n_err += 1
                continue
            with lock:
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
            n_ok += 1
    print(f"done: {n_ok} ok, {n_err} failed" + (" (rerun to retry)" if n_err else ""))


if __name__ == "__main__":
    main()
