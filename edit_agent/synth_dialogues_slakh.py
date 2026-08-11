"""Conversations for the Slakh stem-edit pairs — including multi-turn trajectories.

Two conversation shapes:
- single edit (pair rows): like the music dialogues, grounded on the EXACT stem
  inventory (input_stems) — the strongest grounding we have, no hallucination possible.
- trajectory (grouped by trajectory_id): one conversation with an [EDIT] block per
  step — "start with just the piano" -> [EDIT] -> "now add drums and bass" -> [EDIT]...
  Every assistant editing turn ends with the block.

Appends to data/edit_dataset/dialogues/dialogues.jsonl. Requires the vLLM server
on :9003.

Usage:
  python src/edit_agent/synth_dialogues_slakh.py --limit 20 --workers 4   # dry run
  python src/edit_agent/synth_dialogues_slakh.py --workers 16
"""

import argparse
import json
import random
import sys
import threading
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils.llm_client import LLM_MODEL, LLMError, chat_json, map_concurrent  # noqa: E402
from edit_agent.tokens import EDIT_BLOCK, typed_block  # noqa: E402

PAIRS = PROJECT_ROOT / "data/edit_dataset/slakh/slakh_pairs.jsonl"
OUT_PATH = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"

SYSTEM_PROMPT = """\
You write realistic training conversations between a USER and a music-production
ASSISTANT. The user is working on one section of a multi-track song; the assistant can
hear it and knows exactly which instrument stems are playing. You are given the exact
stem list and the edit(s) the user performs. Ground everything in those stems — never
mention instruments that are not listed.

Rules:
- The user speaks like a musician/producer, casual and specific ("drop the choir, it's
  too much", "give me just bass and drums").
- Strictly alternate user/assistant turns, starting with user, ending with assistant.
- EVERY assistant turn that confirms an edit is ONE short sentence confirming exactly
  that edit. No special tokens or placeholders.
- Never break character."""

DIALOGUE_SCHEMA = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array", "minItems": 2, "maxItems": 8,
            "items": {"type": "object",
                      "properties": {"role": {"type": "string", "enum": ["user", "assistant"]},
                                     "content": {"type": "string", "minLength": 5}},
                      "required": ["role", "content"]},
        }
    },
    "required": ["turns"],
}


def gen_messages_single(row: dict, n_turns: int) -> list[dict]:
    task = (f"Instruments currently playing in this section: {', '.join(row['input_stems'])}.\n"
            f"The edit the user requests (reach exactly this, casually): "
            f"\"{row['instruction_seed']}\".\n"
            f"Write a {n_turns}-turn conversation; final assistant turn = one short "
            f"confirmation of the edit.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task}]


def gen_messages_traj(steps: list[dict]) -> list[dict]:
    lines = [f"Instruments available in this section: "
             f"{', '.join(sorted(set(steps[0]['input_stems']) | set(steps[-1]['target_stems'])))}.",
             f"The user starts from a stripped-down state ({', '.join(steps[0]['input_stems'])}) "
             f"and builds the mix up over {len(steps)} steps:"]
    for i, s in enumerate(steps):
        lines.append(f"  step {i + 1}: {s['instruction_seed']}")
    lines.append(
        f"Write a {2 * len(steps)}-turn conversation: each user turn requests the next "
        f"step (in their own words), each assistant turn is one short sentence "
        f"confirming exactly that step's change.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


def validate(turns: list[dict], expect_edits: int) -> str | None:
    if len(turns) != 2 * expect_edits and expect_edits > 1:
        return f"traj_turn_count:{len(turns)}!={2*expect_edits}"
    if len(turns) < 2 or len(turns) % 2 != 0:
        return f"turn_count:{len(turns)}"
    for i, t in enumerate(turns):
        if t["role"] != ("user" if i % 2 == 0 else "assistant"):
            return f"role_order:{i}"
        if "[EDIT" in t["content"]:
            return f"llm_emitted_block:{i}"
        if not t["content"].strip():
            return f"empty:{i}"
    return None


def build_record(rid, song_id, conv_type, turns, audio_path, edit_blocks_on, extra,
                 block_ops=None):
    messages = []
    for i, t in enumerate(turns):
        if i == 0:
            content = [{"type": "audio", "audio": audio_path},
                       {"type": "text", "text": t["content"]}]
        else:
            text = t["content"]
            if i in edit_blocks_on:
                blk = typed_block(block_ops[i]) if block_ops and i in block_ops \
                    else EDIT_BLOCK
                text = text.rstrip() + " " + blk
            content = [{"type": "text", "text": text}]
        messages.append({"role": t["role"], "content": content})
    return {"id": rid, "song_id": song_id, "chunk_id": rid,
            "conv_type": conv_type, "image_path": None, "image_category": None,
            "chunk_path": audio_path, "has_edit": True,
            "messages": messages, "llm_model": LLM_MODEL, **extra}


def process(item) -> dict:
    kind, payload = item
    rng = random.Random(str(payload)[:64] + ":slakh")
    if kind == "single":
        row = payload
        n_turns = rng.choice([2, 2, 4])
        for attempt in range(2):
            try:
                res = chat_json(gen_messages_single(row, n_turns), DIALOGUE_SCHEMA,
                                schema_name="dialogue", max_tokens=1200,
                                temperature=0.8 + 0.1 * attempt)
            except LLMError:
                continue
            turns = res["turns"]
            if validate(turns, 1) is None:
                return build_record(
                    f"slakh_{row['pair_id']}", f"slakh_{row['track']}",
                    f"slakh_{row['op']}", turns, row["input_path"],
                    edit_blocks_on={len(turns) - 1},
                    block_ops={len(turns) - 1: row["op"]},
                    extra={"edited_path": row["target_path"],
                           "edit_instruction": row["instruction_seed"],
                           "edit_type": f"slakh_{row['op']}", "split": row["split"]})
        raise LLMError("invalid single dialogue after retries")
    else:  # trajectory
        steps = payload
        for attempt in range(2):
            try:
                res = chat_json(gen_messages_traj(steps), DIALOGUE_SCHEMA,
                                schema_name="dialogue", max_tokens=1800,
                                temperature=0.8 + 0.1 * attempt)
            except LLMError:
                continue
            turns = res["turns"]
            if validate(turns, len(steps)) is None:
                return build_record(
                    f"slakh_{steps[0]['trajectory_id']}", f"slakh_{steps[0]['track']}",
                    "slakh_traj", turns, steps[0]["input_path"],
                    edit_blocks_on={2 * i + 1 for i in range(len(steps))},
                    block_ops={2 * i + 1: steps[i]["op"] for i in range(len(steps))},
                    extra={"edited_path": steps[-1]["target_path"],
                           "edit_instruction": " ; ".join(s["instruction_seed"] for s in steps),
                           "edit_type": "slakh_traj", "split": steps[0]["split"],
                           "steps": [{"instruction": s["instruction_seed"],
                                      "input_path": s["input_path"],
                                      "target_path": s["target_path"]} for s in steps]})
        raise LLMError("invalid trajectory dialogue after retries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(PAIRS)]
    singles = [r for r in rows if not r["trajectory_id"]]
    trajs = defaultdict(list)
    for r in rows:
        if r["trajectory_id"]:
            trajs[r["trajectory_id"]].append(r)
    for t in trajs.values():
        t.sort(key=lambda x: x["step_index"])

    done = {json.loads(l)["id"] for l in open(OUT_PATH)} if OUT_PATH.exists() else set()
    items = [("single", r) for r in singles if f"slakh_{r['pair_id']}" not in done]
    items += [("traj", steps) for tid, steps in trajs.items() if f"slakh_{tid}" not in done]
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} conversations to generate "
          f"({len(singles)} singles, {len(trajs)} trajectories total)")

    lock = threading.Lock()
    n_ok = n_err = 0
    with open(OUT_PATH, "a") as out:
        for item, result, err in map_concurrent(process, items, workers=args.workers,
                                                desc="slakh dialogues"):
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
