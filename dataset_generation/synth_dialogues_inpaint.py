"""Conversations for the SA3 segment-inpainting pairs.

Each conversation is about a LOCALIZED edit: the user asks for a change in a specific
part of the chunk ("drop the drums from about 6 to 14 seconds"), the assistant
confirms — mentioning the segment — and ends with the [EDIT] block. Grounded on the
plan's op/target/segment plus the source metadata embedded in the plan's instruction
and inpaint prompt.

Appends to data/edit_dataset/dialogues/dialogues.jsonl. Requires vLLM on :9003.

Usage:
  python src/edit_agent/synth_dialogues_inpaint.py --limit 20 --workers 4  # dry run
  python src/edit_agent/synth_dialogues_inpaint.py --workers 16
"""

import argparse
import json
import random
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils.llm_client import LLM_MODEL, LLMError, chat_json, map_concurrent  # noqa: E402
import copy

from dataset_generation.synth_dialogues import DIALOGUE_SCHEMA, validate_turns  # noqa: E402
from edit_agent.tokens import EDIT_BLOCK  # noqa: E402

# sessions have 2 turns per edit (up to 4 edits); the music schema caps at 6 turns
SESSION_SCHEMA = copy.deepcopy(DIALOGUE_SCHEMA)
SESSION_SCHEMA["properties"]["turns"]["maxItems"] = 8

PLAN_PATH = PROJECT_ROOT / "data/edit_dataset/inpaint/inpaint_plan.jsonl"
PAIRS_PATH = PROJECT_ROOT / "data/edit_dataset/inpaint/inpaint_pairs.jsonl"
OUT_PATH = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"

SYSTEM_PROMPT = """\
You write realistic training conversations between a USER and a music/audio-editing
ASSISTANT. The user is editing one chunk of audio the assistant can hear; this edit is
LOCALIZED — it applies only to a specific time segment inside the chunk, not the whole
chunk. You get the exact operation, the affected segment in seconds, and a reference
instruction.

Rules:
- The user must indicate WHERE the change applies, in casual terms (exact seconds,
  "around the middle", "in the second half", "for a few seconds after the intro"...).
- The assistant's final turn is ONE short sentence confirming the exact localized edit,
  mentioning the region. No special tokens or placeholders.
- 2 or 4 turns, strictly alternating, user first, assistant last.
- Ground strictly in the given information; never invent other instruments or times.
- Never break character."""


SESSION_PROMPT_ADDON = """\
This is an EXPERIMENTATION SESSION: the user tries several DIFFERENT edits on the SAME
chunk, one at a time. Each new request REPLACES the previous attempt (phrases like
"hmm, let's instead...", "undo that — now try...", "what about..."), it does NOT stack
on top of it. Each assistant turn confirms only the current edit, one short sentence,
mentioning the affected region."""


def gen_messages_session(plans: list[dict]) -> list[dict]:
    lines = [f"The chunk is {plans[0]['duration_s']:.1f}s long. The user will try these "
             f"{len(plans)} alternative localized edits, in this order:"]
    for i, p in enumerate(plans):
        lines.append(f"  {i + 1}. [{p['op']}] {p['edit_instruction']} "
                     f"(segment {p['segment_start_s']:.1f}-{p['segment_end_s']:.1f}s)")
    lines.append(f"Write a {2 * len(plans)}-turn conversation: each user turn requests "
                 f"the next alternative in their own words; each assistant turn is one "
                 f"short confirmation of exactly that edit.")
    return [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + SESSION_PROMPT_ADDON},
            {"role": "user", "content": "\n".join(lines)}]


def gen_messages(plan: dict, n_turns: int) -> list[dict]:
    task = (f"Operation: {plan['op']} (target: {plan['target']})\n"
            f"Affected segment: {plan['segment_start_s']:.1f}s to "
            f"{plan['segment_end_s']:.1f}s of a {plan['duration_s']:.1f}s chunk.\n"
            f"Reference instruction (reach exactly this, in the user's own words): "
            f"\"{plan['edit_instruction']}\"\n"
            f"What the segment sounds like after the edit: {plan['inpaint_prompt']}\n"
            f"Write a {n_turns}-turn conversation.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task}]


def process_session(item) -> dict:
    plans, targets = item  # 4 plans of one source + their target paths
    rng = random.Random(plans[0]["source_id"] + ":sess")
    plans = plans[:]
    rng.shuffle(plans)
    for attempt in range(2):
        try:
            res = chat_json(gen_messages_session(plans), SESSION_SCHEMA,
                            schema_name="dialogue", max_tokens=2000,
                            temperature=0.8 + 0.1 * attempt)
        except LLMError:
            continue
        turns = res["turns"]
        if len(turns) == 2 * len(plans) and validate_turns(turns, True) is None:
            messages = []
            for i, t in enumerate(turns):
                if i == 0:
                    content = [{"type": "audio", "audio": plans[0]["audio_path"]},
                               {"type": "text", "text": t["content"]}]
                else:
                    text = t["content"]
                    if i % 2 == 1:  # every assistant turn confirms an edit
                        text = text.rstrip() + " " + EDIT_BLOCK
                    content = [{"type": "text", "text": text}]
                messages.append({"role": t["role"], "content": content})
            return {
                "id": f"inp_session_{plans[0]['source_id']}",
                "song_id": plans[0]["song_id"], "chunk_id": plans[0]["source_id"],
                "conv_type": "inp_session",
                "image_path": None, "image_category": None,
                "chunk_path": plans[0]["audio_path"],
                "edited_path": targets[plans[-1]["plan_id"]],
                "edit_instruction": " ; ".join(p["edit_instruction"] for p in plans),
                "edit_type": "inp_session", "has_edit": True,
                "steps": [{"op": p["op"], "instruction": p["edit_instruction"],
                           "segment": [p["segment_start_s"], p["segment_end_s"]],
                           "input_path": p["audio_path"],
                           "target_path": targets[p["plan_id"]]} for p in plans],
                "messages": messages,
                "split": plans[0]["split"], "llm_model": LLM_MODEL,
            }
    raise LLMError("invalid session dialogue after retries")


def process(item: tuple[dict, str]) -> dict:
    plan, target_path = item
    rng = random.Random(plan["plan_id"] + ":conv")
    n_turns = rng.choice([2, 2, 4])
    for attempt in range(2):
        try:
            res = chat_json(gen_messages(plan, n_turns), DIALOGUE_SCHEMA,
                            schema_name="dialogue", max_tokens=1200,
                            temperature=0.8 + 0.1 * attempt)
        except LLMError:
            continue
        turns = res["turns"]
        if validate_turns(turns, True) is None:
            messages = []
            for i, t in enumerate(turns):
                if i == 0:
                    content = [{"type": "audio", "audio": plan["audio_path"]},
                               {"type": "text", "text": t["content"]}]
                else:
                    text = t["content"]
                    if i == len(turns) - 1:
                        text = text.rstrip() + " " + EDIT_BLOCK
                    content = [{"type": "text", "text": text}]
                messages.append({"role": t["role"], "content": content})
            return {
                "id": plan["plan_id"], "song_id": plan["song_id"],
                "chunk_id": plan["source_id"],
                "conv_type": f"inp_{plan['op']}",
                "image_path": None, "image_category": None,
                "chunk_path": plan["audio_path"],
                "edited_path": target_path,
                "edit_instruction": plan["edit_instruction"],
                "edit_type": f"inp_{plan['op']}",
                "has_edit": True,
                "segment": [plan["segment_start_s"], plan["segment_end_s"]],
                "messages": messages,
                "split": plan["split"], "llm_model": LLM_MODEL,
            }
    raise LLMError("invalid inpaint dialogue after retries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import hashlib
    from collections import defaultdict

    plans = {json.loads(l)["plan_id"]: json.loads(l) for l in open(PLAN_PATH)}
    targets = {json.loads(l)["plan_id"]: json.loads(l).get("target_path")
               for l in open(PAIRS_PATH) if '"ok"' in l}
    done = {json.loads(l)["id"] for l in open(OUT_PATH)} if OUT_PATH.exists() else set()

    by_source = defaultdict(list)
    for pid, p in plans.items():
        if pid in targets:
            by_source[p["source_id"]].append(p)

    def is_session_source(sid: str) -> bool:
        return int(hashlib.md5((sid + ":sess").encode()).hexdigest(), 16) % 10 < 3

    items = []
    for sid, ps in by_source.items():
        if is_session_source(sid) and len(ps) >= 3:
            if f"inp_session_{sid}" not in done:
                items.append(("session", (sorted(ps, key=lambda x: x["op"]), targets)))
        else:
            for p in ps:
                if p["plan_id"] not in done:
                    items.append(("single", (p, targets[p["plan_id"]])))
    if args.limit:
        items = items[: args.limit]
    n_sess = sum(1 for k, _ in items if k == "session")
    print(f"{len(items)} conversations to generate ({n_sess} sessions, "
          f"{len(items) - n_sess} singles; {len(done)} dialogues exist)")

    def dispatch(item):
        kind, payload = item
        return process_session(payload) if kind == "session" else process(payload)

    lock = threading.Lock()
    n_ok = n_err = 0
    with open(OUT_PATH, "a") as out:
        for item, result, err in map_concurrent(dispatch, items, workers=args.workers,
                                                desc="inpaint dialogues"):
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
