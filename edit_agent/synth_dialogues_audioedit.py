"""Extend the conversation dataset with HKUSTAudio/AudioEdit records.

AudioEdit provides (input audio, output audio, instruction) triplets for general-audio
operations: add / remove / extract a sound event. This script turns a sampled subset
into multi-turn conversations in the same format as the music dialogues (assistant's
final turn ends with the [EDIT_0..7] block) and APPENDS them to dialogues.jsonl.

Grounding: the generator can't hear the audio, so dialogues are grounded strictly on
what the metadata guarantees — the named event is present in `mixed` inputs
(remove/extract) and absent from `residual` inputs (add). A small no-edit slice does
presence/absence QA on the same guarantee.

Usage:
  python src/edit_agent/synth_dialogues_audioedit.py --limit 20 --workers 4  # dry run
  python src/edit_agent/synth_dialogues_audioedit.py --workers 16
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
from edit_agent.synth_dialogues import DIALOGUE_SCHEMA, validate_turns  # noqa: E402
from edit_agent.tokens import EDIT_BLOCK  # noqa: E402

SAMPLE = PROJECT_ROOT / "data/edit_dataset/audioedit_sample.jsonl"
AUDIO_ROOT = PROJECT_ROOT / "data/edit_dataset/audioedit"
OUT_PATH = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"

SYSTEM_PROMPT = """\
You write realistic training conversations between a USER and an audio-editing ASSISTANT.
The user has uploaded a short everyday audio clip (ambience, household sounds, animals,
machines...) which the assistant can hear. You are told one verified fact about the clip
and the edit the user wants. Ground the conversation ONLY in that fact — do not invent
other specific sounds you cannot verify; generic wording ("the background", "the rest of
the clip") is fine.

Rules:
- The user speaks casually ("can you get rid of the dog barking?", "I need just the
  alarm sound isolated").
- Conversations have 2 or 4 turns, strictly alternating, starting with user, ending
  with assistant.
- The FINAL assistant turn is one short sentence confirming the exact agreed operation.
  No special tokens or placeholders.
- Never break character."""

OP_TEXT = {
    "add": ("The input clip does NOT contain the sound of '{ev}'. The user wants that "
            "sound ADDED to the clip (mixed naturally into the existing audio)."),
    "remove": ("The input clip contains the sound of '{ev}' mixed with other audio. "
               "The user wants the '{ev}' sound REMOVED, keeping everything else."),
    "extract": ("The input clip contains the sound of '{ev}' mixed with other audio. "
                "The user wants the '{ev}' sound EXTRACTED/ISOLATED so only it remains."),
}


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def event_of(rec: dict) -> str:
    return rec["prompt"].split("'")[1] if "'" in rec["prompt"] else "the target sound"


def build_generator_messages(rec: dict, no_edit: bool, n_turns: int) -> list[dict]:
    ev = event_of(rec)
    if no_edit:
        present = rec["type"] != "add"
        fact = (f"The clip {'contains' if present else 'does not contain'} the sound "
                f"of '{ev}'.")
        task = (f"{fact}\n\nWrite a {n_turns}-turn conversation where the user just asks "
                f"about the clip (e.g. whether a certain sound is in it, or what to do "
                f"with it) and the assistant answers using only the verified fact. "
                f"NO edit is requested or performed.")
    else:
        task = (OP_TEXT[rec["type"]].format(ev=ev) +
                f"\n\nReference instruction (reach exactly this operation, in the "
                f"user's own casual words): \"{rec['prompt']}\"\n"
                f"Write a {n_turns}-turn conversation. The final assistant turn: one "
                f"short sentence confirming the operation.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task}]


def process(rec: dict) -> dict:
    rng = random.Random(rec["id"] + rec["type"] + ":ae")
    no_edit = rng.random() < 0.08
    n_turns = rng.choice([2, 2, 4])
    input_rel = f"data/edit_dataset/audioedit/{rec['input']}"
    output_rel = f"data/edit_dataset/audioedit/{rec['output']}"

    last_err = None
    for attempt in range(2):
        try:
            result = chat_json(build_generator_messages(rec, no_edit, n_turns),
                               DIALOGUE_SCHEMA, schema_name="dialogue",
                               max_tokens=1200, temperature=0.8 + 0.1 * attempt)
        except LLMError as exc:
            last_err = str(exc)
            continue
        turns = result["turns"]
        err = validate_turns(turns, not no_edit)
        if err is None:
            messages = []
            for i, t in enumerate(turns):
                if i == 0:
                    content = [{"type": "audio", "audio": input_rel},
                               {"type": "text", "text": t["content"]}]
                else:
                    text = t["content"]
                    if not no_edit and i == len(turns) - 1:
                        text = text.rstrip() + " " + EDIT_BLOCK
                    content = [{"type": "text", "text": text}]
                messages.append({"role": t["role"], "content": content})
            return {
                "id": f"audioedit_{rec['id']}_{rec['type']}",
                "song_id": f"ae_{rec['id']}",
                "chunk_id": f"ae_{rec['id']}",
                "conv_type": "ae_no_edit" if no_edit else f"ae_{rec['type']}",
                "image_path": None,
                "image_category": None,
                "chunk_path": input_rel,
                "edited_path": None if no_edit else output_rel,
                "edit_instruction": None if no_edit else rec["prompt"],
                "edit_type": None if no_edit else f"ae_{rec['type']}",
                "has_edit": not no_edit,
                "messages": messages,
                "split": "val" if stable_hash(f"ae_{rec['id']}") % 20 == 0 else "train",
                "llm_model": LLM_MODEL,
            }
        last_err = err
    raise LLMError(f"dialogue invalid after retries: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = [json.loads(l) for l in open(SAMPLE)]
    usable = [r for r in records
              if (AUDIO_ROOT / r["input"]).exists() and (AUDIO_ROOT / r["output"]).exists()]
    print(f"{len(usable)}/{len(records)} records have both audio files on disk")

    done = {json.loads(l)["id"] for l in open(OUT_PATH)} if OUT_PATH.exists() else set()
    pending = [r for r in usable if f"audioedit_{r['id']}_{r['type']}" not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"{len(pending)} conversations to generate (appending to {OUT_PATH.name})")

    lock = threading.Lock()
    n_ok = n_err = 0
    with open(OUT_PATH, "a") as out:
        for item, result, err in map_concurrent(process, pending,
                                                workers=args.workers, desc="ae dialogues"):
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
