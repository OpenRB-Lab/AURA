"""Synthesize multi-turn edit conversations for training the music-editing agent.

For each qa-passed edit record, generates a conversation between a user and the
assistant about editing that music chunk, via the local vLLM Qwen server (:9003,
multimodal — reference images are shown to the generator so image turns are grounded).
The assistant's final turn ends with the literal [EDIT_0..7] block (appended and
validated by this script, never trusted to the LLM).

Conversation mix (deterministic per record): ~60% text-only edit, ~30% image-referenced
(mood reference / image-driven mood target / game-dev scenario), ~10% no-edit QA.

Usage:
  python src/edit_agent/synth_dialogues.py --limit 20 --workers 4   # dry run
  python src/edit_agent/synth_dialogues.py --workers 16             # full run
"""

import argparse
import base64
import hashlib
import json
import random
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_utils.llm_client import LLM_MODEL, LLMError, chat_json  # noqa: E402
from data_utils.llm_client import map_concurrent  # noqa: E402
from dataset_generation.mood_map import CATEGORY_EN, ImagePool, mood_to_category  # noqa: E402
from edit_agent.tokens import EDIT_BLOCK  # noqa: E402

DATASET = PROJECT_ROOT / "data/edit_dataset/manifests/dataset.jsonl"
OUT_DIR = PROJECT_ROOT / "data/edit_dataset/dialogues"
OUT_PATH = OUT_DIR / "dialogues.jsonl"

DIALOGUE_SCHEMA = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "minItems": 2,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant"]},
                    "content": {"type": "string", "minLength": 5},
                },
                "required": ["role", "content"],
            },
        }
    },
    "required": ["turns"],
}

SYSTEM_PROMPT = """\
You write realistic training conversations between a USER and a music-editing ASSISTANT.
The user has uploaded a song; they are working on ONE specific chunk of it, which the
assistant can hear. You are given the chunk's verified audio description and features.

Rules:
- The user speaks casually and sometimes vaguely, like a real person ("this part", "the
  drums here", "can you make it more..."). Ground everything in the ACTUAL audio
  content provided — never invent instruments or qualities that are not listed.
- The assistant is concise, friendly, and knowledgeable; it may reference what it hears
  ("the harpsichord line in this section...").
- Conversations have 2, 4, or 6 turns, strictly alternating user/assistant, starting
  with user and ending with assistant.
- The FINAL assistant turn must be a single short sentence that clearly confirms the
  exact agreed edit (when an edit is requested). Do NOT add any special tokens or
  placeholders — just the natural sentence.
- Do not mention that this is synthetic, and never break character."""


def stable_hash(s: str) -> int:
    """Process-stable hash (python's hash() is salted per run) — split must not drift."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def img_part(rel_path: str) -> dict:
    p = PROJECT_ROOT / rel_path
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def render_record_block(rec: dict) -> str:
    d, f = rec["description"], rec["features"]
    return (
        f"CHUNK AUDIO (verified): {d['caption']}\n"
        f"- instruments: {', '.join(d['instruments'])}\n"
        f"- mood: {', '.join(d['mood'])} | genre: {d['genre']} | energy: {d['energy']}\n"
        f"- tempo: {f['bpm']} BPM, key {f['key']} | effects: {', '.join(d['effects']) or 'none'}\n"
        f"- vocals: {'yes' if d['vocals'] else 'no'}\n"
        f"- position: chunk {rec['chunk_index'] + 1}/{rec['n_chunks']} ({rec['position']}) "
        f"of the song, {rec['duration_s']:.0f}s long"
    )


def build_generator_messages(rec: dict, conv_type: str, image_rel: str | None,
                             image_cat: str | None, n_turns: int) -> list[dict]:
    block = render_record_block(rec)
    task: list[str] = [block, ""]

    if conv_type == "no_edit":
        task.append(
            f"Write a {n_turns}-turn conversation where the user just asks about this "
            "chunk (what instruments play, the tempo, the vibe, whether it fits some "
            "purpose...) and the assistant answers from the audio description. "
            "NO edit is requested or performed anywhere.")
    else:
        task.append(
            f"The edit the user ends up requesting (reach exactly this, in the user's "
            f"own casual words): \"{rec['edit_instruction']}\"")
        if conv_type == "text":
            extra = ("The user may start vague; " if n_turns >= 4 else "") + (
                "the assistant may ask ONE clarifying question before confirming."
                if n_turns >= 4 else "Keep it direct: request then confirmation.")
            task.append(f"Write a {n_turns}-turn text conversation. {extra}")
        elif conv_type == "image_mood":
            task.append(
                f"The user ALSO shares the attached image (a {CATEGORY_EN[image_cat]} "
                "painting) as the scene this music segment accompanies — they mention "
                "it naturally (\"this part plays over this scene...\") and then ask "
                f"for the edit. Write a {n_turns}-turn conversation; the user's first "
                "message must reference something actually visible in the image.")
        elif conv_type == "image_target":
            task.append(
                f"The user shares the attached image (a {CATEGORY_EN[image_cat]} "
                "painting) as the TARGET feeling: they ask to make the chunk feel like "
                "this picture, which corresponds to the edit above. The assistant "
                "connects what it sees in the image to the musical change. "
                f"Write a {n_turns}-turn conversation.")
        elif conv_type == "game_scene":
            task.append(
                "The user is a game developer; the attached image "
                f"(a {CATEGORY_EN[image_cat]} painting) is concept art for a scene or "
                "level in their game, and this chunk is the scene's music. They "
                "describe the scene briefly and ask for the edit so the music fits "
                f"the gameplay moment better. Write a {n_turns}-turn conversation.")
        task.append(
            "The final assistant turn: one short natural sentence confirming the "
            "exact agreed edit.")

    content: list[dict] = []
    if image_rel is not None:
        content.append(img_part(image_rel))
    content.append({"type": "text", "text": "\n".join(task)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def validate_turns(turns: list[dict], has_edit: bool) -> str | None:
    if len(turns) < 2 or len(turns) % 2 != 0:
        return f"bad_turn_count:{len(turns)}"
    for i, t in enumerate(turns):
        want = "user" if i % 2 == 0 else "assistant"
        if t["role"] != want:
            return f"bad_role_order:turn{i}"
        if not t["content"].strip():
            return f"empty_turn:{i}"
        if "[EDIT" in t["content"]:
            return f"llm_emitted_block:turn{i}"
    if not has_edit:
        joined = " ".join(t["content"].lower() for t in turns)
        if "i'll" in joined and "edit" in joined:
            return None  # tolerable; heuristic only
    return None


def assign_conv_type(rec: dict, rng: random.Random) -> tuple[str, str | None]:
    """Returns (conv_type, image_category_or_None). Deterministic per record."""
    r = rng.random()
    if r < 0.10:
        return "no_edit", None
    if r < 0.40:  # 30% image conversations
        if rec["edit_type"] == "mood_shift" or rng.random() < 0.25:
            flavor = "image_target"
        else:
            flavor = "game_scene" if rng.random() < 0.33 else "image_mood"
        # category: target mood for image_target (from instruction), else chunk mood
        if flavor == "image_target":
            cat = mood_to_category([rec["edit_instruction"]], rec["description"]["energy"])
        else:
            cat = mood_to_category(rec["description"]["mood"], rec["description"]["energy"])
        return flavor, cat
    return "text", None


def build_omni_messages(rec: dict, turns: list[dict], image_rel: str | None,
                        has_edit: bool) -> list[dict]:
    """Store in Qwen2.5-Omni chat format: audio (+image) parts on the first user turn."""
    messages = []
    for i, t in enumerate(turns):
        if i == 0:
            content = [{"type": "audio", "audio": rec["chunk_path"]}]
            if image_rel is not None:
                content.append({"type": "image", "image": image_rel})
            content.append({"type": "text", "text": t["content"]})
        else:
            text = t["content"]
            if has_edit and i == len(turns) - 1:
                text = text.rstrip() + " " + EDIT_BLOCK
            content = [{"type": "text", "text": text}]
        messages.append({"role": t["role"], "content": content})
    return messages


def process_record(item: tuple[dict, str, str | None], pool: ImagePool) -> dict:
    rec, conv_type, image_cat = item
    rng = random.Random(rec["edit_id"] + ":conv")
    has_edit = conv_type != "no_edit"

    image_rel = None
    if image_cat is not None:
        if rec["source"] == "painting":
            # painting song_id format: paint_<idx>_<music_stem>
            music_stem = rec["song_id"].split("_", 2)[2]
            image_rel = pool.paired_image(music_stem, rng)
        if image_rel is None:
            image_rel = pool.pick(image_cat, rng)
        image_cat = pool.category_of(image_rel) or image_cat

    n_turns = rng.choice([2, 2, 4]) if conv_type == "text" else rng.choice([2, 4])
    if conv_type == "no_edit":
        n_turns = rng.choice([2, 4])

    last_err = None
    for attempt in range(2):
        gen_messages = build_generator_messages(rec, conv_type, image_rel, image_cat, n_turns)
        try:
            result = chat_json(gen_messages, DIALOGUE_SCHEMA, schema_name="dialogue",
                               max_tokens=1500, temperature=0.8 + 0.1 * attempt)
        except LLMError as exc:
            last_err = str(exc)
            continue
        turns = result["turns"]
        err = validate_turns(turns, has_edit)
        if err is None:
            return {
                "id": rec["edit_id"],
                "song_id": rec["song_id"],
                "chunk_id": rec["chunk_id"],
                "conv_type": conv_type,
                "image_path": image_rel,
                "image_category": image_cat,
                "chunk_path": rec["chunk_path"],
                "edited_path": rec["edited_path"] if has_edit else None,
                "edit_instruction": rec["edit_instruction"] if has_edit else None,
                "edit_type": rec["edit_type"] if has_edit else None,
                "has_edit": has_edit,
                "messages": build_omni_messages(rec, turns, image_rel, has_edit),
                "split": "val" if stable_hash(rec["song_id"]) % 20 == 0 else "train",
                "llm_model": LLM_MODEL,
            }
        last_err = err
    raise LLMError(f"dialogue invalid after retries: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    records = [json.loads(l) for l in open(DATASET)]
    records = [r for r in records if r["qa"]["passed"]]
    print(f"{len(records)} qa-passed records")

    done: set[str] = set()
    if OUT_PATH.exists() and not args.overwrite:
        done = {json.loads(l)["id"] for l in open(OUT_PATH)}
        print(f"resuming: {len(done)} dialogues already generated")

    pool = ImagePool()
    items = []
    for rec in records:
        if rec["edit_id"] in done:
            continue
        rng = random.Random(rec["edit_id"] + ":type")
        conv_type, image_cat = assign_conv_type(rec, rng)
        items.append((rec, conv_type, image_cat))
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} conversations to generate")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    n_ok = n_err = 0
    with open(OUT_PATH, "w" if args.overwrite else "a") as out:
        for item, result, err in map_concurrent(
                lambda it: process_record(it, pool), items,
                workers=args.workers, desc="dialogues"):
            if err is not None:
                n_err += 1
                print(f"\n[error] {item[0]['edit_id']}: {err}")
                continue
            with lock:
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
            n_ok += 1
    print(f"\ndone: {n_ok} ok, {n_err} failed" + (" (rerun to retry)" if n_err else ""))


if __name__ == "__main__":
    main()
