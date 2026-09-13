"""QA report for the synthesized dialogue dataset.

Usage: python src/edit_agent/validate_dialogues.py [--samples 20]
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.tokens import EDIT_BLOCK  # noqa: E402

DIALOGUES = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"


def text_of(msg: dict) -> str:
    return " ".join(p["text"] for p in msg["content"] if p.get("type") == "text")


def check(d: dict) -> list[str]:
    errs = []
    msgs = d["messages"]
    if len(msgs) % 2 != 0 or len(msgs) < 2:
        errs.append(f"turn_count:{len(msgs)}")
    for i, m in enumerate(msgs):
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            errs.append(f"role_order:{i}")
    first = msgs[0]["content"]
    if first[0].get("type") != "audio":
        errs.append("no_audio_first_turn")
    has_img_part = any(p.get("type") == "image" for p in first)
    if bool(d["image_path"]) != has_img_part:
        errs.append("image_part_mismatch")
    final = text_of(msgs[-1])
    if d.get("conv_type") in ("slakh_traj", "inp_session"):
        # every assistant turn confirms a step and must end with the block
        for i, m in enumerate(msgs):
            if m["role"] == "assistant" and not text_of(m).rstrip().endswith(EDIT_BLOCK):
                errs.append(f"traj_block_missing:turn{i}")
    elif d["has_edit"]:
        if not final.rstrip().endswith(EDIT_BLOCK):
            errs.append("block_missing_or_not_final")
        if sum(t.count("[EDIT_0]") for t in map(text_of, msgs)) != 1:
            errs.append("block_count!=1")
    else:
        if "[EDIT" in " ".join(map(text_of, msgs)):
            errs.append("block_in_no_edit")
    if not (PROJECT_ROOT / d["chunk_path"]).exists():
        errs.append("chunk_missing")
    if d["image_path"] and not (PROJECT_ROOT / d["image_path"]).exists():
        errs.append("image_missing")
    return errs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(DIALOGUES)]
    print(f"{len(rows)} dialogues ({len({r['id'] for r in rows})} unique ids)")
    print("by conv_type:", dict(Counter(r["conv_type"] for r in rows)))
    print("by split:", dict(Counter(r["split"] for r in rows)))
    print("has_edit:", dict(Counter(r["has_edit"] for r in rows)))
    print("turns histogram:", dict(sorted(Counter(len(r["messages"]) for r in rows).items())))
    print("image categories:", dict(Counter(r["image_category"] for r in rows if r["image_path"])))
    print("edit types:", dict(Counter(r["edit_type"] for r in rows if r["has_edit"])))

    bad = [(r["id"], errs) for r in rows if (errs := check(r))]
    print(f"\nvalidation: {len(rows) - len(bad)} clean, {len(bad)} with issues")
    for rid, errs in bad[:10]:
        print("  ", rid, errs)

    rng = random.Random(0)
    print("\n--- random samples ---")
    for r in rng.sample(rows, min(args.samples, len(rows))):
        print(f"\n[{r['id']}] type={r['conv_type']} split={r['split']} "
              f"image={r['image_category'] or '-'}")
        if r["has_edit"]:
            print(f"  target edit: {r['edit_instruction']}")
        for m in r["messages"]:
            print(f"  {m['role']:9s}: {text_of(m)[:180]}")


if __name__ == "__main__":
    main()
