"""Retype bare [EDIT_0..7] blocks in Slakh dialogues to typed blocks.

The Slakh dialogue synthesizer historically appended the untyped EDIT_BLOCK;
this rewrites every slakh_* dialogue whose assistant text contains a bare
block (no preceding [EDIT_<KIND>]) to use typed_block(op). Trajectory rows
use each step's own op when available.

Writes backup dialogues_pre_retype.jsonl.bak, prints counts, and emits the
affected bridge example ids (whose cached hidden states become stale) to
data/edit_dataset/bridge_cache/retyped_ids.json.

Usage: conda run -n llama python -u src/edit_agent/fix_typed_blocks.py
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.tokens import EDIT_BLOCK, EDIT_TYPE_TOKENS, typed_block  # noqa: E402

DIALOGUES = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"
BACKUP = DIALOGUES.parent / "dialogues_pre_retype.jsonl.bak"
CACHE = PROJECT_ROOT / "data/edit_dataset/bridge_cache"

KIND_TOKENS = set(EDIT_TYPE_TOKENS.values())
BARE_RE = re.compile(re.escape(EDIT_BLOCK))


def retype_text(text: str, op: str) -> tuple[str, bool]:
    """Replace a bare EDIT_BLOCK (not already kind-prefixed) with typed_block."""
    idx = text.find(EDIT_BLOCK)
    if idx < 0:
        return text, False
    prefix = text[:idx].rstrip()
    if any(prefix.endswith(k) for k in KIND_TOKENS):
        return text, False
    return text[:idx] + typed_block(op) + text[idx + len(EDIT_BLOCK):], True


def main():
    if not BACKUP.exists():
        BACKUP.write_bytes(DIALOGUES.read_bytes())
        print(f"backup -> {BACKUP.name}", flush=True)
    else:
        print(f"backup already exists ({BACKUP.name}), keeping it", flush=True)

    rows = [json.loads(l) for l in open(DIALOGUES)]
    n_rows_changed = n_blocks = 0
    changed_ids = []
    for r in rows:
        et = r.get("edit_type") or r.get("conv_type") or ""
        if not str(et).startswith("slakh"):
            continue
        steps = r.get("steps")
        edit_turn = 0
        row_changed = False
        for msg in r["messages"]:
            if msg["role"] != "assistant":
                continue
            for c in msg["content"]:
                if c.get("type") != "text" or EDIT_BLOCK not in c["text"]:
                    continue
                if steps and edit_turn < len(steps):
                    op = steps[edit_turn].get("op") or et
                else:
                    op = et
                new, did = retype_text(c["text"], str(op))
                if did:
                    c["text"] = new
                    n_blocks += 1
                    row_changed = True
                edit_turn += 1
        if row_changed:
            n_rows_changed += 1
            changed_ids.append(r["id"])

    with open(DIALOGUES, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # affected bridge example ids: single-edit -> id; trajectory -> id__s{i}
    ex_ids = []
    for r in rows:
        if r["id"] not in set(changed_ids):
            continue
        if r.get("steps"):
            ex_ids += [f"{r['id']}__s{i}" for i in range(len(r["steps"]))]
        else:
            ex_ids.append(r["id"])
    json.dump(ex_ids, open(CACHE / "retyped_ids.json", "w"))
    print(f"retyped {n_blocks} blocks in {n_rows_changed} dialogues; "
          f"{len(ex_ids)} bridge example ids affected", flush=True)


if __name__ == "__main__":
    main()
