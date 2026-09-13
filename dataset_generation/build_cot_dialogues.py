"""Build CoT-plan dialogues: prepend a terse parseable stem plan to every
assistant edit turn.

Format (before the natural sentence; the edit block stays terminal):
  <plan> input: drums, bass, keys | action: add guitar | output: copy(mix) + gen(guitar) </plan>

- slakh rows: exact input-stem inventory + instrument from slakh_pairs.jsonl
  (display names -> 10-class taxonomy via build_semantic_labels.inst_from_text).
- other edit rows: coarse plan from bridge_cache/semantic_labels.json
  (`input: unknown`).
- no-edit turns and rows: NO plan (keeps the false-positive check meaningful).

Output: data/edit_dataset/dialogues/dialogues_cot.jsonl
Usage: conda run -n llama python -u src/edit_agent/build_cot_dialogues.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.build_semantic_labels import inst_from_text  # noqa: E402

DIALOGUES = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"
OUT = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues_cot.jsonl"
PAIRS = PROJECT_ROOT / "data/edit_dataset/slakh/slakh_pairs.jsonl"
LABELS = PROJECT_ROOT / "data/edit_dataset/bridge_cache/semantic_labels.json"

KIND_RE = re.compile(r"\[EDIT_([A-Z]+)\](?=\[EDIT_0\])")

# kind -> output spec template ({i} = instrument)
OUTPUT_SPEC = {"ADD": "copy(mix) + gen({i})", "REMOVE": "sep(mix) - {i}",
               "EXTRACT": "sep({i})"}
GEN_MIX = "gen(mix)"


def classes_of(names: list[str]) -> list[str]:
    seen = []
    for n in names:
        c = inst_from_text(n)
        if c and c not in seen:
            seen.append(c)
    return seen


def plan_str(input_insts: list[str] | None, kind: str, inst: str | None) -> str:
    inp = ", ".join(input_insts) if input_insts else "unknown"
    action = f"{kind.lower()} {inst}" if inst else kind.lower()
    spec = OUTPUT_SPEC.get(kind, GEN_MIX).format(i=inst) if inst else GEN_MIX
    return f"<plan> input: {inp} | action: {action} | output: {spec} </plan>"


def slakh_plan(pair: dict, kind: str) -> str:
    inp = classes_of(pair["input_stems"])
    op = pair["op"]
    if op == "add":
        diff = [s for s in pair["target_stems"] if s not in set(pair["input_stems"])]
        inst = inst_from_text(", ".join(diff))
    elif op == "remove":
        diff = [s for s in pair["input_stems"] if s not in set(pair["target_stems"])]
        inst = inst_from_text(", ".join(diff))
    elif op == "isolate":
        inst = inst_from_text(", ".join(pair["target_stems"]))
    elif op == "rebalance":
        inst = inst_from_text(", ".join(pair["gains_db"].keys()))
    else:  # swap: name the incoming stem
        diff = [s for s in pair["target_stems"] if s not in set(pair["input_stems"])]
        inst = inst_from_text(", ".join(diff))
    return plan_str(inp, kind, inst)


def main():
    pairs = {}
    for l in open(PAIRS):
        r = json.loads(l)
        pairs[r["pair_id"]] = r
    labels = json.load(open(LABELS))

    stats = Counter()
    with open(OUT, "w") as f:
        for l in open(DIALOGUES):
            r = json.loads(l)
            is_slakh = r["conv_type"].startswith("slakh_")
            edit_turn = 0
            for m in r["messages"]:
                if m["role"] != "assistant":
                    continue
                for c in m["content"]:
                    if c.get("type") != "text" or "[EDIT_" not in c.get("text", ""):
                        continue
                    text = c["text"]
                    km = KIND_RE.search(text)
                    if not km:
                        continue
                    kind = km.group(1)
                    plan = None
                    if is_slakh:
                        base = r["id"][len("slakh_"):]
                        pid = f"{base}_s{edit_turn}" \
                            if r["conv_type"] == "slakh_traj" else base
                        pair = pairs.get(pid)
                        if pair:
                            plan = slakh_plan(pair, kind)
                            stats["slakh"] += 1
                    if plan is None:
                        lab = labels.get(r["id"]) or \
                            labels.get(f"{r['id']}__s{edit_turn}")
                        inst = (lab or {}).get("inst")
                        inst = inst if inst not in (None, "unknown") else None
                        plan = plan_str(None, kind, inst)
                        stats["coarse"] += 1
                    c["text"] = f"{plan} {text}"
                    edit_turn += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats["rows"] += 1
    print(f"done: {stats['rows']} dialogues -> {OUT.name}; "
          f"plans: {stats['slakh']} exact (slakh), {stats['coarse']} coarse",
          flush=True)


if __name__ == "__main__":
    main()
