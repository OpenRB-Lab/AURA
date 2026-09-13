"""Hierarchical semantic labels for the edit-token classifier.

Per bridge example: L1 = edit kind (7, from conv_type via tokens mapping),
L2 = target instrument class (10-way, keyword match over the best available
text source: slakh target stems > edit_instruction > assistant text).

Output: data/edit_dataset/bridge_cache/semantic_labels.json
        {example_id: {"kind": str, "inst": str}}

Usage: conda run -n llama python -u src/edit_agent/build_semantic_labels.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.tokens import kind_of  # noqa: E402

CACHE = PROJECT_ROOT / "data/edit_dataset/bridge_cache"
DIALOGUES = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"
SLAKH_PAIRS = PROJECT_ROOT / "data/edit_dataset/slakh/slakh_pairs.jsonl"

INST_CLASSES = ["drums", "bass", "guitar", "keys", "strings", "brass_wind",
                "synth", "vocals", "sfx_other", "none"]

KEYWORDS = {
    "drums": ["drum", "kick", "snare", "hi-hat", "hihat", "cymbal", "percussion",
              "beat", "tom-tom", "timpani", "clap"],
    "bass": ["bass"],
    "guitar": ["guitar", "banjo", "ukulele", "mandolin"],
    "keys": ["piano", "keys", "organ", "harpsichord", "clavinet", "celesta",
             "accordion", "keyboard", "rhodes", "e-piano"],
    "strings": ["string", "violin", "viola", "cello", "fiddle", "harp",
                "pizzicato", "orchestra"],
    "brass_wind": ["brass", "trumpet", "trombone", "horn", "tuba", "sax",
                   "flute", "clarinet", "oboe", "bassoon", "wind", "whistle",
                   "recorder", "pipe", "harmonica"],
    "synth": ["synth", "pad", "lead", "sawtooth", "square", "arp", "sub-bass",
              "808", "music box", "bell", "chime", "glockenspiel", "vibraphone",
              "marimba", "xylophone"],
    "vocals": ["vocal", "voice", "choir", "aahs", "oohs", "sing", "vox",
               "acapella"],
    "sfx_other": ["sound effect", "sfx", "noise", "ambience", "ambient sound",
                  "nature", "rain", "bird", "crowd", "applause", "fx"],
}


import re as _re
_BASS_RE = _re.compile(r"\bbass\b|\bsub-?bass\b|\b808\b")


def inst_from_text(text: str) -> str | None:
    t = (text or "").lower()
    # brass/wind first so "brass"/"bassoon" never fall into bass
    for cls in ["brass_wind", "drums", "vocals", "strings", "keys",
                "synth", "guitar", "sfx_other"]:
        if any(k in t for k in KEYWORDS[cls]):
            return cls
    if _BASS_RE.search(t):
        return "bass"
    return None


def main():
    dialogues = {}
    for line in open(DIALOGUES):
        d = json.loads(line)
        dialogues[d["id"]] = d
    slakh = {}
    for line in open(SLAKH_PAIRS):
        p = json.loads(line)
        slakh[f"slakh_{p['pair_id']}"] = p

    labels = {}
    cov = Counter()
    kinds = Counter()
    for line in open(CACHE / "examples.jsonl"):
        e = json.loads(line)
        eid = e["example_id"]
        conv = e["conv_type"]
        d0 = dialogues.get(e.get("dialogue_id") or "")
        kind = None
        if d0:
            # authoritative: the typed block actually in the dialogue
            import re as _re2
            texts = " ".join(c.get("text", "") for m in d0["messages"]
                             if m["role"] == "assistant" for c in m["content"])
            blocks = _re2.findall(r"\[EDIT_([A-Z]+)\]\[EDIT_0\]", texts)
            step_i = e.get("step_index")
            if blocks:
                kind = blocks[min(step_i, len(blocks) - 1)] \
                    if step_i is not None else blocks[-1]
        if kind is None:
            kind = kind_of(conv)
        kinds[kind] += 1

        inst = None
        d = dialogues.get(e.get("dialogue_id") or "")
        # 1) slakh pairs: exact target stem names
        base_id = eid.split("__s")[0]
        p = slakh.get(base_id)
        if p is not None:
            step_i = e.get("step_index")
            if step_i is None:
                inst = inst_from_text(" ".join(p["target_stems"])
                                      if p["op"] != "remove" else
                                      p["instruction_seed"])
            else:
                inst = inst_from_text(p["instruction_seed"])
        # 2) template examples
        if inst is None and e.get("template"):
            inst = inst_from_text(e["template"].get("instruction", ""))
        # 3) dialogue edit_instruction
        if inst is None and d is not None:
            inst = inst_from_text(d.get("edit_instruction", ""))
        # 4) assistant text
        if inst is None and d is not None:
            texts = " ".join(c.get("text", "") for m in d["messages"]
                             if m["role"] == "assistant" for c in m["content"])
            inst = inst_from_text(texts)
        # 5) mood/effect/global edits with no instrument target
        if inst is None and kind in ("MOOD", "EFFECT"):
            inst = "none"
        if inst is None:
            inst = "unknown"
        cov[inst] += 1
        labels[eid] = {"kind": kind, "inst": inst}

    json.dump(labels, open(CACHE / "semantic_labels.json", "w"))
    n = len(labels)
    known = n - cov["unknown"]
    print(f"{n} examples labeled; inst coverage {known}/{n} "
          f"({known / n * 100:.1f}%)", flush=True)
    print("kind dist:", dict(kinds.most_common()), flush=True)
    print("inst dist:", dict(cov.most_common()), flush=True)


if __name__ == "__main__":
    main()
