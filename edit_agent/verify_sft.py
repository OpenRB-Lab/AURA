"""Emission verification for an SFT'd thinker checkpoint.

Checks on held-out val dialogues, per domain:
- single-edit: assistant reply ends with the correct typed block
  [EDIT_<KIND>][EDIT_0]...[EDIT_7] (kind must match the reference turn)
- multi-turn (slakh_traj / inp_session): every assistant edit turn, teacher-forced
- no-edit: no [EDIT_*] tokens emitted (false-positive check)

Usage:
  conda run --no-capture-output -n llama python -u src/edit_agent/verify_sft.py \
      --adapter ckpts/edit_agent/sft/final
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.qwen_wrapper import load_thinker, load_audio_16k  # noqa: E402
from edit_agent.sft_data import SYSTEM_PROMPT  # noqa: E402
from edit_agent.tokens import EDIT_BLOCK  # noqa: E402

DIALOGUES = PROJECT_ROOT / "data/edit_dataset/dialogues/dialogues.jsonl"
IM_END = 151645
KIND_RE = re.compile(r"\[EDIT_([A-Z]+)\](?=\[EDIT_0\])")

DOMAINS = {
    "music": lambda r: r["conv_type"] in ("text", "image_mood", "image_target", "game_scene"),
    "audioedit": lambda r: r["conv_type"].startswith("ae_") and r["conv_type"] != "ae_no_edit",
    "slakh": lambda r: r["conv_type"].startswith("slakh_") and r["conv_type"] != "slakh_traj",
    "inpaint": lambda r: r["conv_type"].startswith("inp_") and r["conv_type"] != "inp_session",
}
MULTI = ("slakh_traj", "inp_session")
NOEDIT = ("no_edit", "ae_no_edit")


def ref_kind(text: str) -> str | None:
    m = KIND_RE.search(text)
    return m.group(1) if m else None


def generate_turn(model, processor, messages, audio, image, device):
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], audio=[audio],
                       images=[image] if image is not None else None,
                       return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                             eos_token_id=IM_END)
    new = out[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new, skip_special_tokens=False) \
        .replace("<|im_end|>", "").strip()


def check_dialogue(model, processor, r, device):
    """Yield (turn_desc, expected_kind|None, generated_text) per assistant turn."""
    audio = load_audio_16k(r["chunk_path"])
    image = None
    if r.get("image_path"):
        from PIL import Image
        image = Image.open(PROJECT_ROOT / r["image_path"]).convert("RGB")
    ctx = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    for msg in r["messages"]:
        if msg["role"] == "assistant":
            ref = " ".join(c["text"] for c in msg["content"] if c.get("type") == "text")
            gen = generate_turn(model, processor, ctx, audio, image, device)
            yield ref_kind(ref), gen
        ctx.append(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="ckpts/edit_agent/sft/final")
    ap.add_argument("--per-domain", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    val = [json.loads(l) for l in open(DIALOGUES)]
    val = [r for r in val if r["split"] == "val"]

    model, processor, _ = load_thinker(lora_dir=str(PROJECT_ROOT / args.adapter),
                                       device=args.device)
    model.eval()

    results = {}
    # single-edit per domain
    for name, pred in DOMAINS.items():
        rows = [r for r in val if pred(r) and r.get("has_edit", True)
                and r["conv_type"] not in NOEDIT][:args.per_domain]
        ok = kind_ok = 0
        for r in rows:
            for exp_kind, gen in check_dialogue(model, processor, r, args.device):
                emitted = EDIT_BLOCK in gen
                gkind = ref_kind(gen)
                ok += emitted
                kind_ok += (emitted and gkind == exp_kind)
                print(f"[{name}] {r['id']}: block={'Y' if emitted else 'N'} "
                      f"kind={gkind} (want {exp_kind})", flush=True)
        results[name] = (ok, kind_ok, len(rows))

    # multi-turn
    for ct in MULTI:
        rows = [r for r in val if r["conv_type"] == ct][:2]
        ok = kind_ok = tot = 0
        for r in rows:
            for exp_kind, gen in check_dialogue(model, processor, r, args.device):
                if exp_kind is None:  # non-edit intermediate turn
                    continue
                tot += 1
                emitted = EDIT_BLOCK in gen
                gkind = ref_kind(gen)
                ok += emitted
                kind_ok += (emitted and gkind == exp_kind)
                print(f"[{ct}] {r['id']} turn: block={'Y' if emitted else 'N'} "
                      f"kind={gkind} (want {exp_kind})", flush=True)
        results[ct] = (ok, kind_ok, tot)

    # false positives
    rows = [r for r in val if r["conv_type"] in NOEDIT][:args.per_domain]
    fp = 0
    for r in rows:
        for _, gen in check_dialogue(model, processor, r, args.device):
            hit = "[EDIT_" in gen
            fp += hit
            print(f"[no-edit] {r['id']}: {'FP!' if hit else 'clean'}", flush=True)
    results["no_edit_fp"] = (fp, 0, len(rows))

    print("\n=== summary ===")
    for name, (ok, kind_ok, tot) in results.items():
        if name == "no_edit_fp":
            print(f"{name}: {ok}/{tot} false positives")
        else:
            print(f"{name}: block {ok}/{tot}, correct kind {kind_ok}/{tot}")


if __name__ == "__main__":
    main()
