"""Benchmark runner for the stem-decomposition pipeline.

Same manifest/protocol as run_impg_bench.py (which stays frozen for the
full-mix baselines), but generation goes through stem_pipeline executors.
Writes <task>/<out-name>/NNN.wav plus <out-name>_plan_log.jsonl per bench dir.

Usage:
  BENCH_DIR=results/impg_bench conda run -n llama python -u \
      src/edit_agent/run_stem_bench.py --executor hybrid \
      --planner programmatic --out-name output_hybrid_prog
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

torch.backends.cudnn.enabled = False
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from edit_agent.musicgen_fusion import build_fusion_bridge  # noqa: E402
from edit_agent.qwen_wrapper import (  # noqa: E402
    generate_with_edit_capture, load_audio_16k, load_thinker)
from edit_agent.sft_data import SYSTEM_PROMPT  # noqa: E402
from edit_agent import stem_pipeline as sp  # noqa: E402

BENCH = PROJECT_ROOT / os.environ.get("BENCH_DIR", "results/impg_bench")
SR = 32000
IM_END = 151645


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qwen", default="ckpts/edit_agent/joint_fusion_r64/final/qwen")
    ap.add_argument("--mg", default="ckpts/edit_agent/joint_fusion_r64/final/musicgen")
    ap.add_argument("--classifier",
                    default="ckpts/edit_agent/joint_cls/final/classifier.pt")
    ap.add_argument("--executor", choices=["hybrid", "pure", "full"],
                    default="hybrid")
    ap.add_argument("--planner", choices=["programmatic", "cot"],
                    default="programmatic")
    ap.add_argument("--stem-gen", action="store_true",
                    help="hybrid add: generate the stem directly (needs a "
                         "stem-mode bridge checkpoint in --mg)")
    ap.add_argument("--limit-per-task", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="per-example seed = seed_base + idx (multi-seed runs)")
    ap.add_argument("--tasks", default=None,
                    help="comma-separated subset of add,remove,extract")
    ap.add_argument("--disable", default=None,
                    help="inference-time ablation: comma list of film,bifam "
                         "(film -> gate=0, exact no-modulation; bifam -> "
                         "alpha1=alpha2=0, no reference signal into FiLM)")
    ap.add_argument("--guidance", type=float, default=2.0,
                    help="CFG scale for the full executor (1.0 = CFG off)")
    ap.add_argument("--ref-fusion", default=None,
                    choices=["bifam", "none", "crossattn", "concat", "both"],
                    help="override the checkpoint's reference-fusion mechanism")
    ap.add_argument("--use-template", action="store_true")
    ap.add_argument("--out-name", default="output_hybrid_prog")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    manifest = json.load(open(BENCH / "manifest.json"))
    if args.tasks:
        keep_tasks = set(args.tasks.split(","))
        manifest = [m for m in manifest if m["task"] in keep_tasks]
    if args.limit_per_task:
        keep, seen = [], {}
        for m in manifest:
            if seen.get(m["task"], 0) < args.limit_per_task:
                keep.append(m)
                seen[m["task"]] = seen.get(m["task"], 0) + 1
        manifest = keep

    model, proc, edit_ids = load_thinker(
        lora_dir=str(PROJECT_ROOT / args.qwen), merge_lora=True, device=dev)
    model.eval()
    bridge = build_fusion_bridge(torch.device(dev))
    bridge.load_adapter(PROJECT_ROOT / args.mg, dev)
    if args.ref_fusion:
        bridge.ref_fusion = args.ref_fusion
    print(f"[ref-fusion] {bridge.ref_fusion}", flush=True)
    if args.disable:
        dis = set(args.disable.split(","))
        if "film" in dis:
            bridge.gate.data.zero_()          # tanh(0)=0 -> o_m unchanged
        if "bifam" in dis:
            bridge.alpha1.data.zero_()         # no reference-stream signal
            bridge.alpha2.data.zero_()
        print(f"[ablate] disabled={sorted(dis)} guidance={args.guidance}",
              flush=True)
    bridge.decoder.to(dev, dtype=torch.bfloat16)
    bridge.eval()
    classifier = sp.load_classifier(PROJECT_ROOT / args.classifier, dev)
    separator = sp.Separator(dev)

    log_p = BENCH / f"{args.out_name}_plan_log.jsonl"
    logged = set()
    if log_p.exists():
        logged = {(r["task"], r["idx"]) for r in
                  (json.loads(l) for l in open(log_p))}
    log_f = open(log_p, "a")

    n_no_edit = 0
    t0 = time.time()
    with torch.no_grad():
        for n, m in enumerate(manifest):
            task, i = m["task"], m["idx"]
            out_dir = BENCH / task / args.out_name
            out_dir.mkdir(exist_ok=True)
            out_p = out_dir / f"{i:03d}.wav"
            if out_p.exists():
                continue
            instruction = (m.get("template") or m["instruction"]) \
                if args.use_template else m["instruction"]
            wav_p = BENCH / task / "input" / f"{i:03d}.wav"
            msgs = [{"role": "system",
                     "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user",
                     "content": [{"type": "audio", "audio": str(wav_p)},
                                 {"type": "text", "text": instruction}]}]
            tmpl = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
            inputs = proc(text=[tmpl], audio=[load_audio_16k(wav_p)],
                          return_tensors="pt", padding=True).to(dev)
            reply, h = generate_with_edit_capture(model, proc, inputs, edit_ids,
                                                  eos_token_id=IM_END)
            if h is None:
                n_no_edit += 1
                print(f"[no-edit] {task}/{i:03d}: {reply!r}", flush=True)
                continue

            plan = sp.parse_plan(reply) if args.planner == "cot" else None
            if plan is None:
                plan = sp.programmatic_plan(reply, h, classifier,
                                            instruction=instruction)
                if args.planner == "cot":
                    plan.fallback_reason = "plan parse failed"

            y, _ = sf.read(wav_p, dtype="float32")
            seed = args.seed_base + i
            if args.executor == "full":
                wav = sp.full_mix_execute(h, y, bridge, seed,
                                          guidance=args.guidance)
            elif args.executor == "pure":
                def thinker_fn(instr):
                    ms = [{"role": "system",
                           "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                          {"role": "user",
                           "content": [{"type": "audio", "audio": str(wav_p)},
                                       {"type": "text", "text": instr}]}]
                    tm = proc.apply_chat_template(ms, tokenize=False,
                                                  add_generation_prompt=True)
                    ins = proc(text=[tm], audio=[load_audio_16k(wav_p)],
                               return_tensors="pt", padding=True).to(dev)
                    _, hh = generate_with_edit_capture(model, proc, ins,
                                                       edit_ids,
                                                       eos_token_id=IM_END)
                    return hh
                wav, plan = sp.pure_execute(plan, y, h, bridge, separator,
                                            seed, thinker_fn)
            else:
                wav, plan = sp.hybrid_execute(plan, y, h, bridge, separator,
                                              seed, stem_gen=args.stem_gen)
            sf.write(out_p, wav[: int(5.0 * SR)], SR, subtype="PCM_16")
            if (task, i) not in logged:
                log_f.write(json.dumps(
                    {"task": task, "idx": i, **plan.to_json()}) + "\n")
                log_f.flush()
            if (n + 1) % 25 == 0:
                el = time.time() - t0
                print(f"{n + 1}/{len(manifest)} ({el / (n + 1):.1f}s/ex)",
                      flush=True)
    log_f.close()
    print(f"done: {len(manifest)} examples, {n_no_edit} no-edit skips",
          flush=True)


if __name__ == "__main__":
    main()
