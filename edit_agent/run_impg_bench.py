"""Generate IMPG-benchmark outputs with the production system.

For each example: thinker (live, merged LoRA) on (input.wav, instruction) ->
edit-token hidden states -> fusion bridge (5 s = 250 frames) -> EnCodec decode.

Usage:
  conda run -n llama python -u src/edit_agent/run_impg_bench.py \
      [--limit-per-task N] [--qwen ...] [--mg ...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
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

BENCH = PROJECT_ROOT / "results/impg_bench"
SR = 32000
N_FRAMES = 250  # 5 s at 50 Hz
IM_END = 151645


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qwen", default="ckpts/edit_agent/joint_fusion_r64/final/qwen")
    ap.add_argument("--mg", default="ckpts/edit_agent/joint_fusion_r64/final/musicgen")
    ap.add_argument("--limit-per-task", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    manifest = json.load(open(BENCH / "manifest.json"))
    if args.limit_per_task:
        keep = []
        seen: dict[str, int] = {}
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
    bridge.decoder.to(dev, dtype=torch.bfloat16)
    bridge.eval()

    n_no_edit = 0
    t0 = time.time()
    with torch.no_grad():
        for n, m in enumerate(manifest):
            task, i = m["task"], m["idx"]
            out_dir = BENCH / task / "output"
            out_dir.mkdir(exist_ok=True)
            out_p = out_dir / f"{i:03d}.wav"
            if out_p.exists():
                continue
            wav_p = BENCH / task / "input" / f"{i:03d}.wav"
            msgs = [{"role": "system",
                     "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user",
                     "content": [{"type": "audio", "audio": str(wav_p)},
                                 {"type": "text", "text": m["instruction"]}]}]
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
            y, _ = sf.read(wav_p, dtype="float32")
            src_wav = torch.from_numpy(y).to(dev).view(1, 1, -1)
            codes = bridge.audio_encoder.encode(src_wav, bandwidth=None).audio_codes
            src = (codes[0] if codes.dim() == 4 else codes)[0, :, :N_FRAMES]
            gen = bridge.generate(h.float().unsqueeze(0).to(dev),
                                  src.long().unsqueeze(0),
                                  max_frames=N_FRAMES, guidance=2.0, seed=1000 + i)
            wav = bridge.decode_audio(gen).squeeze().float().cpu().numpy()
            p = np.abs(wav).max()
            wav = wav / p * 0.95 if p > 0 else wav
            sf.write(out_p, wav[: int(5.0 * SR)], SR, subtype="PCM_16")
            if (n + 1) % 25 == 0:
                el = time.time() - t0
                print(f"{n + 1}/{len(manifest)} ({el / (n + 1):.1f}s/ex)", flush=True)
    print(f"done: {len(manifest)} examples, {n_no_edit} no-edit skips", flush=True)


if __name__ == "__main__":
    main()
