"""Render segment-inpainting targets with Stable Audio 3 medium (runs in `sa3` env).

Reads inpaint_plan.jsonl, and for each plan regenerates ONLY the planned segment of
the source chunk via SA3's native time-masked inpainting (everything outside the mask
is preserved by construction). Writes 44.1 kHz stereo FLAC targets + manifest.

Usage:
  SA3_DEVICE=cuda:0 python src/edit_agent/render_inpaint.py --limit 8   # smoke
  python src/edit_agent/render_inpaint.py
"""

import argparse
import json
import os
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = PROJECT_ROOT / "data/edit_dataset/inpaint/inpaint_plan.jsonl"
OUT_DIR = PROJECT_ROOT / "data/edit_dataset/inpaint/pairs"
MANIFEST = PROJECT_ROOT / "data/edit_dataset/inpaint/inpaint_pairs.jsonl"

DEVICE = os.environ.get("SA3_DEVICE", "cuda:0")
STEPS = 8          # adversarially post-trained checkpoint defaults
CFG = 1.0
TARGET_SR = 44100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=str, default=None, help="i/n")
    args = parser.parse_args()

    plans = [json.loads(l) for l in open(PLAN_PATH)]
    done = set()
    if MANIFEST.exists():
        done = {json.loads(l)["plan_id"] for l in open(MANIFEST)
                if json.loads(l).get("status") == "ok"}
    pending = [p for p in plans if p["plan_id"] not in done]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        pending = [p for k, p in enumerate(pending) if k % n == i]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(pending)} renders pending ({len(done)} done)", flush=True)
    if not pending:
        return

    from stable_audio_3 import StableAudioModel
    print("loading stable-audio-3 medium ...", flush=True)
    model = StableAudioModel.from_pretrained("medium")
    print("model ready", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a") as mf:
        for k, p in enumerate(pending):
            t0 = time.time()
            try:
                wav, sr = torchaudio.load(PROJECT_ROOT / p["audio_path"])
                if wav.shape[0] == 1:
                    wav = wav.repeat(2, 1)
                out = model.generate(
                    prompt=p["inpaint_prompt"],
                    duration=p["duration_s"],
                    steps=STEPS, cfg_scale=CFG,
                    inpaint_audio=(sr, wav),
                    inpaint_mask_start_seconds=p["segment_start_s"],
                    inpaint_mask_end_seconds=p["segment_end_s"],
                    seed=k,
                )
                audio = out[0].float().cpu()
                if audio.abs().max() > 1:
                    audio = audio / audio.abs().max()
                # trim/pad to source length so pairs stay aligned
                n_src = int(p["duration_s"] * TARGET_SR)
                audio = audio[..., :n_src]
                tgt = OUT_DIR / p["source_id"] / f"{p['plan_id']}_out.flac"
                tgt.parent.mkdir(parents=True, exist_ok=True)
                sf.write(tgt, audio.T.numpy(), TARGET_SR, format="FLAC")
                mf.write(json.dumps({
                    "plan_id": p["plan_id"], "target_path": str(tgt.relative_to(PROJECT_ROOT)),
                    "status": "ok", "gen_time_s": round(time.time() - t0, 2),
                    "sa3_params": {"model": "stable-audio-3-medium", "steps": STEPS,
                                   "cfg_scale": CFG},
                }) + "\n")
                mf.flush()
                if (k + 1) % 25 == 0:
                    print(f"{k + 1}/{len(pending)} rendered "
                          f"({time.time() - t0:.1f}s last)", flush=True)
            except Exception as exc:  # noqa: BLE001 — log and continue
                mf.write(json.dumps({"plan_id": p["plan_id"],
                                     "status": f"error: {str(exc)[:200]}"}) + "\n")
                mf.flush()
                print(f"[error] {p['plan_id']}: {exc}", flush=True)
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
