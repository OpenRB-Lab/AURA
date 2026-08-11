"""End-to-end eval of the JOINT thinker + MusicGen bridge.

Per val example: teacher-forced dialogue through the joint thinker -> live [9,3584]
hidden states -> MusicGen bridge AR generation (10 s, CFG) -> decode ->
- MuLan-cos(gen, tgt crop) vs cos(src crop, tgt crop) baseline
- conditioning ablation (zeroed h, same seed) -> code-difference rate
Writes summary JSON + before/after/target triplets to results/musicgen_eval/.

Usage:
  conda run -n llama python -u src/edit_agent/eval_musicgen.py \
      [--qwen ckpts/edit_agent/joint/final/qwen] \
      [--mg ckpts/edit_agent/joint/final/musicgen] [--n-seg 15 --n-global 25]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

torch.backends.cudnn.enabled = False
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from edit_agent.joint_data import JointDataset  # noqa: E402
from edit_agent.musicgen_bridge import build_musicgen_bridge  # noqa: E402
from edit_agent.qwen_wrapper import load_audio_16k, load_thinker  # noqa: E402
from edit_agent.sft_data import SYSTEM_PROMPT  # noqa: E402

DEVICE = "cuda"


def crop(path, win, sr=24000):
    import librosa
    y, _ = librosa.load(PROJECT_ROOT / path, sr=sr, mono=True,
                        offset=win[0], duration=win[1] - win[0])
    return y


@torch.no_grad()
def mulan_of(mulan, y_24k):
    e = mulan(wavs=torch.from_numpy(y_24k.astype("float32")).unsqueeze(0).to(DEVICE))[0]
    return torch.nn.functional.normalize(e.float(), dim=-1)


@torch.no_grad()
def live_hidden(model, proc, edit_ids, ex):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    msgs += ex["messages"]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    images = None
    if ex.get("image_path"):
        from PIL import Image
        images = [Image.open(PROJECT_ROOT / ex["image_path"]).convert("RGB")]
    inputs = proc(text=[text], audio=[load_audio_16k(ex["chunk_path"])],
                  images=images, return_tensors="pt", padding=True).to(DEVICE)
    out = model(**inputs, output_hidden_states=True)
    ids = inputs["input_ids"][0]
    starts = (ids == edit_ids[0]).nonzero(as_tuple=True)[0]
    s = int(starts[-1])
    return out.hidden_states[-1][0, s - 1: s - 1 + 9]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qwen", default="ckpts/edit_agent/joint/final/qwen")
    ap.add_argument("--mg", default="ckpts/edit_agent/joint/final/musicgen")
    ap.add_argument("--n-seg", type=int, default=15)
    ap.add_argument("--n-global", type=int, default=25)
    ap.add_argument("--n-ablate", type=int, default=8)
    ap.add_argument("--n-triplets", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--arch", choices=["mem", "kv", "fusion"], default="mem")
    ap.add_argument("--out", default="results/musicgen_eval")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / args.out
    (out_dir / "triplets").mkdir(parents=True, exist_ok=True)

    ds = JointDataset("val")
    # pick edit rows: localized first, then a spread of global conv types
    seg_idx, glob_by = [], {}
    for i, r in enumerate(ds.rows):
        if r.get("kind") != "edit":
            continue
        ex = None
        enc_path = PROJECT_ROOT / "data/edit_dataset/bridge_cache/encodec" / \
            f"{r['example_id']}.pt"
        if not enc_path.exists():
            continue
        seg = torch.load(enc_path, weights_only=True).get("seg_frames")
        if seg and len(seg_idx) < args.n_seg:
            seg_idx.append(i)
        else:
            glob_by.setdefault(r["conv_type"].split("_")[0], []).append(i)
    glob_idx = []
    while len(glob_idx) < args.n_global and any(glob_by.values()):
        for k in list(glob_by):
            if glob_by[k] and len(glob_idx) < args.n_global:
                glob_idx.append(glob_by[k].pop(0))
    picks = seg_idx + glob_idx
    print(f"eval on {len(picks)} examples ({len(seg_idx)} localized)", flush=True)

    model, proc, edit_ids = load_thinker(lora_dir=str(PROJECT_ROOT / args.qwen))
    model.eval()
    if args.arch == "fusion":
        from edit_agent.musicgen_fusion import build_fusion_bridge
        bridge = build_fusion_bridge(torch.device(DEVICE))
    elif args.arch == "kv":
        from edit_agent.musicgen_kv import build_kv_bridge
        bridge = build_kv_bridge(torch.device(DEVICE))
    else:
        bridge = build_musicgen_bridge(torch.device(DEVICE))
    bridge.load_adapter(PROJECT_ROOT / args.mg, DEVICE)
    bridge.decoder.to(DEVICE, dtype=torch.bfloat16)
    bridge.eval()
    from muq import MuQMuLan
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large",
                                     cache_dir=str(PROJECT_ROOT / "weights")).to(DEVICE).eval()

    import soundfile as sf
    rows_out = []
    for k, i in enumerate(picks):
        r = ds.rows[i]
        ex = ds[i]
        enc = ex["enc"]
        try:
            h = live_hidden(model, proc, edit_ids, ex)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {r['example_id']}: {exc}", flush=True)
            continue
        src_codes = enc["src"].long().unsqueeze(0).to(DEVICE)
        gen_kwargs = {}
        if args.arch in ("kv", "fusion") and enc.get("seg_frames"):
            gen_kwargs["segment"] = tuple(enc["seg_frames"])  # anchored localized edit
        codes = bridge.generate(h.float().unsqueeze(0), src_codes,
                                max_frames=500, guidance=args.guidance, seed=500 + k,
                                **gen_kwargs)
        wav = bridge.decode_audio(codes).squeeze().float().cpu().numpy()
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * 0.95

        import librosa
        gen24 = librosa.resample(wav, orig_sr=32000, target_sr=24000)
        src24 = crop(r["src_path"], enc["win"])
        tgt24 = crop(r["tgt_path"], enc["win"])
        e_g, e_s, e_t = (mulan_of(mulan, y) for y in (gen24, src24, tgt24))
        row = {"example_id": r["example_id"], "conv_type": r["conv_type"],
               "cos_gen_tgt": float((e_g * e_t).sum()),
               "cos_src_tgt": float((e_s * e_t).sum()),
               "cos_gen_src": float((e_g * e_s).sum())}

        if k < args.n_ablate:
            codes0 = bridge.generate(torch.zeros_like(h).unsqueeze(0), src_codes,
                                     max_frames=500, guidance=args.guidance, seed=500 + k)
            t = min(codes.shape[-1], codes0.shape[-1])
            row["ablate_code_diff"] = float(
                (codes[..., :t] != codes0[..., :t]).float().mean())

        if k < args.n_triplets:
            trip = out_dir / "triplets"
            sf.write(trip / f"{k:02d}_{r['conv_type']}_BEFORE.wav", src24, 24000)
            sf.write(trip / f"{k:02d}_{r['conv_type']}_TARGET.wav", tgt24, 24000)
            sf.write(trip / f"{k:02d}_{r['conv_type']}_GEN.wav", wav, 32000)

        rows_out.append(row)
        print(f"[{k + 1}/{len(picks)}] {r['conv_type']} "
              f"cos(gen,tgt)={row['cos_gen_tgt']:.3f} cos(src,tgt)={row['cos_src_tgt']:.3f}"
              + (f" abl_diff={row.get('ablate_code_diff'):.3f}"
                 if "ablate_code_diff" in row else ""), flush=True)

    def avg(key):
        v = [r[key] for r in rows_out if key in r and np.isfinite(r[key])]
        return float(np.mean(v)) if v else None

    summary = {"n": len(rows_out), "cos_gen_tgt": avg("cos_gen_tgt"),
               "cos_src_tgt": avg("cos_src_tgt"), "cos_gen_src": avg("cos_gen_src"),
               "ablate_code_diff": avg("ablate_code_diff")}
    json.dump({"summary": summary, "rows": rows_out},
              open(out_dir / "eval.json", "w"), indent=1)
    print("=== summary ===")
    for k2, v in summary.items():
        print(f"{k2}: {v}")


if __name__ == "__main__":
    main()
