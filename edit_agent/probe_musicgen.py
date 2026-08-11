"""20-clip FAD+cos probe for MusicGen K/V checkpoint selection (NOT CE).

Uses CACHED v5 hidden states (what the bridge trains on) — no Qwen load, fast.
Fixed 20 val examples (10 localized/anchored + 10 global), fixed seeds, guidance 2.

Usage:
  conda run -n llama python -u src/edit_agent/probe_musicgen.py \
      --adapter ckpts/edit_agent/musicgen_kv_r64/step_10000 [--device cuda:0]
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

from edit_agent.musicgen_data import MusicGenBridgeDataset  # noqa: E402

CACHE = PROJECT_ROOT / "data/edit_dataset/bridge_cache"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--arch", choices=["kv", "fusion"], default="kv")
    ap.add_argument("--filter", default=None,
                    help="restrict probe clips to example ids containing this")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-seg", type=int, default=10)
    ap.add_argument("--n-glob", type=int, default=10)
    ap.add_argument("--out", default=None, help="scratch dir; default per-adapter")
    args = ap.parse_args()
    dev = args.device

    ds = MusicGenBridgeDataset(args.split, id_filter=args.filter)
    seg_idx, glob_idx = [], []
    for i, r in enumerate(ds.rows):
        enc_meta = torch.load(CACHE / "encodec" / f"{r['example_id']}.pt",
                              weights_only=True)
        if enc_meta.get("seg_frames") and len(seg_idx) < args.n_seg:
            seg_idx.append(i)
        elif not enc_meta.get("seg_frames") and len(glob_idx) < args.n_glob:
            glob_idx.append(i)
        if len(seg_idx) == args.n_seg and len(glob_idx) == args.n_glob:
            break
    picks = seg_idx + glob_idx
    print(f"picks: {len(seg_idx)} localized + {len(glob_idx)} global "
          f"from split={args.split}", flush=True)

    if args.arch == "fusion":
        from edit_agent.musicgen_fusion import build_fusion_bridge
        bridge = build_fusion_bridge(torch.device(dev))
    else:
        from edit_agent.musicgen_kv import build_kv_bridge
        bridge = build_kv_bridge(torch.device(dev))
    bridge.load_adapter(PROJECT_ROOT / args.adapter, dev)
    bridge.decoder.to(dev, dtype=torch.bfloat16)
    bridge.eval()
    from muq import MuQMuLan
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large",
                                     cache_dir=str(PROJECT_ROOT / "weights")).to(dev).eval()

    import librosa
    import soundfile as sf
    out_dir = Path(args.out) if args.out else \
        PROJECT_ROOT / "results/probe" / Path(args.adapter).name
    (out_dir / "GEN").mkdir(parents=True, exist_ok=True)
    (out_dir / "TGT").mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def emb(y, sr):
        y24 = librosa.resample(y, orig_sr=sr, target_sr=24000) if sr != 24000 else y
        e = mulan(wavs=torch.from_numpy(y24.astype("float32")).unsqueeze(0).to(dev))[0]
        return torch.nn.functional.normalize(e.float(), dim=-1)

    cos_gt = []
    with torch.no_grad():
        for k, i in enumerate(picks):
            r = ds.rows[i]
            enc = torch.load(CACHE / "encodec" / f"{r['example_id']}.pt",
                             weights_only=True)
            h = torch.load(CACHE / "hidden" / f"{r['example_id']}.pt",
                           weights_only=True)["h"].float().unsqueeze(0).to(dev)
            src = enc["src"].long().unsqueeze(0).to(dev)
            seg = tuple(enc["seg_frames"]) if enc.get("seg_frames") else None
            codes = bridge.generate(h, src, max_frames=500, guidance=args.guidance,
                                    seed=9000 + k, segment=seg)
            wav = bridge.decode_audio(codes).squeeze().float().cpu().numpy()
            p = np.abs(wav).max()
            wav = wav / p * 0.95 if p > 0 else wav
            tgt, _ = librosa.load(PROJECT_ROOT / r["tgt_path"], sr=24000, mono=True,
                                  offset=enc["win"][0],
                                  duration=enc["win"][1] - enc["win"][0])
            cos_gt.append(float((emb(wav, 32000) * emb(tgt, 24000)).sum()))
            sf.write(out_dir / "GEN" / f"{k:02d}.wav", wav, 32000)
            sf.write(out_dir / "TGT" / f"{k:02d}.wav", tgt, 24000)
            print(f"[{k + 1}/{len(picks)}] cos {cos_gt[-1]:.3f}", flush=True)

    from evaluation.fad import compute_fad
    fad = compute_fad(str(out_dir / "GEN"), str(out_dir / "TGT"), verbose=False)
    result = {"adapter": args.adapter, "fad": fad,
              "cos_gen_tgt": float(np.mean(cos_gt))}
    json.dump(result, open(out_dir / "probe.json", "w"), indent=1)
    print(f"PROBE {args.adapter}: fad {fad:.3f} cos {result['cos_gen_tgt']:.3f}")


if __name__ == "__main__":
    main()
