"""Sampling eval for the trained bridge adapter.

On val examples: sample_edit from cached src latents + [EDIT] hidden states, decode,
then measure
- MuLan-cos(gen, tgt)  vs the src->tgt baseline cos (did we move toward the target?)
- outside-segment waveform corr vs src for localized (inp_*) examples (preservation)
- conditioning ablation: same seed with zeroed hidden states -> latent L2 distance

Writes summary JSON + listening triplets to results/bridge_eval/.

Usage:
  conda run --no-capture-output -n llama python -u src/edit_agent/eval_bridge.py \
      [--adapter ckpts/edit_agent/bridge/final] [--n-global 40] [--n-seg 20]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "DiffRhythm"))

torch.backends.cudnn.enabled = False
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from edit_agent.bridge import build_bridge, load_adapter  # noqa: E402
from edit_agent.bridge_data import BridgeDataset, FPS  # noqa: E402

CACHE = PROJECT_ROOT / "data/edit_dataset/bridge_cache"
SR = 44100
DEVICE = "cuda"


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def load_audio(path, sr=SR):
    import librosa
    y, _ = librosa.load(PROJECT_ROOT / path, sr=sr, mono=True)
    return y


@torch.no_grad()
def mulan_embed(mulan, y_44k: np.ndarray) -> torch.Tensor:
    import librosa
    y = librosa.resample(y_44k, orig_sr=SR, target_sr=24000)
    dur = len(y) / 24000
    off = int(max(dur / 2 - 5, 0) * 24000)
    y = y[off:off + 10 * 24000]
    emb = mulan(wavs=torch.from_numpy(y).unsqueeze(0).to(DEVICE))[0]
    return torch.nn.functional.normalize(emb.float(), dim=-1)


def cos(a, b):
    return float((a * b).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="ckpts/edit_agent/bridge/final")
    ap.add_argument("--n-global", type=int, default=40)
    ap.add_argument("--n-seg", type=int, default=20)
    ap.add_argument("--n-ablate", type=int, default=10)
    ap.add_argument("--n-triplets", type=int, default=10)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--out", default="results/bridge_eval")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / args.out
    (out_dir / "triplets").mkdir(parents=True, exist_ok=True)

    ds = BridgeDataset("val")
    seg_rows = [i for i, r in enumerate(ds.rows) if r.get("segment")][:args.n_seg]
    # spread global rows across conv types
    by_type = {}
    for i, r in enumerate(ds.rows):
        if r.get("segment"):
            continue
        by_type.setdefault(r["conv_type"].split("_")[0], []).append(i)
    glob_rows = []
    while len(glob_rows) < args.n_global and any(by_type.values()):
        for k in list(by_type):
            if by_type[k] and len(glob_rows) < args.n_global:
                glob_rows.append(by_type[k].pop(0))
    picks = seg_rows + glob_rows

    print(f"eval on {len(picks)} val examples ({len(seg_rows)} localized)", flush=True)

    model = build_bridge(torch.device(DEVICE))
    load_adapter(model, PROJECT_ROOT / args.adapter, DEVICE)
    model.projector.eval(); model.cross_attn.eval()

    from huggingface_hub import hf_hub_download
    vae = torch.jit.load(hf_hub_download("ASLP-lab/DiffRhythm-vae", "vae_model.pt",
                                         cache_dir=str(PROJECT_ROOT / "weights")),
                         map_location=DEVICE)
    from muq import MuQMuLan
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large",
                                     cache_dir=str(PROJECT_ROOT / "weights")).to(DEVICE).eval()
    from src.DiffRhythm.infer.infer_utils import decode_audio
    neg = np.load(PROJECT_ROOT / "src/DiffRhythm/infer/example/vocal.npy")
    NEG_STYLE = torch.from_numpy(neg).unsqueeze(0).to(DEVICE).half() \
        if torch.from_numpy(neg).dim() == 1 else torch.from_numpy(neg).to(DEVICE).half()

    import soundfile as sf
    rows_out = []
    for k, i in enumerate(picks):
        r = ds.rows[i]
        x = ds[i]
        t = x["n_frames"]
        src = x["src"].unsqueeze(0).to(DEVICE).half()
        h = x["h"].unsqueeze(0).to(DEVICE)
        seg = x["segment"]
        with torch.no_grad():
            gen_lat = model.sample(src, h, NEG_STYLE, t, segment=seg,
                                   steps=args.steps, cfg_strength=args.cfg, seed=1000 + k)
            audio = decode_audio(gen_lat.float().permute(0, 2, 1), vae, chunked=True)
        gen = audio.squeeze().float().cpu().numpy()
        if gen.ndim > 1:
            gen = gen.mean(axis=0)
        peak = np.abs(gen).max()
        if peak > 0:
            gen = gen / peak * 0.95

        src_y = load_audio(r["src_path"])
        tgt_y = load_audio(r["tgt_path"])
        e_gen = mulan_embed(mulan, gen)
        e_src = mulan_embed(mulan, src_y)
        e_tgt = mulan_embed(mulan, tgt_y)
        row = {"example_id": r["example_id"], "conv_type": r["conv_type"],
               "cos_gen_tgt": cos(e_gen, e_tgt), "cos_src_tgt": cos(e_src, e_tgt),
               "cos_gen_src": cos(e_gen, e_src)}

        if seg is not None:
            n = min(len(gen), len(src_y))
            m = np.ones(n, dtype=bool)
            s0, s1 = int(seg[0] / FPS * SR), int(seg[1] / FPS * SR)
            m[s0:min(s1, n)] = False
            if m.sum() > SR:
                a, b = gen[:n][m], src_y[:n][m]
                row["outside_corr"] = float(np.corrcoef(a, b)[0, 1])

        if k < args.n_ablate:
            with torch.no_grad():
                gen0 = model.sample(src, torch.zeros_like(h), NEG_STYLE, t, segment=seg,
                                    steps=args.steps, cfg_strength=args.cfg, seed=1000 + k)
            row["ablate_latent_l2"] = float((gen_lat - gen0).float().pow(2).mean().sqrt())

        if k < args.n_triplets:
            trip = out_dir / "triplets"
            sf.write(trip / f"{k:02d}_src.wav", src_y, SR)
            sf.write(trip / f"{k:02d}_tgt.wav", tgt_y, SR)
            sf.write(trip / f"{k:02d}_gen.wav", gen, SR)

        rows_out.append(row)
        print(f"[{k + 1}/{len(picks)}] {r['conv_type']} "
              f"cos(gen,tgt)={row['cos_gen_tgt']:.3f} cos(src,tgt)={row['cos_src_tgt']:.3f}"
              + (f" out_corr={row.get('outside_corr'):.3f}" if "outside_corr" in row else "")
              + (f" abl_l2={row['ablate_latent_l2']:.3f}" if "ablate_latent_l2" in row else ""),
              flush=True)

    def avg(key, rows):
        v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(v)) if v else None

    summary = {
        "n": len(rows_out),
        "cos_gen_tgt": avg("cos_gen_tgt", rows_out),
        "cos_src_tgt": avg("cos_src_tgt", rows_out),
        "cos_gen_src": avg("cos_gen_src", rows_out),
        "outside_corr": avg("outside_corr", rows_out),
        "ablate_latent_l2": avg("ablate_latent_l2", rows_out),
    }
    json.dump({"summary": summary, "rows": rows_out},
              open(out_dir / "eval.json", "w"), indent=1)
    print("=== summary ===")
    for k2, v in summary.items():
        print(f"{k2}: {v}")


if __name__ == "__main__":
    main()
