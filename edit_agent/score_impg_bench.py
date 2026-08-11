"""Score the IMPG benchmark outputs: FAD, CLAP, KL, SSIM, P-Demucs, SI-SDR(i).

Usage: conda run -n llama python -u src/edit_agent/score_impg_bench.py
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
torch.backends.cudnn.enabled = False
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

BENCH = PROJECT_ROOT / "results/impg_bench"
TASKS = ["add", "remove", "extract"]
DEV = "cuda:0"
SR = 32000


def pairs(task):
    out = []
    for p in sorted((BENCH / task / "output").glob("*.wav")):
        gt = BENCH / task / "ground_truth" / p.name
        inp = BENCH / task / "input" / p.name
        txt = (BENCH / task / "instruction" / f"{p.stem}.txt").read_text().splitlines()[0]
        if gt.exists():
            out.append((p, gt, inp, txt))
    return out


def load(p, sr=SR):
    y, _ = librosa.load(p, sr=sr, mono=True)
    return y


scores: dict[str, dict] = {t: {} for t in TASKS}

# ── FAD ─────────────────────────────────────────────────────────
from evaluation.fad import compute_fad  # noqa: E402
for t in TASKS:
    scores[t]["fad"] = compute_fad(str(BENCH / t / "output"),
                                   str(BENCH / t / "ground_truth"), verbose=False)
    print(f"{t}: FAD {scores[t]['fad']:.3f}", flush=True)

# ── CLAP (audio-text vs instruction) ───────────────────────────
import laion_clap  # noqa: E402
clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
clap.load_ckpt()
with torch.no_grad():
    for t in TASKS:
        vals = []
        for p, gt, inp, txt in pairs(t):
            g = load(p, 48000)
            ea = clap.get_audio_embedding_from_data(x=g[None, :], use_tensor=False)[0]
            et = clap.get_text_embedding([txt, txt], use_tensor=False)[0]
            ea /= np.linalg.norm(ea); et /= np.linalg.norm(et)
            vals.append(float(ea @ et))
        scores[t]["clap"] = float(np.mean(vals))
        print(f"{t}: CLAP {scores[t]['clap']:.3f}", flush=True)
del clap
torch.cuda.empty_cache()

# ── KL (PANNs CNN14) ───────────────────────────────────────────
from panns_inference import AudioTagging  # noqa: E402
at = AudioTagging(checkpoint_path=None, device=DEV)
with torch.no_grad():
    for t in TASKS:
        vals = []
        for p, gt, inp, txt in pairs(t):
            pg = at.inference(load(p)[None, :])[0][0]
            pt = at.inference(load(gt)[None, :])[0][0]
            pg = np.clip(pg, 1e-8, 1); pt = np.clip(pt, 1e-8, 1)
            pg /= pg.sum(); pt /= pt.sum()
            vals.append(float(np.sum(pt * np.log(pt / pg))))
        scores[t]["kl"] = float(np.mean(vals))
        print(f"{t}: KL {scores[t]['kl']:.3f}", flush=True)
del at
torch.cuda.empty_cache()

# ── SSIM (log-mel) ─────────────────────────────────────────────
from skimage.metrics import structural_similarity  # noqa: E402
for t in TASKS:
    vals = []
    for p, gt, inp, txt in pairs(t):
        g, r = load(p), load(gt)
        L = min(len(g), len(r)); g, r = g[:L], r[:L]
        mg = librosa.power_to_db(librosa.feature.melspectrogram(y=g, sr=SR, n_mels=128))
        mr = librosa.power_to_db(librosa.feature.melspectrogram(y=r, sr=SR, n_mels=128))
        dr = max(mg.max() - mg.min(), mr.max() - mr.min(), 1e-6)
        vals.append(structural_similarity(mg, mr, data_range=dr))
    scores[t]["ssim"] = float(np.mean(vals))
    print(f"{t}: SSIM {scores[t]['ssim']:.3f}", flush=True)

# ── P-Demucs (stem-direction agreement) ────────────────────────
from demucs.pretrained import get_model  # noqa: E402
from demucs.apply import apply_model  # noqa: E402
dm = get_model("htdemucs").to(DEV).eval()
DSR = dm.samplerate
TH = 1.5


def stem_db(y):
    x = torch.from_numpy(y).float().to(DEV)
    x = x.unsqueeze(0).repeat(2, 1).unsqueeze(0)
    ref = x.mean(0)
    x = (x - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(dm, x, device=DEV)[0]
    e = out.pow(2).mean(dim=(1, 2)).cpu().numpy()
    return 10 * np.log10(e + 1e-10)


for t in TASKS:
    hits = tot = 0
    for p, gt, inp, txt in pairs(t):
        s = load(inp, DSR); g = load(p, DSR); r = load(gt, DSR)
        L = min(len(s), len(g), len(r)); s, g, r = s[:L], g[:L], r[:L]
        dg, dr_ = stem_db(g) - stem_db(s), stem_db(r) - stem_db(s)
        for k in range(len(dr_)):
            a = 0 if abs(dr_[k]) < TH else np.sign(dr_[k])
            b = 0 if abs(dg[k]) < TH else np.sign(dg[k])
            tot += 1; hits += int(a == b)
    scores[t]["p_demucs"] = hits / max(tot, 1)
    print(f"{t}: P-Demucs {scores[t]['p_demucs']:.3f}", flush=True)
del dm
torch.cuda.empty_cache()

# ── SI-SDR / SI-SDRi (remove, extract) ─────────────────────────
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as sisdr  # noqa: E402,E501
for t in ["remove", "extract"]:
    v, vi = [], []
    for p, gt, inp, txt in pairs(t):
        g, r, s = load(p), load(gt), load(inp)
        L = min(len(g), len(r), len(s))
        g, r, s = (torch.from_numpy(x[:L]) for x in (g, r, s))
        sd = float(sisdr(g, r))
        v.append(sd)
        vi.append(sd - float(sisdr(s, r)))
    scores[t]["si_sdr"] = float(np.mean(v))
    scores[t]["si_sdri"] = float(np.mean(vi))
    print(f"{t}: SI-SDR {scores[t]['si_sdr']:.2f} SI-SDRi {scores[t]['si_sdri']:.2f}",
          flush=True)

json.dump(scores, open(BENCH / "scores.json", "w"), indent=1)
print(json.dumps(scores, indent=1), flush=True)
