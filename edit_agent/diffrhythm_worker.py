"""DiffRhythm render worker for the web demo (llama env) — replaces the MelodyFlow one.

Same HTTP contract as melodyflow_worker (:7861 by default):
  GET  /health
  POST /edit {"wav_path": ..., "prompt": ..., "segment": [start_s, end_s] | null}

Rendering (text-conditioned baseline until the bridge lands):
- style = normalized blend of MuLan(source audio) and MuLan(edit text) so the output
  keeps the source character while moving toward the requested change.
- cond = VAE latent of the source chunk; when `segment` is given, only that span is
  regenerated (DiffRhythm native edit mode -> outside preserved EXACTLY); otherwise
  the full chunk is regenerated.
"""

import os
import sys
import threading
import time
from pathlib import Path

import torch

torch.backends.cudnn.enabled = False
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "DiffRhythm"))

import torchaudio  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import numpy as np  # noqa: E402

from infer.infer_utils import (  # noqa: E402
    decode_audio, encode_audio, normalize_audio, prepare_audio, vae_sample,
)


def load_negative_style(device: str) -> torch.Tensor:
    # infer_utils.get_negative_style_prompt uses a CWD-relative path; load absolutely
    arr = np.load(PROJECT_ROOT / "src/DiffRhythm/infer/example/vocal.npy")
    return torch.from_numpy(arr).to(device).half()
from src.image_cond.lora_dit import load_base_cfm  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "webapp_edits_diffrhythm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"
SAMPLE_RATE = 44100
DOWNSAMPLE = 2048
FPS = SAMPLE_RATE / DOWNSAMPLE          # ~21.53 latent frames / s
MAX_FRAMES = 2048
DIT_CONFIG = str(PROJECT_ROOT / "src/DiffRhythm/config/diffrhythm-1b.json")
STEPS, CFG = 32, 4.0

print("loading DiffRhythm CFM + VAE + MuLan ...", flush=True)
cfm = load_base_cfm(DIT_CONFIG, torch.device(DEVICE), MAX_FRAMES, str(PROJECT_ROOT / "weights"))
cfm.half().eval()
from huggingface_hub import hf_hub_download  # noqa: E402
vae = torch.jit.load(hf_hub_download("ASLP-lab/DiffRhythm-vae", "vae_model.pt",
                                     cache_dir=str(PROJECT_ROOT / "weights")),
                     map_location="cpu").to(DEVICE)
from muq import MuQMuLan  # noqa: E402
mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large",
                                 cache_dir=str(PROJECT_ROOT / "weights")).to(DEVICE).eval()
NEG_STYLE = load_negative_style(DEVICE)

# optional Stage-A bridge: renders straight from [EDIT] hidden states
BRIDGE = None
BRIDGE_DIR = os.environ.get("BRIDGE_ADAPTER")
if BRIDGE_DIR:
    print(f"loading bridge adapter from {BRIDGE_DIR} ...", flush=True)
    from edit_agent.bridge import build_bridge, load_adapter  # noqa: E402
    BRIDGE = build_bridge(torch.device(DEVICE))  # separate CFM copy: keeps /edit unpatched
    load_adapter(BRIDGE, PROJECT_ROOT / BRIDGE_DIR, DEVICE)
    BRIDGE.projector.eval(); BRIDGE.cross_attn.eval()
print("worker ready", flush=True)

app = FastAPI()
gpu_lock = threading.Lock()


class EditRequest(BaseModel):
    wav_path: str
    prompt: str
    segment: list[float] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": "DiffRhythm-1.2 (text-conditioned edit baseline)",
            "bridge": BRIDGE is not None}


def load_wav_any(path: Path) -> tuple[torch.Tensor, int]:
    """torchaudio 2.10 needs torchcodec for decoding; use librosa instead."""
    import librosa
    import numpy as np
    y, sr = librosa.load(path, sr=44100, mono=False)  # resample here: torchaudio 2.10 lacks functional.Resample
    y = np.atleast_2d(y)
    return torch.from_numpy(y.astype("float32")), int(sr)


@torch.no_grad()
def encode_chunk(wav_path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = load_wav_any(wav_path)
    n_frames = min(int(wav.shape[-1] / sr * FPS), MAX_FRAMES)
    audio = prepare_audio(wav, in_sr=sr, target_sr=SAMPLE_RATE,
                          target_length=int(n_frames * DOWNSAMPLE),
                          target_channels=2, device=torch.device(DEVICE))
    audio = normalize_audio(audio, -6)
    latents = encode_audio(audio.float(), vae, chunked=True)
    mean, scale = latents.chunk(2, dim=1)
    z, _kl = vae_sample(mean, scale)
    latent = z.transpose(1, 2)  # [1, T, 64]
    return latent.half(), n_frames


@torch.no_grad()
def style_for(wav_path: Path, prompt: str) -> torch.Tensor:
    import librosa
    dur = librosa.get_duration(path=str(wav_path))
    off = max(dur / 2 - 5, 0)
    y, _ = librosa.load(wav_path, sr=24000, mono=True, offset=off, duration=min(10, dur))
    audio_emb = mulan(wavs=torch.from_numpy(y).unsqueeze(0).to(DEVICE))
    text_emb = mulan(texts=prompt)
    style = 0.5 * torch.nn.functional.normalize(audio_emb, dim=-1) \
        + 0.5 * torch.nn.functional.normalize(text_emb, dim=-1)
    return torch.nn.functional.normalize(style, dim=-1).half()


@app.post("/edit")
def edit(req: EditRequest):
    wav_path = Path(req.wav_path)
    if not wav_path.is_absolute():
        wav_path = PROJECT_ROOT / wav_path
    if not wav_path.exists():
        raise HTTPException(404, f"wav not found: {wav_path}")
    t0 = time.time()
    with gpu_lock, torch.no_grad():
        latent, n_frames = encode_chunk(wav_path)
        cond = torch.zeros(1, MAX_FRAMES, 64, device=DEVICE, dtype=torch.half)
        cond[:, :n_frames] = latent
        if req.segment:
            s = max(0, int(req.segment[0] * FPS))
            e = min(n_frames, int(req.segment[1] * FPS))
            segs = [(s, max(e, s + 8))]
        else:
            segs = [(0, n_frames)]
        style = style_for(wav_path, req.prompt)
        lrc = torch.zeros(1, MAX_FRAMES, dtype=torch.long, device=DEVICE)
        start_time = torch.zeros(1, device=DEVICE, dtype=torch.half)
        norm_dur = torch.tensor([n_frames / MAX_FRAMES], device=DEVICE, dtype=torch.half)
        out, _ = cfm.sample(cond=cond, text=lrc, duration=MAX_FRAMES,
                            style_prompt=style, negative_style_prompt=NEG_STYLE,
                            steps=STEPS, cfg_strength=CFG, start_time=start_time,
                            latent_pred_segments=segs, song_duration=norm_dur)
        gen = out[0][:, :n_frames, :].float().permute(0, 2, 1)
        audio = decode_audio(gen, vae, chunked=True).squeeze(0).float().cpu()
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak * 0.95
        out_path = OUT_DIR / f"{wav_path.stem}_{int(time.time())}.wav"
        import soundfile as sf
        sf.write(out_path, (audio.clamp(-1, 1) * 32767).numpy().astype("int16").T,
                 SAMPLE_RATE, subtype="PCM_16")
    return {"edited_path": str(out_path), "gen_time_s": round(time.time() - t0, 1),
            "segment_frames": segs[0]}


class BridgeEditRequest(BaseModel):
    wav_path: str
    hidden_b64: str                      # base64 of fp16 [9, 3584] bytes
    segment: list[float] | None = None


@app.post("/edit_bridge")
def edit_bridge(req: BridgeEditRequest):
    if BRIDGE is None:
        raise HTTPException(400, "bridge adapter not loaded (set BRIDGE_ADAPTER)")
    wav_path = Path(req.wav_path)
    if not wav_path.is_absolute():
        wav_path = PROJECT_ROOT / wav_path
    if not wav_path.exists():
        raise HTTPException(404, f"wav not found: {wav_path}")
    import base64
    raw = base64.b64decode(req.hidden_b64)
    h = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(1, 9, 3584)
    t0 = time.time()
    with gpu_lock, torch.no_grad():
        latent, n_frames = encode_chunk(wav_path)
        seg = None
        if req.segment:
            s = max(0, int(req.segment[0] * FPS))
            e = min(n_frames, int(req.segment[1] * FPS))
            seg = (s, max(e, s + 8))
        gen_lat = BRIDGE.sample(latent.to(DEVICE), h.to(DEVICE), NEG_STYLE, n_frames,
                                segment=seg, steps=STEPS, cfg_strength=2.0)
        audio = decode_audio(gen_lat.float().permute(0, 2, 1), vae,
                             chunked=True).squeeze(0).float().cpu()
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak * 0.95
        out_path = OUT_DIR / f"{wav_path.stem}_bridge_{int(time.time())}.wav"
        import soundfile as sf
        sf.write(out_path, (audio.clamp(-1, 1) * 32767).numpy().astype("int16").T,
                 SAMPLE_RATE, subtype="PCM_16")
    return {"edited_path": str(out_path), "gen_time_s": round(time.time() - t0, 1),
            "segment_frames": list(seg) if seg else [0, n_frames]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1",
                port=int(os.environ.get("DIFFRHYTHM_PORT", "7861")), log_level="warning")
