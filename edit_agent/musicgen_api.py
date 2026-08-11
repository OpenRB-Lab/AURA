"""HTTP API serving the production edit model (Qwen thinker + fusion bridge).

Single process, one GPU (set CUDA_VISIBLE_DEVICES), FastAPI on API_PORT
(default 9004). Audio moves as filesystem paths (same-host clients), pattern
inherited from diffrhythm_worker.py. Output audio is 32 kHz (MusicGen/EnCodec).

  QWEN_ADAPTER  default ckpts/edit_agent/joint_fusion_r64/final/qwen
  MG_ADAPTER    default ckpts/edit_agent/joint_fusion_r64/final/musicgen

Run: src/scripts/serve_musicgen_api.sh
"""

import base64
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.qwen_wrapper import (  # noqa: E402  (also fixes cudnn flags)
    generate_with_edit_capture, load_audio_16k, load_thinker)
from edit_agent.musicgen_fusion import build_fusion_bridge  # noqa: E402
from edit_agent.sft_data import SYSTEM_PROMPT  # noqa: E402

QWEN_ADAPTER = os.environ.get(
    "QWEN_ADAPTER", "ckpts/edit_agent/joint_fusion_r64/final/qwen")
MG_ADAPTER = os.environ.get(
    "MG_ADAPTER", "ckpts/edit_agent/joint_fusion_r64/final/musicgen")
OUT_DIR = PROJECT_ROOT / "results/api_edits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 32000
WIN_S = 10.0
FRAME_RATE = 50
N_FRAMES = 500
IM_END = 151645  # thinker has no eos_token_id — always pass explicitly
EDIT_TOKEN_RE = re.compile(r"\[EDIT_[A-Z0-9_]+\]")

print(f"[api] loading thinker from {QWEN_ADAPTER} (merged)", flush=True)
MODEL, PROC, EDIT_IDS = load_thinker(
    lora_dir=str(PROJECT_ROOT / QWEN_ADAPTER), merge_lora=True)
MODEL.eval()
print(f"[api] loading fusion bridge from {MG_ADAPTER}", flush=True)
BRIDGE = build_fusion_bridge(torch.device("cuda"))
BRIDGE.load_adapter(PROJECT_ROOT / MG_ADAPTER, "cuda")
BRIDGE.decoder.to("cuda", dtype=torch.bfloat16)
BRIDGE.eval()
gpu_lock = threading.Lock()
print("[api] models ready", flush=True)

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="music-edit-api")


def _resolve(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise HTTPException(404, f"audio file not found: {p}")
    return path


UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="api_uploads_"))


def _audio_input(audio_b64: str | None, wav_path: str | None) -> Path:
    """Canonical input: base64-encoded audio bytes (any librosa-readable
    format). Legacy alternative: a same-host filesystem path."""
    if audio_b64:
        try:
            raw = base64.b64decode(audio_b64, validate=True)
        except Exception:  # noqa: BLE001
            raise HTTPException(422, "audio: invalid base64")
        p = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}.audio"
        p.write_bytes(raw)
        return p
    if wav_path:
        return _resolve(wav_path)
    raise HTTPException(422, "provide 'audio' (base64) or 'wav_path'")


def _wav_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _thinker_reply(wav_path: Path, text: str, history: list | None):
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    turns = list(history or []) + [["user", text]]
    for i, (role, t) in enumerate(turns):
        if i == 0 and role == "user":
            content = [{"type": "audio", "audio": str(wav_path)},
                       {"type": "text", "text": t}]
        else:
            content = [{"type": "text", "text": t}]
        msgs.append({"role": role, "content": content})
    tmpl = PROC.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = PROC(text=[tmpl], audio=[load_audio_16k(wav_path)],
                  return_tensors="pt", padding=True).to("cuda")
    reply, h = generate_with_edit_capture(MODEL, PROC, inputs, EDIT_IDS,
                                          max_new_tokens=256, eos_token_id=IM_END)
    return EDIT_TOKEN_RE.sub("", reply).strip(), h


def _load_window(wav_path: Path, segment):
    import librosa
    dur = librosa.get_duration(path=str(wav_path))
    if segment:
        mid = (segment[0] + segment[1]) / 2
        start = min(max(mid - WIN_S / 2, 0), max(dur - WIN_S, 0))
    else:
        start = max((dur - WIN_S) / 2, 0)
    y, _ = librosa.load(wav_path, sr=SR, mono=True, offset=start, duration=WIN_S)
    if len(y) < int(WIN_S * SR):
        y = np.pad(y, (0, int(WIN_S * SR) - len(y)))
    seg_frames = None
    if segment:
        s = max(0, int((segment[0] - start) * FRAME_RATE))
        e = min(N_FRAMES, int((segment[1] - start) * FRAME_RATE))
        if e > s:
            seg_frames = (s, e)
    return y.astype("float32"), start, seg_frames


class ChatRequest(BaseModel):
    text: str                     # the user message
    audio: str | None = None      # base64-encoded audio bytes (canonical)
    wav_path: str | None = None   # legacy: same-host file path
    history: list | None = None


class EditRequest(BaseModel):
    text: str | None = None       # the edit instruction
    prompt: str | None = None     # alias accepted from the gradio webapp
    audio: str | None = None      # base64-encoded audio bytes (canonical)
    wav_path: str | None = None   # legacy: same-host file path
    segment: list | None = None
    guidance: float = 2.0
    seed: int | None = None
    history: list | None = None


@app.get("/health")
def health():
    return {"status": "ok", "qwen_adapter": QWEN_ADAPTER,
            "mg_adapter": MG_ADAPTER,
            "device": torch.cuda.get_device_name(0)}


@app.post("/chat")
def chat(req: ChatRequest):
    wav = _audio_input(req.audio, req.wav_path)
    with gpu_lock:
        reply, _ = _thinker_reply(wav, req.text, req.history)
    return {"text": reply}


@app.post("/edit")
def edit(req: EditRequest):
    instruction = req.text or req.prompt
    if not instruction:
        raise HTTPException(422, "provide 'text' (or legacy 'prompt')")
    wav = _audio_input(req.audio, req.wav_path)
    t0 = time.time()
    with gpu_lock, torch.no_grad():
        reply, h = _thinker_reply(wav, instruction, req.history)
        if h is None:
            raise HTTPException(
                400, f"model did not emit an edit block (reply: {reply!r})")
        y, win0, seg_frames = _load_window(wav, req.segment)
        src_wav = torch.from_numpy(y).to("cuda").view(1, 1, -1)
        codes = BRIDGE.audio_encoder.encode(src_wav, bandwidth=None).audio_codes
        src = (codes[0] if codes.dim() == 4 else codes)[0, :, :N_FRAMES]
        src = src.long().unsqueeze(0)
        gen = BRIDGE.generate(h.float().unsqueeze(0).to("cuda"), src,
                              max_frames=N_FRAMES, guidance=req.guidance,
                              seed=req.seed, segment=seg_frames)
        if seg_frames:
            out = BRIDGE.decode_audio_smooth(gen, src, seg_frames)
        else:
            out = BRIDGE.decode_audio(gen)
        wav_out = out.squeeze().float().cpu().numpy()
        peak = np.abs(wav_out).max()
        if peak > 0:
            wav_out = wav_out / peak * 0.95
    import soundfile as sf
    out_path = OUT_DIR / f"{uuid.uuid4().hex[:12]}.wav"
    sf.write(out_path, wav_out, SR, subtype="PCM_16")
    return {"text": reply,
            "audio": _wav_b64(out_path),  # base64 wav (32 kHz PCM_16)
            "edited_path": str(out_path.relative_to(PROJECT_ROOT)),
            "gen_time_s": round(time.time() - t0, 1),
            "window": [round(win0, 2), round(win0 + WIN_S, 2)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",
                port=int(os.environ.get("API_PORT", "9004")),
                log_level="warning")
