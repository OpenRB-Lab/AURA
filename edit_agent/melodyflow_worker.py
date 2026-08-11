"""MelodyFlow render worker for the web demo (runs in the `melodyflow` conda env).

Holds facebook/melodyflow-t24-30secs on the GPU and renders edits on demand:
  GET  /health          -> {"status": "ok", "model": ...}
  POST /edit            {"wav_path": ..., "prompt": ...} -> {"edited_path": ..., "gen_time_s": ...}

Same editing params as the dataset generation, so demo renders match training targets.
"""

import functools
import threading
import time
from pathlib import Path

import torch

# MelodyFlow ckpts store omegaconf objects; torch>=2.6 rejects them by default
torch.load = functools.partial(torch.load, weights_only=False)

import torchaudio  # noqa: E402
import uvicorn  # noqa: E402
from audiocraft.data.audio import audio_write  # noqa: E402
from audiocraft.data.audio_utils import convert_audio  # noqa: E402
from audiocraft.models import MelodyFlow  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "webapp_edits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/melodyflow-t24-30secs"
EDIT_PARAMS = dict(solver="euler", steps=25, target_flowstep=0.0, regularize=True,
                   regularize_iters=4, keep_last_k_iters=2, lambda_kl=0.2)

print(f"loading {MODEL_NAME} ...", flush=True)
model = MelodyFlow.get_pretrained(MODEL_NAME, device="cuda")
model.set_editing_params(**EDIT_PARAMS)
print("model ready", flush=True)

app = FastAPI()
gpu_lock = threading.Lock()


class EditRequest(BaseModel):
    wav_path: str
    prompt: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/edit")
def edit(req: EditRequest):
    wav_path = Path(req.wav_path)
    if not wav_path.is_absolute():
        wav_path = PROJECT_ROOT / wav_path
    if not wav_path.exists():
        raise HTTPException(404, f"wav not found: {wav_path}")
    t0 = time.time()
    with gpu_lock:
        wav, sr = torchaudio.load(wav_path)
        wav = convert_audio(wav, sr, model.sample_rate, model.audio_channels)
        wav = wav[..., : int(model.sample_rate * model.duration)]
        with torch.no_grad():
            tokens = model.encode_audio(wav.unsqueeze(0).cuda())
            out = model.edit(prompt_tokens=tokens, descriptions=[req.prompt],
                             src_descriptions=[""], progress=False, return_tokens=False)
        out_path = OUT_DIR / f"{wav_path.stem}_{int(time.time())}.wav"
        audio_write(str(out_path.with_suffix("")), out[0].cpu().float(), model.sample_rate,
                    strategy="loudness", loudness_headroom_db=16,
                    loudness_compressor=True, add_suffix=True)
    return {"edited_path": str(out_path), "gen_time_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import os
    uvicorn.run(app, host="127.0.0.1",
                port=int(os.environ.get("MELODYFLOW_PORT", "7861")), log_level="warning")
