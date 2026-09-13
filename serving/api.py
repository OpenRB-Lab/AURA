"""AURA FastAPI service.

  CUDA_VISIBLE_DEVICES=0 API_PORT=9005 conda run -n llama python -u src/serving/api.py

Endpoints:
  GET  /health  -> {ok, checkpoints}
  POST /chat    -> {audio_b64, instruction[, history]}      -> {reply}
  POST /edit    -> {audio_b64, instruction[, history, executor, guidance, seed]}
                   -> {reply, audio_b64 (32 kHz wav), plan, edited (bool), gen_time_s}
Audio is base64-encoded WAV in and out.
"""
import base64
import io
import os

import soundfile as sf
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from engine import AuraEngine, write_temp_wav

app = FastAPI(title="aura-api")
ENGINE: AuraEngine | None = None


class ChatReq(BaseModel):
    audio_b64: str
    instruction: str
    history: list[dict] | None = None


class EditReq(ChatReq):
    executor: str = "hybrid"        # hybrid | pure | full
    planner: str = "programmatic"   # programmatic | cot
    guidance: float = 2.0
    seed: int = 1234


def _wav_b64(wav, sr) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


@app.on_event("startup")
def _load():
    global ENGINE
    ENGINE = AuraEngine(device="cuda")


@app.get("/health")
def health():
    return {"ok": ENGINE is not None,
            "checkpoints": None if ENGINE is None else
            {"qwen": ENGINE.qwen, "mg": ENGINE.mg, "classifier": ENGINE.classifier}}


@app.post("/chat")
def chat(req: ChatReq):
    path = write_temp_wav(base64.b64decode(req.audio_b64))
    reply, _ = ENGINE._thinker(path, req.instruction, req.history)
    os.unlink(path)
    return {"reply": reply}


@app.post("/edit")
def edit(req: EditReq):
    path = write_temp_wav(base64.b64decode(req.audio_b64))
    r = ENGINE.edit(path, req.instruction, history=req.history,
                    executor=req.executor, planner=req.planner,
                    guidance=req.guidance, seed=req.seed)
    os.unlink(path)
    out = {"reply": r["reply"], "plan": r["plan"], "gen_time_s": r["gen_time_s"],
           "edited": r["wav"] is not None}
    if r["wav"] is not None:
        out["audio_b64"] = _wav_b64(r["wav"], r["sr"])
    return out


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 9005)))
