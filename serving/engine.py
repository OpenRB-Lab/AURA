"""AURA inference engine — loads the thinker + fusion bridge + stem executor
once and edits audio from a natural-language instruction. Shared by the FastAPI
service (api.py) and the Gradio app (gradio_app.py).

Replicates the single-example path of edit_agent/run_stem_bench.py:
thinker -> generate_with_edit_capture -> stem_pipeline executor (hybrid default).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]        # .../diffusion
sys.path.insert(0, str(PROJECT_ROOT / "src"))
torch.backends.cudnn.enabled = False                      # repo disables cuDNN
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from edit_agent.models.qwen_wrapper import (              # noqa: E402
    generate_with_edit_capture, load_audio_16k, load_thinker)
from edit_agent.models.musicgen_fusion import build_fusion_bridge  # noqa: E402
from edit_agent.dataloaders.sft_data import SYSTEM_PROMPT  # noqa: E402
from edit_agent import stem_pipeline as sp                # noqa: E402

SR = 32000
IM_END = 151645
CKPT = PROJECT_ROOT / "ckpts" / "edit_agent"


def _p(env, default):
    return os.environ.get(env, str(CKPT / default))


class AuraEngine:
    """Load once, edit many times."""

    def __init__(self, qwen: str | None = None, mg: str | None = None,
                 classifier: str | None = None, device: str = "cuda"):
        self.device = device
        self.qwen = qwen or _p("AURA_QWEN", "joint_fusion_r64/final/qwen")
        self.mg = mg or _p("AURA_MG", "joint_fusion_r64/final/musicgen")
        self.classifier = classifier or _p("AURA_CLASSIFIER", "joint_cls/final/classifier.pt")

        self.model, self.proc, self.edit_ids = load_thinker(
            lora_dir=self.qwen, merge_lora=True, device=device)
        self.model.eval()
        self.bridge = build_fusion_bridge(torch.device(device))
        self.bridge.load_adapter(self.mg, device)
        self.bridge.decoder.to(device, dtype=torch.bfloat16)
        self.bridge.eval()
        self.classifier_m = sp.load_classifier(self.classifier, device)
        self.separator = sp.Separator(device)

    @torch.no_grad()
    def _thinker(self, wav_path: str, instruction: str, history=None):
        """Return (reply_text, edit hidden states h or None)."""
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        for turn in (history or []):                      # [{"role","text"}] prior turns
            msgs.append({"role": turn["role"],
                         "content": [{"type": "text", "text": turn["text"]}]})
        msgs.append({"role": "user",
                     "content": [{"type": "audio", "audio": wav_path},
                                 {"type": "text", "text": instruction}]})
        tmpl = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[tmpl], audio=[load_audio_16k(wav_path)],
                           return_tensors="pt", padding=True).to(self.device)
        return generate_with_edit_capture(self.model, self.proc, inputs,
                                          self.edit_ids, eos_token_id=IM_END)

    @torch.no_grad()
    def edit(self, audio_path: str, instruction: str, history=None,
             executor: str = "hybrid", planner: str = "programmatic",
             guidance: float = 2.0, seed: int = 1234, max_seconds: float = 5.0):
        """Edit `audio_path` per `instruction`. Returns dict with the reply,
        the output wav (float32 numpy, 32 kHz), sample rate, plan, and timing.
        If the thinker emits no edit block, returns the reply with wav=None."""
        t0 = time.time()
        reply, h = self._thinker(audio_path, instruction, history)
        if h is None:                                     # conversational, no edit
            return {"reply": reply, "wav": None, "sr": SR, "plan": None,
                    "gen_time_s": round(time.time() - t0, 2)}

        plan = sp.parse_plan(reply) if planner == "cot" else None
        if plan is None:
            plan = sp.programmatic_plan(reply, h, self.classifier_m, instruction=instruction)
            if planner == "cot":
                plan.fallback_reason = "plan parse failed"

        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        y = y.astype(np.float32)
        if executor == "full":
            wav = sp.full_mix_execute(h, y, self.bridge, seed, guidance=guidance)
        elif executor == "pure":
            wav, plan = sp.pure_execute(plan, y, h, self.bridge, self.separator,
                                        seed, lambda instr: self._thinker(audio_path, instr)[1])
        else:                                             # hybrid (flagship)
            wav, plan = sp.hybrid_execute(plan, y, h, self.bridge, self.separator, seed)
        wav = wav[: int(max_seconds * SR)]
        return {"reply": reply, "wav": wav.astype(np.float32), "sr": SR,
                "plan": plan.to_json() if hasattr(plan, "to_json") else None,
                "gen_time_s": round(time.time() - t0, 2)}


def write_temp_wav(data: bytes) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.write(data); f.flush(); f.close()
    return f.name
