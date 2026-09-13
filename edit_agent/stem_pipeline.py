"""Stem-decomposition pipeline: plan -> per-stem execute -> merge.

The agent reasons about which stems the output needs (a CoT plan, either
parsed from the thinker reply or built programmatically), executes each stem
with the cheapest faithful operator (copy the input / Demucs-separate /
MusicGen-generate), and merges by waveform sum.

Executors:
- hybrid: add    = input + (stem separated from the generated full mix)
          remove = input - separated target stem
          extract= separated target stem
          anything else (or unmappable instrument / low confidence)
          falls back to the production full-mix bridge path.
- full:   the exact run_impg_bench.py generation path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import librosa
import numpy as np
import torch

from edit_agent.models.edit_classifier import EditSemanticClassifier, INST_CLASSES

SR = 32000
KIND_RE = re.compile(r"\[EDIT_([A-Z]+)\](?=\[EDIT_0\])")
PLAN_RE = re.compile(
    r"<plan>\s*input:\s*(?P<input>.*?)\s*\|\s*action:\s*(?P<action>.*?)"
    r"\s*\|\s*output:\s*(?P<output>.*?)\s*</plan>", re.S)

# 10-class taxonomy -> htdemucs_6s source (unmappable classes -> full-mix)
INST_TO_SOURCE = {"drums": "drums", "bass": "bass", "guitar": "guitar",
                  "keys": "piano", "vocals": "vocals"}
KIND_TO_TASK = {"ADD": "add", "REMOVE": "remove", "EXTRACT": "extract"}
# extract of a near-absent stem would return silence; below this target-source
# energy fraction (dB re: total) we fall back to the full-mix path.
EXTRACT_CONF_DB = -25.0


@dataclass
class Plan:
    kind: str | None = None          # EDIT kind token, e.g. "ADD"
    inst: str | None = None          # 10-class instrument
    input_insts: list = field(default_factory=list)
    source: str = "programmatic"     # "cot" | "programmatic"
    fallback_reason: str | None = None

    def to_json(self):
        return {"kind": self.kind, "inst": self.inst,
                "input_insts": self.input_insts, "source": self.source,
                "fallback_reason": self.fallback_reason}


# ── separation ───────────────────────────────────────────────────


class Separator:
    """htdemucs_6s wrapper working on 32 kHz mono numpy audio."""

    def __init__(self, device: str = "cuda"):
        from demucs.pretrained import get_model
        self.model = get_model("htdemucs_6s").to(device).eval()
        self.device = device
        self.sr = self.model.samplerate  # 44100
        self.sources = list(self.model.sources)

    @torch.no_grad()
    def separate(self, wav_32k: np.ndarray) -> dict[str, np.ndarray]:
        """-> {source: mono 32 kHz wav of the same length as the input}."""
        from demucs.apply import apply_model
        n = len(wav_32k)
        y = librosa.resample(wav_32k.astype(np.float32), orig_sr=SR,
                             target_sr=self.sr)
        x = torch.from_numpy(y).float().to(self.device)
        x = x.unsqueeze(0).repeat(2, 1).unsqueeze(0)      # [1, 2, T]
        ref = x.mean(1)
        mean, std = ref.mean(), ref.std() + 1e-8
        x = (x - mean) / std
        out = apply_model(self.model, x, device=self.device)[0]  # [S, 2, T]
        out = out * std + mean
        stems = {}
        for k, name in enumerate(self.sources):
            s = out[k].mean(0).cpu().numpy()
            s = librosa.resample(s, orig_sr=self.sr, target_sr=SR)
            stems[name] = _fit_length(s, n)
        return stems


def _fit_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def peak_norm(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    p = np.abs(y).max()
    return y / p * peak if p > 0 else y


def source_energy_db(stems: dict[str, np.ndarray], source: str) -> float:
    e = {k: float(np.mean(v ** 2)) for k, v in stems.items()}
    tot = sum(e.values()) + 1e-12
    return 10 * np.log10(e.get(source, 0.0) / tot + 1e-12)


def active_sources(stems: dict[str, np.ndarray], th_db: float = -30.0) -> list:
    return [k for k in stems if source_energy_db(stems, k) > th_db]


# ── planning ─────────────────────────────────────────────────────


def parse_plan(reply: str) -> Plan | None:
    """Parse a thinker-emitted <plan> block. None if absent/unparseable."""
    m = PLAN_RE.search(reply)
    if not m:
        return None
    action = m.group("action").split()
    kind = action[0].upper() if action else None
    inst = action[1].lower() if len(action) > 1 else None
    if inst is not None and inst not in INST_CLASSES:
        inst = None
    ins = [s.strip() for s in m.group("input").split(",")]
    ins = [s for s in ins if s in INST_CLASSES]
    return Plan(kind=kind, inst=inst, input_insts=ins, source="cot")


def load_classifier(path, device: str = "cuda") -> EditSemanticClassifier:
    clf = EditSemanticClassifier()
    clf.load_state_dict(torch.load(path, map_location="cpu"))
    return clf.to(device).eval()


@torch.no_grad()
def programmatic_plan(reply: str, h: torch.Tensor,
                      classifier: EditSemanticClassifier,
                      instruction: str = "") -> Plan:
    """Kind from the emitted edit token; instrument from the instruction/reply
    text (same keyword map the semantic labels use), classifier as fallback.

    Text first: the classifier was trained on hidden states of the thinker it
    was jointly trained with and does not transfer across thinker checkpoints.
    """
    from edit_agent.build_semantic_labels import inst_from_text
    m = KIND_RE.search(reply)
    kind = m.group(1) if m else None
    inst = inst_from_text(instruction) or \
        inst_from_text(KIND_RE.split(reply)[0] if kind else reply)
    if inst is None and h is not None and classifier is not None:
        dev = next(classifier.parameters()).device
        _, il = classifier(h.float().unsqueeze(0).to(dev))
        inst = INST_CLASSES[int(il.argmax(-1).item())]
    return Plan(kind=kind, inst=inst, source="programmatic")


# ── executors ────────────────────────────────────────────────────


N_FRAMES = 250  # 5 s at 50 Hz


@torch.no_grad()
def full_mix_execute(h, input_wav, bridge, seed, n_frames=N_FRAMES, guidance=2.0):
    """The production path (run_impg_bench.py) as a function."""
    dev = next(bridge.decoder.parameters()).device
    src_wav = torch.from_numpy(input_wav).to(dev).view(1, 1, -1)
    codes = bridge.audio_encoder.encode(src_wav, bandwidth=None).audio_codes
    src = (codes[0] if codes.dim() == 4 else codes)[0, :, :n_frames]
    gen = bridge.generate(h.float().unsqueeze(0).to(dev),
                          src.long().unsqueeze(0),
                          max_frames=n_frames, guidance=guidance, seed=seed)
    wav = bridge.decode_audio(gen).squeeze().float().cpu().numpy()
    return peak_norm(wav)


@torch.no_grad()
def gen_stem(h, input_wav, bridge, seed, n_frames=N_FRAMES, stem_flag=True):
    """Generate audio at raw level (no normalization — training targets live
    at the source mix's scale). stem_flag=True turns the stem-mode memory
    row/bias on; False = native conditioning (for isolate-instruction h)."""
    dev = next(bridge.decoder.parameters()).device
    src_wav = torch.from_numpy(input_wav).to(dev).view(1, 1, -1)
    codes = bridge.audio_encoder.encode(src_wav, bandwidth=None).audio_codes
    src = (codes[0] if codes.dim() == 4 else codes)[0, :, :n_frames]
    sm = torch.tensor([True], device=dev) if stem_flag else None
    gen = bridge.generate(h.float().unsqueeze(0).to(dev),
                          src.long().unsqueeze(0), max_frames=n_frames,
                          guidance=2.0, seed=seed, stem_mode=sm)
    return bridge.decode_audio(gen).squeeze().float().cpu().numpy()


@torch.no_grad()
def hybrid_execute(plan: Plan, input_wav: np.ndarray, h, bridge,
                   separator: Separator, seed: int,
                   n_frames: int = N_FRAMES,
                   stem_gen: bool = False) -> tuple[np.ndarray, Plan]:
    """Copy/separate where possible; generate only what doesn't exist yet.

    stem_gen: generate the added stem directly (stem-mode bridge) instead of
    separating it out of a generated full mix.
    Mutates plan.fallback_reason when the full-mix path is used.
    """
    task = KIND_TO_TASK.get(plan.kind or "")
    source = INST_TO_SOURCE.get(plan.inst or "")
    if task is None:
        plan.fallback_reason = f"kind={plan.kind}"
    elif source is None:
        plan.fallback_reason = f"unmappable inst={plan.inst}"
    if plan.fallback_reason:
        return full_mix_execute(h, input_wav, bridge, seed, n_frames), plan

    if task == "add":
        if stem_gen:
            stem = gen_stem(h, input_wav, bridge, seed, n_frames)
            out = input_wav + _fit_length(stem, len(input_wav))
            return peak_norm(out), plan
        gen_mix = full_mix_execute(h, input_wav, bridge, seed, n_frames)
        gen_mix = _fit_length(gen_mix, len(input_wav))
        stems = separator.separate(gen_mix)
        out = input_wav + stems[source]
        return peak_norm(out), plan

    stems = separator.separate(input_wav)
    if not plan.input_insts:
        plan.input_insts = active_sources(stems)
    if task == "extract":
        if source_energy_db(stems, source) < EXTRACT_CONF_DB:
            plan.fallback_reason = f"low sep confidence for {source}"
            return full_mix_execute(h, input_wav, bridge, seed, n_frames), plan
        return peak_norm(stems[source]), plan
    # remove: subtraction keeps untouched content bit-faithful up to residual
    out = input_wav - stems[source]
    return peak_norm(out), plan


SOURCE_PROMPT = {"drums": "drums", "bass": "bass", "guitar": "guitar",
                 "piano": "piano", "vocals": "vocals"}


@torch.no_grad()
def pure_execute(plan: Plan, input_wav: np.ndarray, h, bridge,
                 separator: Separator, seed: int, thinker_fn,
                 n_frames: int = N_FRAMES) -> tuple[np.ndarray, Plan]:
    """Fully generative ablation: EVERY output stem is generated by the
    stem-mode bridge, then summed.

    Existing stems are regenerated with a synthetic "keep only the X"
    instruction (matches the slakh_isolate training distribution) via
    thinker_fn(instruction) -> h. The edit's own stem uses the user-turn h.
    """
    task = KIND_TO_TASK.get(plan.kind or "")
    source = INST_TO_SOURCE.get(plan.inst or "")
    if task is None or source is None:
        plan.fallback_reason = (f"kind={plan.kind}" if task is None
                                else f"unmappable inst={plan.inst}")
        return full_mix_execute(h, input_wav, bridge, seed, n_frames), plan

    if task == "extract":
        return peak_norm(gen_stem(h, input_wav, bridge, seed, n_frames)), plan

    stems = separator.separate(input_wav)
    keep = [s for s in active_sources(stems) if s in SOURCE_PROMPT]
    plan.input_insts = keep
    if task == "remove":
        keep = [s for s in keep if s != source]
    out = np.zeros_like(input_wav)
    for j, s in enumerate(keep):
        h_s = thinker_fn(f"keep only the {SOURCE_PROMPT[s]}")
        if h_s is None:
            continue
        # native mode: isolate-instruction conditioning already targets a
        # lone stem in training — no stem flag needed for existing stems
        stem = gen_stem(h_s, input_wav, bridge, seed + 17 * (j + 1),
                        n_frames, stem_flag=False)
        out = out + _fit_length(stem, len(out))
    if task == "add":
        stem = gen_stem(h, input_wav, bridge, seed, n_frames)
        out = out + _fit_length(stem, len(out))
    if not np.abs(out).max():
        plan.fallback_reason = "no stems generated"
        return full_mix_execute(h, input_wav, bridge, seed, n_frames), plan
    return peak_norm(out), plan
