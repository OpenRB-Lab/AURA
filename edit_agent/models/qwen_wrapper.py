"""Qwen2.5-Omni thinker loading, [EDIT] token registration, and audio helpers.

Everything that touches the Omni model goes through here so the gotchas live in one
place: sdpa attention (no flash-attn in this env), cuDNN disabled (broken in env),
audio at 16 kHz mono with dynamic padding, and [EDIT] token ids resolved at runtime
(they may fit inside the checkpoint's padded vocab without a resize).
"""

import sys
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.enabled = False  # cuDNN init is broken in the llama env
# SDPA picks its own backends independently of cudnn.enabled — exclude the cuDNN one
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.tokens import ALL_SPECIAL_TOKENS, EDIT_TOKENS, EDIT_TYPE_TOKENS  # noqa: E402

WEIGHTS_CACHE = PROJECT_ROOT / "weights"
DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-7B"
AUDIO_SR = 16000


def add_edit_tokens(processor, model=None) -> list[int]:
    """Register [EDIT_0..7] as special tokens; resize embeddings only if needed."""
    tok = processor.tokenizer
    n_added = tok.add_tokens(ALL_SPECIAL_TOKENS, special_tokens=True)
    ids = tok.convert_tokens_to_ids(EDIT_TOKENS)  # payload block ids (bridge interface)
    assert all(isinstance(i, int) and i >= 0 for i in ids), f"bad edit token ids: {ids}"
    if model is not None:
        vocab = model.config.text_config.vocab_size \
            if hasattr(model.config, "text_config") else model.config.vocab_size
        if max(ids) >= vocab:
            model.resize_token_embeddings(len(tok))
            print(f"[tokens] resized embeddings to {len(tok)} (ids {ids})")
        else:
            print(f"[tokens] {n_added} tokens fit in padded vocab {vocab} (ids {ids})")
    return ids


def load_thinker(model_id: str = DEFAULT_MODEL, lora_dir: str | None = None,
                 device: str = "cuda", dtype=torch.bfloat16, merge_lora: bool = False):
    """Load thinker + processor (+ optional LoRA adapter). Returns (model, processor, edit_ids)."""
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    processor = Qwen2_5OmniProcessor.from_pretrained(model_id, cache_dir=WEIGHTS_CACHE)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id, cache_dir=WEIGHTS_CACHE, torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    edit_ids = add_edit_tokens(processor, model)
    if lora_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        if merge_lora:
            model = model.merge_and_unload()
    return model.to(device), processor, edit_ids


def load_audio_16k(wav_path: str | Path) -> np.ndarray:
    """48 kHz stereo dataset wav → 16 kHz mono float32 for the Omni audio encoder."""
    import librosa
    path = Path(wav_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    y, _ = librosa.load(path, sr=AUDIO_SR, mono=True)
    return y.astype(np.float32)


@torch.no_grad()
def generate_with_edit_capture(model, processor, inputs, edit_ids: list[int],
                               max_new_tokens: int = 256, eos_token_id: int = 151645):
    """Greedy generate; if a typed edit block [EDIT_<KIND>][EDIT_0..7] is emitted,
    capture the 9 last-layer hidden states that predicted those tokens.

    Returns (reply_text, h) with h a [9, 3584] tensor or None. The state layout
    matches the teacher-forced bridge cache (precompute_bridge run_hidden):
    hidden_states[step j] is the forward that predicted generated token j, so the
    9 states for steps j0..j0+8 predict [KIND], [EDIT_0], ..., [EDIT_7].
    """
    tok = processor.tokenizer
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         eos_token_id=eos_token_id, pad_token_id=eos_token_id,
                         return_dict_in_generate=True, output_hidden_states=True)
    prompt_len = inputs["input_ids"].shape[1]
    gen = out.sequences[0][prompt_len:].tolist()
    kind_ids = {tok.convert_tokens_to_ids(t) for t in EDIT_TYPE_TOKENS.values()}
    h = None
    for j, tid in enumerate(gen):
        if tid in kind_ids and gen[j + 1:j + 9] == list(edit_ids):
            # step 0's pass covers the whole prompt; [0, -1] is uniform-correct
            h = torch.stack([out.hidden_states[j + m][-1][0, -1] for m in range(9)])
            break
    text = tok.decode(gen, skip_special_tokens=False).replace("<|im_end|>", "").strip()
    return text, h


def find_edit_positions(input_ids: torch.Tensor, edit_ids: list[int]) -> torch.Tensor:
    """Positions of the contiguous [EDIT_0..7] block in PROCESSED ids (1-D tensor)."""
    pos = (input_ids == edit_ids[0]).nonzero(as_tuple=True)[0]
    assert len(pos) == 1, f"expected exactly one [EDIT_0], found {len(pos)}"
    start = int(pos[0])
    block = input_ids[start:start + len(edit_ids)].tolist()
    assert block == edit_ids, f"edit block not contiguous at {start}: {block}"
    return torch.arange(start, start + len(edit_ids))
