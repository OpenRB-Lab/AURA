"""MusicGen edit bridge v2 — per-layer K/V source fusion (Instruct-MusicGen style).

Difference from musicgen_bridge.MusicGenEditBridge: the source audio conditions
the decoder through its OWN per-layer self-attention K/V (the target sequence is
generated as a continuation of the source sequence, so every layer attends to
full-resolution source states), instead of a projected-embedding cross-attn memory.
The 9 [EDIT] hidden states remain the cross-attention memory (the instruction).

Training: input = [delayed src ; delayed tgt] (2*T frames), CE on the tgt half.
Inference: one forward over the source fills past_key_values; the target is then
sampled autoregressively on top. CFG contrasts edit-conditioned vs zeroed-h
branches (both keep the source prefix).

Trainable: LoRA r16/a32 on self-attn AND cross-attn projections + proj_h.
"""

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.musicgen_bridge import SPECIAL, delay_codes, undelay_codes  # noqa: E402

D_LLM = 3584


class MusicGenKVBridge(nn.Module):
    def __init__(self, model_id: str = "facebook/musicgen-medium",
                 cache_dir: str | None = None, lora_r: int = 16, lora_alpha: int = 32,
                 lora_mlp: bool = False):
        super().__init__()
        from transformers import MusicgenForConditionalGeneration
        cache_dir = cache_dir or str(PROJECT_ROOT / "weights")
        full = MusicgenForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, torch_dtype=torch.float32)
        self.decoder = full.decoder
        self.audio_encoder = full.audio_encoder
        del full
        self.cfg = self.decoder.config
        d = self.cfg.hidden_size
        self.num_codebooks = self.cfg.num_codebooks

        for p in self.decoder.parameters():
            p.requires_grad_(False)
        for p in self.audio_encoder.parameters():
            p.requires_grad_(False)

        from peft import LoraConfig, get_peft_model
        targets = r".*(self_attn|encoder_attn)\.(q_proj|k_proj|v_proj|out_proj)"
        if lora_mlp:
            targets = (r".*((self_attn|encoder_attn)\.(q_proj|k_proj|v_proj|out_proj)"
                       r"|fc1|fc2)")
        lcfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
            target_modules=targets, bias="none")
        self.decoder = get_peft_model(self.decoder, lcfg)

        self.h_norm = nn.LayerNorm(D_LLM)
        self.proj_h = nn.Sequential(nn.Linear(D_LLM, d), nn.GELU(), nn.Linear(d, d))

    def _dtype(self):
        return next(iter(self.decoder.parameters())).dtype

    def edit_memory(self, h, drop_edit=None):
        """[B,9,3584] -> [B,9,d] cross-attn memory (the edit instruction)."""
        hh = self.proj_h(self.h_norm(h.float()))
        if drop_edit is not None:
            hh = hh * (~drop_edit)[:, None, None]
        return hh.to(self._dtype())

    # ── training ────────────────────────────────────────────────
    def forward(self, h, src_codes, tgt_codes, n_frames=None,
                edit_drop_prob: float = 0.1):
        """Input [delayed src ; delayed tgt]; CE on the target half only."""
        b, k, t = tgt_codes.shape
        dev = tgt_codes.device
        drop = (torch.rand(b, device=dev) < edit_drop_prob) if self.training else None
        mem = self.edit_memory(h, drop_edit=drop)

        d_src = delay_codes(src_codes)
        d_tgt = delay_codes(tgt_codes)
        seq = torch.cat([d_src, d_tgt], dim=-1)                      # [B,K,2T]
        inp = torch.cat([torch.full((b, k, 1), SPECIAL, device=dev,
                                    dtype=seq.dtype), seq[:, :, :-1]], dim=-1)
        labels = seq.permute(0, 2, 1).contiguous()                   # [B,2T,K]
        labels[:, :t, :] = SPECIAL                                   # src half: no loss
        if n_frames is not None:                                     # padded tgt tail
            pos = torch.arange(t, device=dev)[None, :]
            pad = (pos >= n_frames[:, None])                         # [B,T]
            labels[:, t:, :] = labels[:, t:, :].masked_fill(pad[..., None], SPECIAL)

        out = self.decoder(input_ids=inp.reshape(b * k, 2 * t),
                           encoder_hidden_states=mem,
                           labels=labels)
        return out.loss

    # ── inference ───────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, h, src_codes, max_frames: int = 500, guidance: float = 2.0,
                 temperature: float = 1.0, top_k: int = 250, seed: int | None = None,
                 segment=None):
        """Sample target as continuation of the source K/V prefix.

        segment: optional (start_frame, end_frame) — outside it, source codes are
        teacher-forced (bit-exact preservation for localized edits).
        """
        if seed is not None:
            torch.manual_seed(seed)
        dev = src_codes.device
        k = self.num_codebooks
        bsz = 2 if guidance != 1.0 else 1
        mem_c = self.edit_memory(h)
        mem = torch.cat([mem_c, torch.zeros_like(mem_c)], 0) if bsz == 2 else mem_c

        d_src = delay_codes(src_codes)                                # [1,K,T]
        prefix = torch.cat([torch.full((1, k, 1), SPECIAL, device=dev,
                                       dtype=torch.long), d_src], dim=-1)
        out = self.decoder(input_ids=prefix.expand(bsz, k, -1).reshape(bsz * k, -1),
                           encoder_hidden_states=mem, use_cache=True)
        past = out.past_key_values
        logits0 = out.logits[:, -1]

        d_src_pad = torch.cat([d_src, torch.full((1, k, max_frames + k), SPECIAL,
                                                 device=dev, dtype=torch.long)], -1)
        seq = []
        step_logits = logits0
        for step in range(max_frames + k - 1):
            logits = step_logits.view(bsz, k, -1)
            if bsz == 2:
                lc, ln = logits[:1], logits[1:]
                logits = ln + (lc - ln) * guidance
            logits = logits / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits.float(), dim=-1)
            nxt = torch.multinomial(probs.view(k, -1), 1).view(1, k, 1)
            for i in range(k):
                if step < i:
                    nxt[0, i] = SPECIAL
                elif segment is not None:
                    fr = step - i                                     # undelayed frame
                    if fr < segment[0] or fr >= segment[1]:
                        nxt[0, i] = d_src_pad[0, i, step]             # copy source
            seq.append(nxt)
            out = self.decoder(input_ids=nxt.expand(bsz, k, 1).reshape(bsz * k, 1),
                               encoder_hidden_states=mem,
                               past_key_values=past, use_cache=True)
            past = out.past_key_values
            step_logits = out.logits[:, -1]
        delayed = torch.cat(seq, dim=-1)
        return undelay_codes(delayed).clamp(0, 2047)

    @torch.no_grad()
    def decode_audio(self, codes):
        wav = self.audio_encoder.decode(codes.unsqueeze(0), audio_scales=[None])
        return wav.audio_values[0]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_adapter(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.decoder.save_pretrained(save_dir / "lora")
        torch.save({"h_norm": self.h_norm.state_dict(),
                    "proj_h": self.proj_h.state_dict()}, save_dir / "projectors.pt")

    def load_adapter(self, load_dir, device="cuda"):
        from peft import PeftModel
        load_dir = Path(load_dir)
        self.decoder = PeftModel.from_pretrained(self.decoder.get_base_model(),
                                                 load_dir / "lora", is_trainable=True)
        extra = torch.load(load_dir / "projectors.pt", map_location=device)
        self.h_norm.load_state_dict(extra["h_norm"])
        self.proj_h.load_state_dict(extra["proj_h"])
        return self


def build_kv_bridge(device: torch.device, dtype=torch.bfloat16, lora_r: int = 16,
                    lora_alpha: int = 32, lora_mlp: bool = False) -> MusicGenKVBridge:
    m = MusicGenKVBridge(lora_r=lora_r, lora_alpha=lora_alpha, lora_mlp=lora_mlp)
    m.decoder.to(device, dtype=dtype)
    m.audio_encoder.to(device)
    m.h_norm.to(device).float()
    m.proj_h.to(device).float()
    return m
