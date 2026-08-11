"""Text-instruction MusicGen editor — faithful Instruct-MusicGen baseline.

No MuLan, no LLM hidden states. Conditioning memory for the frozen
musicgen-medium decoder:
  [ enc_to_dec_proj(T5(instruction))   <- pretrained text path, untouched
  ; proj_src(embed(src EnCodec codes)) <- audio fusion, zero-init projection ]
Trainable: LoRA r16/a32 on cross-attn q/k/v/out + proj_src. Loss: per-codebook
CE on delayed target codes. CFG at inference drops the TEXT part only (source
audio stays in both branches, so guidance steers the edit).
"""

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.musicgen_bridge import SPECIAL, delay_codes, undelay_codes  # noqa: E402


class MusicGenTextEdit(nn.Module):
    def __init__(self, model_id: str = "facebook/musicgen-medium",
                 cache_dir: str | None = None, lora_r: int = 16, lora_alpha: int = 32):
        super().__init__()
        from transformers import MusicgenForConditionalGeneration
        cache_dir = cache_dir or str(PROJECT_ROOT / "weights")
        full = MusicgenForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, torch_dtype=torch.float32)
        self.decoder = full.decoder
        self.text_encoder = full.text_encoder
        self.enc_to_dec_proj = full.enc_to_dec_proj
        self.audio_encoder = full.audio_encoder
        del full
        self.cfg = self.decoder.config
        d = self.cfg.hidden_size
        self.num_codebooks = self.cfg.num_codebooks

        for m in (self.decoder, self.text_encoder, self.enc_to_dec_proj,
                  self.audio_encoder):
            for p in m.parameters():
                p.requires_grad_(False)

        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                          target_modules=r".*encoder_attn\.(q_proj|k_proj|v_proj|out_proj)",
                          bias="none")
        self.decoder = get_peft_model(self.decoder, lcfg)

        self.proj_src = nn.Linear(d, d)
        nn.init.zeros_(self.proj_src.weight)  # init = pure pretrained text conditioning

    def _dtype(self):
        return next(iter(self.decoder.parameters())).dtype

    def _embed_codes(self, codes):
        base = self.decoder.get_base_model() if hasattr(self.decoder, "get_base_model") \
            else self.decoder
        emb = base.model.decoder.embed_tokens
        x = sum(emb[k](codes[:, k].clamp(0, SPECIAL)) for k in range(self.num_codebooks))
        return x.float()

    def build_memory(self, text_ids, text_mask, src_codes,
                     drop_text: torch.Tensor | None = None):
        th = self.text_encoder(input_ids=text_ids,
                               attention_mask=text_mask).last_hidden_state
        mem_t = self.enc_to_dec_proj(th)
        mask_t = text_mask.clone()
        if drop_text is not None:
            mem_t = mem_t * (~drop_text)[:, None, None]
            mask_t = mask_t * (~drop_text)[:, None]
        src = self.proj_src(self._embed_codes(src_codes)).to(mem_t.dtype)
        mem = torch.cat([mem_t, src], dim=1)
        mask = torch.cat([mask_t, torch.ones(src.shape[:2], device=src.device,
                                             dtype=mask_t.dtype)], dim=1)
        return mem.to(self._dtype()), mask

    def forward(self, text_ids, text_mask, src_codes, tgt_codes, n_frames=None,
                text_drop_prob: float = 0.1):
        b, k, t = tgt_codes.shape
        dev = tgt_codes.device
        drop = (torch.rand(b, device=dev) < text_drop_prob) if self.training else None
        mem, mask = self.build_memory(text_ids, text_mask, src_codes, drop_text=drop)

        delayed = delay_codes(tgt_codes)
        inp = torch.cat([torch.full((b, k, 1), SPECIAL, device=dev,
                                    dtype=tgt_codes.dtype), delayed[:, :, :-1]], dim=-1)
        labels = delayed.permute(0, 2, 1).contiguous()
        if n_frames is not None:
            pos = torch.arange(t, device=dev)[None, :]
            labels = labels.masked_fill((pos >= n_frames[:, None])[..., None], SPECIAL)
        out = self.decoder(input_ids=inp.reshape(b * k, t),
                           encoder_hidden_states=mem,
                           encoder_attention_mask=mask,
                           labels=labels)
        return out.loss

    @torch.no_grad()
    def generate(self, text_ids, text_mask, src_codes, max_frames: int = 500,
                 guidance: float = 3.0, temperature: float = 1.0, top_k: int = 250,
                 seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        dev = src_codes.device
        k = self.num_codebooks
        mem_c, mask_c = self.build_memory(text_ids, text_mask, src_codes)
        if guidance != 1.0:
            drop = torch.ones(text_ids.shape[0], device=dev, dtype=torch.bool)
            mem_n, mask_n = self.build_memory(text_ids, text_mask, src_codes,
                                              drop_text=drop)
            mem = torch.cat([mem_c, mem_n], 0)
            mask = torch.cat([mask_c, mask_n], 0)
        else:
            mem, mask = mem_c, mask_c

        seq = torch.full((1, k, 1), SPECIAL, device=dev, dtype=torch.long)
        past = None
        for step in range(max_frames + k - 1):
            bsz = mem.shape[0]
            inp = seq[:, :, -1:].expand(bsz, k, 1) if past is not None \
                else seq.expand(bsz, k, seq.shape[-1])
            out = self.decoder(input_ids=inp.reshape(bsz * k, -1),
                               encoder_hidden_states=mem,
                               encoder_attention_mask=mask,
                               past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1].view(bsz, k, -1)
            if guidance != 1.0:
                lc, ln = logits[:1], logits[1:]
                logits = ln + (lc - ln) * guidance
            logits = logits / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs.view(k, -1), 1).view(1, k, 1)
            for i in range(k):
                if step < i:
                    nxt[0, i] = SPECIAL
            seq = torch.cat([seq, nxt], dim=-1)
        return undelay_codes(seq[:, :, 1:]).clamp(0, 2047)

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
        torch.save({"proj_src": self.proj_src.state_dict()}, save_dir / "projectors.pt")

    def load_adapter(self, load_dir, device="cuda"):
        from peft import PeftModel
        load_dir = Path(load_dir)
        self.decoder = PeftModel.from_pretrained(self.decoder.get_base_model(),
                                                 load_dir / "lora", is_trainable=True)
        extra = torch.load(load_dir / "projectors.pt", map_location=device)
        self.proj_src.load_state_dict(extra["proj_src"])
        return self


def build_text_editor(device: torch.device, dtype=torch.bfloat16) -> MusicGenTextEdit:
    m = MusicGenTextEdit()
    m.decoder.to(device, dtype=dtype)
    m.text_encoder.to(device, dtype=dtype)
    m.enc_to_dec_proj.to(device, dtype=dtype)
    m.audio_encoder.to(device)
    m.proj_src.to(device).float()
    return m
