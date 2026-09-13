"""MusicGen edit bridge, Instruct-MusicGen style (arXiv 2405.18386).

Base: facebook/musicgen-medium decoder (48 layers, d=1536, 4 codebooks), FROZEN.
Conditioning (cross-attention memory, mirrors the paper's text+audio fusion):
  [ proj_h([EDIT] hidden states, 9 x 3584)  ;  proj_src(embed(src EnCodec codes)) ]
where embed() sums the decoder's own (frozen) codebook input embeddings.
Trainable: LoRA r16/a32 on every cross-attn q/k/v/out projection + the two
projectors (+ their LayerNorms). Loss: per-codebook CE on the DELAYED target
codes (audiocraft delay pattern), pad/-100 on the k leading BOS positions.
"""

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

D_LLM = 3584
SPECIAL = 2048  # musicgen BOS == PAD == 2048 (outside the 2048-way vocab)


def delay_codes(codes: torch.Tensor) -> torch.Tensor:
    """[B, K, T] -> delayed [B, K, T]: codebook k shifted right by k, BOS-filled."""
    b, k, t = codes.shape
    out = torch.full_like(codes, SPECIAL)
    for i in range(k):
        out[:, i, i:] = codes[:, i, : t - i]
    return out


def undelay_codes(delayed: torch.Tensor) -> torch.Tensor:
    """Inverse of delay_codes (tail of each codebook is lost -> trimmed by K-1)."""
    b, k, t = delayed.shape
    out = torch.full_like(delayed, SPECIAL)
    for i in range(k):
        out[:, i, : t - i] = delayed[:, i, i:]
    return out[:, :, : t - (k - 1)]


class MusicGenEditBridge(nn.Module):
    def __init__(self, model_id: str = "facebook/musicgen-medium",
                 cache_dir: str | None = None, lora_r: int = 16, lora_alpha: int = 32):
        super().__init__()
        from transformers import MusicgenForConditionalGeneration
        cache_dir = cache_dir or str(PROJECT_ROOT / "weights")
        full = MusicgenForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, torch_dtype=torch.float32)
        self.decoder = full.decoder            # MusicgenForCausalLM
        self.audio_encoder = full.audio_encoder  # EnCodec (kept for inference)
        del full
        self.cfg = self.decoder.config
        d = self.cfg.hidden_size
        self.num_codebooks = self.cfg.num_codebooks

        for p in self.decoder.parameters():
            p.requires_grad_(False)
        for p in self.audio_encoder.parameters():
            p.requires_grad_(False)

        # LoRA on cross-attention projections only (paper: text-fusion via LoRA'd
        # cross-attn; audio fusion adds projected condition tokens to the memory)
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                          target_modules=r".*encoder_attn\.(q_proj|k_proj|v_proj|out_proj)",
                          bias="none")
        self.decoder = get_peft_model(self.decoder, lcfg)

        self.h_norm = nn.LayerNorm(D_LLM)
        self.proj_h = nn.Sequential(nn.Linear(D_LLM, d), nn.GELU(), nn.Linear(d, d))
        self.proj_src = nn.Linear(d, d)
        nn.init.zeros_(self.proj_src.weight)   # start as pure bias -> stable warmup
        self.mem_norm = nn.LayerNorm(d)

    # ── conditioning memory ─────────────────────────────────────
    def _embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """[B, K, T] codes -> [B, T, d] via the frozen decoder codebook embeddings."""
        base = self.decoder.get_base_model() if hasattr(self.decoder, "get_base_model") \
            else self.decoder
        emb_list = base.model.decoder.embed_tokens
        x = sum(emb_list[k](codes[:, k].clamp(0, SPECIAL)) for k in range(self.num_codebooks))
        return x.float()  # projectors are fp32

    def build_memory(self, h: torch.Tensor, src_codes: torch.Tensor,
                     drop_edit: torch.Tensor | None = None) -> torch.Tensor:
        """h [B,9,3584] fp32, src_codes [B,K,T] -> memory [B, 9+T, d]."""
        hh = self.proj_h(self.h_norm(h.float()))
        if drop_edit is not None:
            hh = hh * (~drop_edit)[:, None, None]
        src = self.proj_src(self._embed_codes(src_codes))
        mem = self.mem_norm(torch.cat([hh, src], dim=1))
        # cross-attn runs in the (frozen) decoder's dtype; grads flow through the cast
        return mem.to(next(iter(self.decoder.parameters())).dtype)

    # ── training ────────────────────────────────────────────────
    def forward(self, h, src_codes, tgt_codes, n_frames=None,
                edit_drop_prob: float = 0.1):
        """CE on delayed target codes. tgt_codes [B,K,T] int64."""
        b, k, t = tgt_codes.shape
        dev = tgt_codes.device
        drop = (torch.rand(b, device=dev) < edit_drop_prob) if self.training else None
        mem = self.build_memory(h, src_codes, drop_edit=drop)

        delayed = delay_codes(tgt_codes)
        inp = torch.cat([torch.full((b, k, 1), SPECIAL, device=dev, dtype=tgt_codes.dtype),
                         delayed[:, :, :-1]], dim=-1)
        labels = delayed.permute(0, 2, 1).contiguous()  # [B, T, K]; 2048 -> -100 inside
        if n_frames is not None:                        # mask padded tail
            pos = torch.arange(t, device=dev)[None, :]
            labels = labels.masked_fill((pos >= n_frames[:, None])[..., None], SPECIAL)

        out = self.decoder(input_ids=inp.reshape(b * k, t),
                           encoder_hidden_states=mem,
                           labels=labels)
        return out.loss

    # ── inference ───────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, h, src_codes, max_frames: int = 500, guidance: float = 3.0,
                 temperature: float = 1.0, top_k: int = 250, seed: int | None = None):
        """Sample delayed codes autoregressively with CFG; returns [1,K,T'] codes."""
        if seed is not None:
            torch.manual_seed(seed)
        dev = src_codes.device
        k = self.num_codebooks
        mem_c = self.build_memory(h, src_codes)
        mem_n = self.build_memory(torch.zeros_like(h), src_codes)
        mem = torch.cat([mem_c, mem_n], 0) if guidance != 1.0 else mem_c

        seq = torch.full((1, k, 1), SPECIAL, device=dev, dtype=torch.long)  # BOS step
        past = None
        for step in range(max_frames + k - 1):
            bsz = mem.shape[0]
            inp = seq[:, :, -1:].expand(bsz, k, 1) if past is not None \
                else seq.expand(bsz, k, seq.shape[-1])
            out = self.decoder(input_ids=inp.reshape(bsz * k, -1),
                               encoder_hidden_states=mem,
                               past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1]                       # [bsz*k, V] -> view
            logits = logits.view(bsz, k, -1)
            if guidance != 1.0:
                lc, ln = logits[:1], logits[1:]
                logits = ln + (lc - ln) * guidance
            logits = logits / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs.view(k, -1), 1).view(1, k, 1)
            # enforce delay: codebook i emits BOS for the first i steps
            for i in range(k):
                if step < i:
                    nxt[0, i] = SPECIAL
            seq = torch.cat([seq, nxt], dim=-1)
        delayed = seq[:, :, 1:]                              # drop BOS column
        codes = undelay_codes(delayed)
        return codes.clamp(0, 2047)

    @torch.no_grad()
    def decode_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """[1,K,T] codes -> [1, samples] float wav @32 kHz."""
        wav = self.audio_encoder.decode(codes.unsqueeze(0), audio_scales=[None])
        return wav.audio_values[0]

    # ── persistence ─────────────────────────────────────────────
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_adapter(self, save_dir: str | Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.decoder.save_pretrained(save_dir / "lora")
        extra = {n: m.state_dict() for n, m in
                 [("h_norm", self.h_norm), ("proj_h", self.proj_h),
                  ("proj_src", self.proj_src), ("mem_norm", self.mem_norm)]}
        torch.save(extra, save_dir / "projectors.pt")

    def load_adapter(self, load_dir: str | Path, device="cuda"):
        from peft import PeftModel
        load_dir = Path(load_dir)
        base = self.decoder.get_base_model()
        self.decoder = PeftModel.from_pretrained(base, load_dir / "lora",
                                                 is_trainable=True)
        extra = torch.load(load_dir / "projectors.pt", map_location=device)
        self.h_norm.load_state_dict(extra["h_norm"])
        self.proj_h.load_state_dict(extra["proj_h"])
        self.proj_src.load_state_dict(extra["proj_src"])
        self.mem_norm.load_state_dict(extra["mem_norm"])
        return self


def build_musicgen_bridge(device: torch.device, dtype=torch.bfloat16) -> MusicGenEditBridge:
    m = MusicGenEditBridge()
    m.decoder.to(device, dtype=dtype)
    m.audio_encoder.to(device)                 # keep EnCodec fp32 (quality)
    m.h_norm.to(device).float(); m.proj_h.to(device).float()
    m.proj_src.to(device).float(); m.mem_norm.to(device).float()
    return m
