"""MusicGen edit bridge v3 — dual-stream fusion (user-specified architecture).

Per decoder layer (48, all base weights FROZEN, reusing the layer's own
self-attn projections for every new attention):
  music stream : O_m = masked self-attn(LN(z_m))                (stock, causal)
  cond stream  : z_c <- z_c + full self-attn(LN(z_c))           (music-free ->
                 K_c/V_c/Q_c precomputable at inference, full source visibility)
  bi-dual attn : Q = Q_cond + Q_music (position-aligned, causal-safe)
                 s1 = MHA(Q, K_cond, V_cond)   full
                 s2 = MHA(Q, K_music, V_music) causal
                 s_fuse = a1*s1 + a2*s2        (learnable per-layer scalars)
  FiLM         : [gamma; beta] = W2 GELU(W1 s_fuse)  (low-rank, zero-init)
                 O_m <- O_m * (1 + tanh(g)*gamma) + tanh(g)*beta   (gate g init 0
                 -> exact identity to frozen MusicGen at init)
  then frozen residual -> LN -> cross-attn(K,V = proj_h(edit tokens), LoRA r64
  on k_proj/v_proj ONLY) -> frozen FFN.

Trainable: FiLM MLPs + alphas + gates + proj_h/h_norm + cross-attn k/v LoRA.
"""

import math
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edit_agent.musicgen_bridge import SPECIAL, delay_codes, undelay_codes  # noqa: E402

D_LLM = 3584


def _mha(q, k, v, num_heads, causal=False):
    """q,k,v: [B, T*, d] pre-projected. Returns [B, Tq, d] (no out_proj)."""
    b, tq, d = q.shape
    tk = k.shape[1]
    hd = d // num_heads
    q = q.view(b, tq, num_heads, hd).transpose(1, 2) * (hd ** -0.5)
    k = k.view(b, tk, num_heads, hd).transpose(1, 2)
    v = v.view(b, tk, num_heads, hd).transpose(1, 2)
    scores = q @ k.transpose(-2, -1)
    if causal:
        mask = torch.triu(torch.full((tq, tk), float("-inf"), device=q.device), 1 + tk - tq)
        scores = scores + mask
    attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    return (attn @ v).transpose(1, 2).reshape(b, tq, d)


class MusicGenFusionBridge(nn.Module):
    def __init__(self, model_id: str = "facebook/musicgen-medium",
                 cache_dir: str | None = None, lora_r: int = 64,
                 lora_alpha: int = 128, film_rank: int = 256):
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
        self.num_heads = self.cfg.num_attention_heads
        self.num_codebooks = self.cfg.num_codebooks
        self.n_layers = self.cfg.num_hidden_layers

        for p in self.decoder.parameters():
            p.requires_grad_(False)
        for p in self.audio_encoder.parameters():
            p.requires_grad_(False)

        # LoRA ONLY on cross-attn k/v (per user spec)
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                          target_modules=r".*encoder_attn\.(k_proj|v_proj)",
                          bias="none")
        self.decoder = get_peft_model(self.decoder, lcfg)

        # edit-token memory
        self.h_norm = nn.LayerNorm(D_LLM)
        self.proj_h = nn.Sequential(nn.Linear(D_LLM, d), nn.GELU(), nn.Linear(d, d))

        # fusion params (fp32)
        self.film = nn.ModuleList([
            nn.Sequential(nn.Linear(d, film_rank), nn.GELU(),
                          nn.Linear(film_rank, 2 * d))
            for _ in range(self.n_layers)])
        for m in self.film:
            nn.init.zeros_(m[2].weight)
            nn.init.zeros_(m[2].bias)
        self.alpha1 = nn.Parameter(torch.full((self.n_layers,), 0.5))
        self.alpha2 = nn.Parameter(torch.full((self.n_layers,), 0.5))
        # gate init 1.0 (NOT 0): with the FiLM output zero-init, identity at init
        # is already guaranteed, and g=0 would deadlock gradients (d/dg ∝ γ=0 and
        # d/dγ ∝ g=0). tanh(1)≈0.76 lets the FiLM weights receive gradient.
        self.gate = nn.Parameter(torch.ones(self.n_layers))
        self.ssm_weight = 0.0  # structural consistency loss (set by trainer)
        self.latent_weight = 0.0  # hybrid CE + L2-latent loss (set by trainer)
        self._rvq_embed = None  # lazy [K, bins, 128] EnCodec codebook buffer

    # ── plumbing ────────────────────────────────────────────────
    def _base(self):
        return self.decoder.get_base_model() if hasattr(self.decoder, "get_base_model") \
            else self.decoder

    def _dec(self):
        return self._base().model.decoder

    def _dtype(self):
        return next(iter(self.decoder.parameters())).dtype

    def _embed(self, codes):
        """delayed codes [B,K,T] -> [B,T,d] token+pos embeddings."""
        dec = self._dec()
        x = sum(dec.embed_tokens[k](codes[:, k].clamp(0, SPECIAL))
                for k in range(self.num_codebooks))
        pos = dec.embed_positions(codes, 0).to(x.dtype)
        return x + pos.unsqueeze(0)

    def edit_memory(self, h, drop_edit=None):
        hh = self.proj_h(self.h_norm(h.float()))
        if drop_edit is not None:
            hh = hh * (~drop_edit)[:, None, None]
        return hh.to(self._dtype())

    def _film(self, l, o_m, s_fuse):
        gb = self.film[l](s_fuse.float())
        gamma, beta = gb.chunk(2, dim=-1)
        g = torch.tanh(self.gate[l])
        out = o_m.float() * (1.0 + g * gamma) + g * beta
        return out.to(o_m.dtype)

    def _cross_ffn(self, layer, z, mem):
        res = z
        x = layer.encoder_attn_layer_norm(z)
        ca = layer.encoder_attn
        o = _mha(ca.q_proj(x), ca.k_proj(mem), ca.v_proj(mem), self.num_heads)
        z = res + ca.out_proj(o)
        res = z
        x = layer.final_layer_norm(z)
        z = res + layer.fc2(F.gelu(layer.fc1(x)))
        return z

    def _rvq_codebooks(self):
        """[K, bins, 128] frozen EnCodec RVQ codebook embeddings."""
        if self._rvq_embed is None:
            layers = self.audio_encoder.quantizer.layers[: self.num_codebooks]
            self._rvq_embed = torch.stack(
                [l.codebook.embed.detach() for l in layers]).float()
        return self._rvq_embed

    # ── training forward ────────────────────────────────────────
    def forward(self, h, src_codes, tgt_codes, n_frames=None,
                edit_drop_prob: float = 0.1, z_tgt=None):
        b, k, t = tgt_codes.shape
        dev = tgt_codes.device
        drop = (torch.rand(b, device=dev) < edit_drop_prob) if self.training else None
        mem = self.edit_memory(h, drop_edit=drop)

        d_src = delay_codes(src_codes)
        d_tgt = delay_codes(tgt_codes)
        inp = torch.cat([torch.full((b, k, 1), SPECIAL, device=dev,
                                    dtype=tgt_codes.dtype), d_tgt[:, :, :-1]], -1)
        z_m = self._embed(inp)
        z_c = self._embed(d_src)

        layers = self._dec().layers
        for l, layer in enumerate(layers):
            sa = layer.self_attn
            # music masked self-attn (frozen weights, own math)
            res = z_m
            x = layer.self_attn_layer_norm(z_m)
            qm, km, vm = sa.q_proj(x), sa.k_proj(x), sa.v_proj(x)
            o_m = sa.out_proj(_mha(qm, km, vm, self.num_heads, causal=True))
            # cond features (music-free)
            xc = layer.self_attn_layer_norm(z_c)
            qc, kc, vc = sa.q_proj(xc), sa.k_proj(xc), sa.v_proj(xc)
            # bi-dual attention, shared position-aligned query
            q = qc + qm
            s1 = _mha(q, kc, vc, self.num_heads, causal=False)
            s2 = _mha(q, km, vm, self.num_heads, causal=True)
            s_fuse = self.alpha1[l] * s1 + self.alpha2[l] * s2
            o_m = self._film(l, o_m, s_fuse)
            z_m = res + o_m
            z_m = self._cross_ffn(layer, z_m, mem)
            # cond stream update (frozen full self-attn, music-free)
            z_c = z_c + sa.out_proj(_mha(qc, kc, vc, self.num_heads, causal=False))

        z_m = self._dec().layer_norm(z_m)
        heads = self._base().lm_heads
        loss = torch.zeros((), device=dev)
        labels = d_tgt.permute(0, 2, 1)                      # [B,T,K]
        if n_frames is not None:
            pos = torch.arange(t, device=dev)[None, :]
            labels = labels.masked_fill((pos >= n_frames[:, None])[..., None], SPECIAL)
        labels = labels.masked_fill(labels == SPECIAL, -100)
        logit_list = []
        for kk in range(self.num_codebooks):
            logits = heads[kk](z_m)
            logit_list.append(logits)
            loss = loss + F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                          labels[..., kk].reshape(-1),
                                          ignore_index=-100)
        loss = loss / self.num_codebooks

        if self.latent_weight > 0 and z_tgt is not None:
            # hybrid L2 in EnCodec latent space: softmax-expected RVQ embedding
            # per codebook, undelayed (codebook k at delayed pos f+k is true
            # frame f) and summed over codebooks vs the pre-RVQ target latent
            W = self._rvq_codebooks().to(z_tgt.device)      # [K,bins,128]
            K = self.num_codebooks
            tprime = t - (K - 1)
            z_hat = 0.0
            for kk in range(K):
                p = F.softmax(logit_list[kk][:, :, :W.shape[1]].float(), dim=-1)
                z_hat = z_hat + p[:, kk: kk + tprime] @ W[kk]  # [B,T',128]
            zt = z_tgt.float().permute(0, 2, 1)[:, :tprime]    # [B,T',128]
            if n_frames is not None:
                lm = (torch.arange(tprime, device=dev)[None, :]
                      < torch.clamp(n_frames, max=tprime)[:, None]).float()
            else:
                lm = torch.ones(b, tprime, device=dev)
            l2 = (((z_hat - zt) ** 2).mean(-1) * lm).sum() / lm.sum().clamp(min=1)
            loss = loss + self.latent_weight * l2

        if self.ssm_weight > 0:
            # structural consistency: match the self-similarity matrix of the
            # decoder features to that of the ground-truth code embeddings, so
            # repeated material in the target stays repeated in the prediction
            dec = self._dec()
            with torch.no_grad():
                f_tgt = sum(dec.embed_tokens[kk](d_tgt[:, kk].clamp(0, SPECIAL))
                            for kk in range(self.num_codebooks)).float()
            f_gen = z_m.float()
            f_gen = F.normalize(f_gen, dim=-1)
            f_tgt = F.normalize(f_tgt, dim=-1)
            s_gen = f_gen @ f_gen.transpose(1, 2)            # [B,T,T]
            s_tgt = f_tgt @ f_tgt.transpose(1, 2)
            if n_frames is not None:
                valid = (torch.arange(t, device=dev)[None, :] < n_frames[:, None])
            else:
                valid = torch.ones(b, t, dtype=torch.bool, device=dev)
            m2 = (valid[:, :, None] & valid[:, None, :]).float()
            ssm = ((s_gen - s_tgt) ** 2 * m2).sum() / m2.sum().clamp(min=1)
            loss = loss + self.ssm_weight * ssm
        return loss

    # ── inference ───────────────────────────────────────────────
    @torch.no_grad()
    def _precompute_cond(self, src_codes):
        """Run the cond stream once; per-layer (q_c, k_c, v_c) for the bi-dual attn."""
        z_c = self._embed(delay_codes(src_codes))
        feats = []
        for layer in self._dec().layers:
            sa = layer.self_attn
            xc = layer.self_attn_layer_norm(z_c)
            qc, kc, vc = sa.q_proj(xc), sa.k_proj(xc), sa.v_proj(xc)
            feats.append((qc, kc, vc))
            z_c = z_c + sa.out_proj(_mha(qc, kc, vc, self.num_heads, causal=False))
        return feats

    @torch.no_grad()
    def generate(self, h, src_codes, max_frames: int = 500, guidance: float = 2.0,
                 temperature: float = 1.0, top_k: int = 250, seed: int | None = None,
                 segment=None):
        if seed is not None:
            torch.manual_seed(seed)
        dev = src_codes.device
        k = self.num_codebooks
        bsz = 2 if guidance != 1.0 else 1
        mem_c = self.edit_memory(h)
        mem = torch.cat([mem_c, torch.zeros_like(mem_c)], 0) if bsz == 2 else mem_c
        feats = self._precompute_cond(src_codes)
        if bsz == 2:
            feats = [(qc.expand(2, -1, -1), kc.expand(2, -1, -1), vc.expand(2, -1, -1))
                     for qc, kc, vc in feats]
        d_src = delay_codes(src_codes)
        d_src_pad = torch.cat([d_src, torch.full((1, k, max_frames + k), SPECIAL,
                                                 device=dev, dtype=torch.long)], -1)

        kv_cache = [[None, None] for _ in range(self.n_layers)]  # per-layer K,V music
        dec = self._dec()
        prev = torch.full((1, k, 1), SPECIAL, device=dev, dtype=torch.long)
        seq = []
        for step in range(max_frames + k - 1):
            x = sum(dec.embed_tokens[cb](prev[:, cb].clamp(0, SPECIAL)) for cb in range(k))
            pos = dec.embed_positions(prev, step).to(x.dtype)
            z = (x + pos.unsqueeze(0)).expand(bsz, -1, -1)
            for l, layer in enumerate(dec.layers):
                sa = layer.self_attn
                res = z
                xn = layer.self_attn_layer_norm(z)
                qm, km, vm = sa.q_proj(xn), sa.k_proj(xn), sa.v_proj(xn)
                if kv_cache[l][0] is None:
                    kv_cache[l] = [km, vm]
                else:
                    kv_cache[l][0] = torch.cat([kv_cache[l][0], km], 1)
                    kv_cache[l][1] = torch.cat([kv_cache[l][1], vm], 1)
                K, V = kv_cache[l]
                o_m = sa.out_proj(_mha(qm, K, V, self.num_heads))
                qc, kc, vc = feats[l]
                if step < qc.shape[1]:
                    q = qc[:, step:step + 1] + qm
                else:
                    q = qm
                s1 = _mha(q, kc, vc, self.num_heads)
                s2 = _mha(q, K, V, self.num_heads)
                s_fuse = self.alpha1[l] * s1 + self.alpha2[l] * s2
                o_m = self._film(l, o_m, s_fuse)
                z = res + o_m
                z = self._cross_ffn(layer, z, mem)
            z = dec.layer_norm(z)
            heads = self._base().lm_heads
            logits = torch.stack([heads[cb](z[:, -1]) for cb in range(k)], 1)  # [bsz,K,V]
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
                    fr = step - i
                    if fr < segment[0] or fr >= segment[1]:
                        nxt[0, i] = d_src_pad[0, i, step]
            seq.append(nxt)
            prev = nxt
        delayed = torch.cat(seq, -1)
        return undelay_codes(delayed).clamp(0, 2047)

    @torch.no_grad()
    def decode_audio(self, codes):
        wav = self.audio_encoder.decode(codes.unsqueeze(0), audio_scales=[None])
        return wav.audio_values[0]

    @torch.no_grad()
    def decode_audio_smooth(self, gen_codes, src_codes, segment, fade_s=0.15,
                            frame_rate=50, sr=32000):
        """Seam-free decode for anchored localized edits: raised-cosine
        crossfade between decoded source (outside the segment) and decoded
        generation (inside) at the segment boundaries."""
        y_gen = self.decode_audio(gen_codes).squeeze(0)
        y_src = self.decode_audio(src_codes).squeeze(0)
        n = min(y_gen.shape[-1], y_src.shape[-1])
        y_gen, y_src = y_gen[..., :n], y_src[..., :n]
        s0 = int(segment[0] / frame_rate * sr)
        s1 = int(segment[1] / frame_rate * sr)
        f = int(fade_s * sr)
        w = torch.zeros(n, device=y_gen.device)
        w[s0:s1] = 1.0
        ramp = 0.5 - 0.5 * torch.cos(
            torch.linspace(0, torch.pi, max(f, 2), device=y_gen.device))
        a, bnd = max(s0 - f // 2, 0), min(s0 + f // 2, n)
        if bnd > a and s0 > 0:
            w[a:bnd] = ramp[: bnd - a]
        a, bnd = max(s1 - f // 2, 0), min(s1 + f // 2, n)
        if bnd > a and s1 < n:
            w[a:bnd] = ramp.flip(0)[: bnd - a]
        return (w * y_gen + (1 - w) * y_src).unsqueeze(0)

    # ── persistence ─────────────────────────────────────────────
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_adapter(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.decoder.save_pretrained(save_dir / "lora")
        torch.save({"h_norm": self.h_norm.state_dict(),
                    "proj_h": self.proj_h.state_dict(),
                    "film": self.film.state_dict(),
                    "alpha1": self.alpha1.data, "alpha2": self.alpha2.data,
                    "gate": self.gate.data}, save_dir / "projectors.pt")

    def load_adapter(self, load_dir, device="cuda"):
        from peft import PeftModel
        load_dir = Path(load_dir)
        self.decoder = PeftModel.from_pretrained(self.decoder.get_base_model(),
                                                 load_dir / "lora", is_trainable=True)
        extra = torch.load(load_dir / "projectors.pt", map_location=device)
        self.h_norm.load_state_dict(extra["h_norm"])
        self.proj_h.load_state_dict(extra["proj_h"])
        self.film.load_state_dict(extra["film"])
        self.alpha1.data = extra["alpha1"].to(device)
        self.alpha2.data = extra["alpha2"].to(device)
        self.gate.data = extra["gate"].to(device)
        return self


def build_fusion_bridge(device: torch.device, dtype=torch.bfloat16,
                        lora_r: int = 64, lora_alpha: int = 128) -> MusicGenFusionBridge:
    m = MusicGenFusionBridge(lora_r=lora_r, lora_alpha=lora_alpha)
    m.decoder.to(device, dtype=dtype)
    m.audio_encoder.to(device)
    m.h_norm.to(device).float(); m.proj_h.to(device).float()
    m.film.to(device).float()
    m.alpha1.data = m.alpha1.data.to(device)
    m.alpha2.data = m.alpha2.data.to(device)
    m.gate.data = m.gate.data.to(device)
    return m
