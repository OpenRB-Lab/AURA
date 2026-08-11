"""EditCFM: DiffRhythm CFM specialized for source-conditioned edit training.

Deviations from stock CFM.forward (src/DiffRhythm/model/cfm.py):
- cond = the SOURCE latent (never zeroed): the model learns src + edit semantics -> tgt.
- The flow-matching loss span is the full sequence for global edits, or the edit
  segment only for localized (inpaint-tier) pairs via `pred_mask`.
- CFG drops: edit conditioning (style + cross-attn tokens) p=0.1; source cond p=0.1.

Sampling keeps cond = src in BOTH CFG branches so guidance steers the edit itself.
"""

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "DiffRhythm"))

from model.cfm import CFM  # noqa: E402


def lens_to_mask(lens: torch.Tensor, max_len: int) -> torch.Tensor:
    seq = torch.arange(max_len, device=lens.device)
    return seq[None, :] < lens[:, None]


class EditCFM(CFM):
    edit_cond_drop_prob: float = 0.1
    src_cond_drop_prob: float = 0.1

    def edit_forward(self, src_latent, tgt_latent, style_prompt, lens,
                     start_time, pred_mask=None, drop_edit=None, drop_src=None):
        """Flow-matching loss for an edit pair. Shapes: latents [B,T,64]."""
        b, t, _ = tgt_latent.shape
        device = tgt_latent.device
        mask = lens_to_mask(lens, t)                       # [B, T] valid frames

        x1 = tgt_latent
        x0 = torch.randn_like(x1)
        time = torch.normal(0, 1, (b,), device=device).sigmoid()
        time = time.to(x1.dtype)
        tt = time[:, None, None]
        phi = (1 - tt) * x0 + tt * x1
        flow = x1 - x0

        cond = src_latent.clone()
        if drop_src is None:
            drop_src = torch.rand(b, device=device) < self.src_cond_drop_prob
        cond[drop_src] = 0.0

        if drop_edit is None:
            drop_edit = torch.rand(b, device=device) < self.edit_cond_drop_prob
        style = style_prompt.clone()
        style[drop_edit] = 0.0

        text = torch.zeros(b, t, dtype=torch.long, device=device)  # instrumental
        pred = self.transformer(
            x=phi, cond=cond, text=text, time=time,
            drop_audio_cond=False, drop_text=True, drop_prompt=False,
            style_prompt=style, start_time=start_time,
        )

        loss_mask = mask if pred_mask is None else (mask & pred_mask)
        loss = torch.nn.functional.mse_loss(pred.float(), flow.float(), reduction="none")
        if loss_mask.any():
            loss = loss[loss_mask].mean()
        else:  # e.g. segment entirely beyond the truncated latent
            loss = (pred.float() * 0).sum()
        return loss, drop_edit

    @torch.no_grad()
    def sample_edit(self, src_latent, style_prompt, negative_style_prompt,
                    n_frames: int, segment=None, steps: int = 32,
                    cfg_strength: float = 2.0, seed: int | None = None,
                    set_tokens=None, clear_tokens=None):
        """ODE sampling for one edit. src_latent [1,T,64] padded to max_frames.

        set_tokens/clear_tokens: callables to install/remove cross-attn edit tokens on
        the DiT for the conditional branch only (CFG null branch runs without them).
        """
        device = src_latent.device
        max_frames = src_latent.shape[1]
        if seed is not None:
            torch.manual_seed(seed)

        if segment is not None:
            s, e = segment
            segs = [(max(0, s), min(n_frames, max(e, s + 8)))]
        else:
            segs = [(0, n_frames)]

        text = torch.zeros(1, max_frames, dtype=torch.long, device=device)
        start_time = torch.zeros(1, device=device, dtype=src_latent.dtype)
        norm_dur = torch.tensor([n_frames / max_frames], device=device,
                                dtype=src_latent.dtype)

        # differs from stock sample(): keep cond = src inside the pred span too, and
        # run CFG manually so the null branch drops the edit tokens
        seq = torch.arange(max_frames, device=device)
        span_mask = ((seq >= segs[0][0]) & (seq < segs[0][1]))[None, :, None]  # [1,T,1]

        y = torch.randn_like(src_latent)
        ts = torch.linspace(0, 1, steps + 1, device=device, dtype=src_latent.dtype)
        for i in range(steps):
            t_cur, t_next = ts[i], ts[i + 1]
            time = t_cur.repeat(1)
            if set_tokens is not None:
                set_tokens()
            v_cond = self.transformer(
                x=y, cond=src_latent, text=text, time=time,
                drop_audio_cond=False, drop_text=True, drop_prompt=False,
                style_prompt=style_prompt, start_time=start_time)
            if clear_tokens is not None:
                clear_tokens()
            v_null = self.transformer(
                x=y, cond=src_latent, text=text, time=time,
                drop_audio_cond=False, drop_text=True, drop_prompt=True,
                style_prompt=negative_style_prompt, start_time=start_time)
            v = v_null + (v_cond - v_null) * cfg_strength
            y = y + v * (t_next - t_cur)
            # keep non-edited region locked to a noised interpolation of the source
            anchor = (1 - t_next) * torch.randn_like(src_latent) + t_next * src_latent
            y = torch.where(span_mask, y, anchor)
        # final: outside the span, exact source
        y = torch.where(span_mask, y, src_latent)
        return y
