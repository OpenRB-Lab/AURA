"""EditCondCFM: [EDIT] hidden states -> frozen DiffRhythm, ControlNet-style.

Per the final design:
- DiffRhythm checkpoint is FULLY FROZEN (no LoRA).
- Trainable parts only: a projector (pooled hidden -> 512 style vector) and one
  zero-initialized cross-attention adapter AFTER EVERY DiT layer (all 16), consuming
  the 9 conditioning states ([EDIT_<KIND>] + [EDIT_0..7], 3584-d each).

Reuses the proven modules from src/image_cond (projector MLP, zero-init cross-attn)
with d_img=3584, and the same monkey-patch/save/load conventions.
"""

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "DiffRhythm"))

from src.image_cond.image_cross_attn import ImageConditioningAdapter  # noqa: E402
from src.image_cond.projector import ImageToMuLanProjector  # noqa: E402
from src.image_cond.lora_dit import load_base_cfm  # noqa: E402

from edit_agent.edit_cfm import EditCFM  # noqa: E402

D_LLM = 3584
N_COND_TOKENS = 9  # [EDIT_<KIND>] + 8 payload tokens


class EditCondCFM(nn.Module):
    def __init__(self, cfm: EditCFM, projector: nn.Module, cross_attn: nn.Module):
        super().__init__()
        self.cfm = cfm
        self.projector = projector
        self.cross_attn = cross_attn
        self.h_norm = nn.LayerNorm(D_LLM)  # tame fp16-scale hidden states
        self._patch_dit_forward()

    # ── DiT patch: adapter after every block ────────────────────
    def _get_dit(self):
        return self.cfm.transformer

    def _patch_dit_forward(self):
        dit = self._get_dit()
        dit._edit_tokens = None
        dit._edit_adapter = self.cross_attn
        # Rather than reimplement DiT.forward, wrap each block so the adapter runs
        # after EVERY layer output — robust to upstream forward changes.
        for i, block in enumerate(dit.transformer_blocks):
            orig_block_forward = block.forward

            def make_wrapped(bf, layer_idx):
                def wrapped(*args, **kwargs):
                    out = bf(*args, **kwargs)
                    x = out[0] if isinstance(out, tuple) else out
                    tokens = dit._edit_tokens
                    if tokens is not None:
                        # adapter is fp32 (fp16 AdamW NaNs: eps underflow + 0-grad
                        # q/k/v at step 1 behind the zero-init out_proj)
                        x = dit._edit_adapter.forward_layer(
                            layer_idx, x.float(), tokens).to(x.dtype)
                    return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
                return wrapped

            block.forward = make_wrapped(orig_block_forward, i)

    def set_tokens(self, tokens: torch.Tensor | None):
        self._get_dit()._edit_tokens = tokens

    # ── training ────────────────────────────────────────────────
    def forward(self, src_latent, tgt_latent, edit_hidden, mulan_tgt, lens,
                start_time, pred_mask=None):
        """edit_hidden: [B, 9, 3584] fp32; latents [B,T,64] half."""
        h = self.h_norm(edit_hidden.float())
        style = self.projector(h.mean(dim=1))                    # [B, 512]

        b = src_latent.shape[0]
        drop_edit = torch.rand(b, device=src_latent.device) < self.cfm.edit_cond_drop_prob
        tokens = h.clone()
        tokens[drop_edit] = 0.0
        self.set_tokens(tokens)

        flow_loss, _ = self.cfm.edit_forward(
            src_latent, tgt_latent, style.half(), lens, start_time,
            pred_mask=pred_mask, drop_edit=drop_edit)
        self.set_tokens(None)

        style_n = torch.nn.functional.normalize(style, dim=-1)
        tgt_n = torch.nn.functional.normalize(mulan_tgt.float(), dim=-1)
        align = (1 - (style_n * tgt_n).sum(-1)).mean()

        nce = torch.tensor(0.0, device=style.device)
        if b > 1:
            logits = style_n @ tgt_n.t() / 0.07
            labels = torch.arange(b, device=style.device)
            nce = 0.5 * (torch.nn.functional.cross_entropy(logits, labels)
                         + torch.nn.functional.cross_entropy(logits.t(), labels))
        return flow_loss, align, nce

    # ── inference ───────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, src_latent, edit_hidden, negative_style_prompt, n_frames,
               segment=None, steps=32, cfg_strength=2.0, seed=None):
        h = self.h_norm(edit_hidden.float())
        style = self.projector(h.mean(dim=1)).half()
        tokens = h
        return self.cfm.sample_edit(
            src_latent, style, negative_style_prompt, n_frames, segment=segment,
            steps=steps, cfg_strength=cfg_strength, seed=seed,
            set_tokens=lambda: self.set_tokens(tokens),
            clear_tokens=lambda: self.set_tokens(None))


def build_bridge(device: torch.device, dit_config: str | None = None,
                 pretrained_cache: str | None = None) -> EditCondCFM:
    dit_config = dit_config or str(PROJECT_ROOT / "src/DiffRhythm/config/diffrhythm-1b.json")
    pretrained_cache = pretrained_cache or str(PROJECT_ROOT / "weights")
    base = load_base_cfm(dit_config, device, 2048, pretrained_cache)
    # rebuild as EditCFM sharing the loaded transformer/weights
    cfm = EditCFM(transformer=base.transformer, num_channels=64, max_frames=2048).to(device)
    cfm.load_state_dict(base.state_dict(), strict=False)
    cfm.half()
    for p in cfm.parameters():                # FROZEN DiffRhythm
        p.requires_grad_(False)
    cfm.eval()

    projector = ImageToMuLanProjector(d_img=D_LLM, d_text=512, d_mulan=512,
                                      hidden_dim=1024, num_layers=2,
                                      dropout=0.1, l2_normalize=True).to(device).float()
    cross_attn = ImageConditioningAdapter(
        dit_dim=2048, d_img=D_LLM, num_heads=16, num_layers=16,
        total_dit_layers=16, dropout=0.1).to(device).float()  # after EVERY layer; fp32
    model = EditCondCFM(cfm, projector, cross_attn).to(device)
    return model


def save_adapter(model: EditCondCFM, save_dir: str | Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.projector.state_dict(), save_dir / "projector.pt")
    torch.save(model.cross_attn.state_dict(), save_dir / "edit_cross_attn.pt")
    torch.save(model.h_norm.state_dict(), save_dir / "h_norm.pt")


def load_adapter(model: EditCondCFM, load_dir: str | Path, device) -> EditCondCFM:
    load_dir = Path(load_dir)
    model.projector.load_state_dict(torch.load(load_dir / "projector.pt", map_location=device))
    model.cross_attn.load_state_dict(torch.load(load_dir / "edit_cross_attn.pt", map_location=device))
    model.h_norm.load_state_dict(torch.load(load_dir / "h_norm.pt", map_location=device))
    return model
