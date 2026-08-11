"""Freeze base DiT, inject LoRA adapters + image cross-attention, swap style conditioning."""

import json
import sys
from pathlib import Path
from functools import partial

import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
from peft import LoraConfig, get_peft_model

DIFFRHYTHM_ROOT = Path(__file__).resolve().parents[2] / "src" / "DiffRhythm"
sys.path.insert(0, str(DIFFRHYTHM_ROOT))

from model import DiT, CFM
from infer.infer_utils import load_checkpoint
from model.modules import _prepare_decoder_attention_mask


class ImageCondCFM(nn.Module):
    """Wraps CFM with image-conditioned style injection via projector + cross-attention."""

    def __init__(self, cfm: CFM, projector: nn.Module, img_cross_attn=None):
        super().__init__()
        self.cfm = cfm
        self.projector = projector
        self.img_cross_attn = img_cross_attn

        if img_cross_attn is not None:
            self._patch_dit_forward()

    def _patch_dit_forward(self):
        """Replace the DiT forward to inject image cross-attention after self-attention blocks."""
        dit = self._get_dit()
        dit._original_forward = dit.forward
        dit._img_cross_attn = self.img_cross_attn
        dit._img_tokens = None  # set before each forward call

        def patched_forward(self_dit, x, cond, text, time, drop_audio_cond, drop_text,
                            drop_prompt=False, style_prompt=None, start_time=None, duration=None):
            batch, seq_len = x.shape[0], x.shape[1]
            if time.ndim == 0:
                time = time.repeat(batch)

            t = self_dit.time_embed(time)
            s_t = self_dit.start_time_embed(start_time)
            d_t = self_dit.duration_time_embed(duration) if self_dit.max_frames == 6144 else torch.zeros_like(s_t)
            c = t + s_t + d_t
            text_embed = self_dit.text_embed(text, seq_len, drop_text=drop_text)

            if drop_prompt:
                style_prompt = torch.zeros_like(style_prompt)

            x = self_dit.input_embed(x, cond, text_embed, style_prompt, c, drop_audio_cond=drop_audio_cond)

            if self_dit.long_skip_connection is not None:
                residual = x

            pos_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)
            rotary_embed = self_dit.rotary_emb(x, pos_ids)
            attention_mask = torch.ones((batch, seq_len), dtype=torch.bool, device=x.device)
            attention_mask = _prepare_decoder_attention_mask(attention_mask, (batch, seq_len), x)

            for i, block in enumerate(self_dit.transformer_blocks):
                out = block(x, attention_mask=attention_mask, position_embeddings=rotary_embed)
                x = out if isinstance(out, torch.Tensor) else out[0]
                if i < self_dit.depth // 2:
                    x = x + self_dit.text_fusion_linears[i](text_embed)
                # Image cross-attention
                if self_dit._img_tokens is not None and self_dit._img_cross_attn is not None:
                    x = self_dit._img_cross_attn.forward_layer(i, x, self_dit._img_tokens)

            if self_dit.long_skip_connection is not None:
                x = self_dit.long_skip_connection(torch.cat((x, residual), dim=-1))

            x = self_dit.norm_out(x, c)
            return self_dit.proj_out(x)

        import types
        dit.forward = types.MethodType(patched_forward, dit)

    def _get_dit(self):
        """Get the underlying DiT, handling PEFT wrapping."""
        transformer = self.cfm.transformer
        if hasattr(transformer, 'base_model'):
            return transformer.base_model.model
        return transformer

    def _set_img_tokens(self, img_tokens):
        dit = self._get_dit()
        if img_tokens is not None:
            dtype = next(dit.parameters()).dtype
            dit._img_tokens = img_tokens.to(dtype)
        else:
            dit._img_tokens = None

    def forward(self, inp, text, z_img, z_text=None, z_img_tokens=None,
                lens=None, start_time=None,
                z_mulan_target=None, align_weight=1.0,
                contrastive_weight=0.5, contrastive_temp=0.07):
        style = self.projector(z_img, z_text)  # [B, 512]

        align_loss = torch.tensor(0.0, device=style.device)
        contrastive_loss = torch.tensor(0.0, device=style.device)

        if z_mulan_target is not None:
            cos_sim = nn.functional.cosine_similarity(style, z_mulan_target, dim=-1)
            align_loss = (1.0 - cos_sim).mean()

            if style.shape[0] > 1:
                style_norm = nn.functional.normalize(style, dim=-1)
                target_norm = nn.functional.normalize(z_mulan_target, dim=-1)
                logits = style_norm @ target_norm.T / contrastive_temp
                labels = torch.arange(style.shape[0], device=style.device)
                loss_s2a = nn.functional.cross_entropy(logits, labels)
                loss_a2s = nn.functional.cross_entropy(logits.T, labels)
                contrastive_loss = (loss_s2a + loss_a2s) / 2

        # Set image patch tokens for cross-attention
        self._set_img_tokens(z_img_tokens)

        dtype = next(self.cfm.transformer.parameters()).dtype
        style = style.to(dtype)
        inp = inp.to(dtype)
        if start_time is not None:
            start_time = start_time.to(dtype)
        fm_loss, cond, pred = self.cfm(
            inp, text=text, style_prompt=style,
            style_prompt_lens=None, lens=lens,
            start_time=start_time,
        )

        self._set_img_tokens(None)

        total_loss = fm_loss + align_weight * align_loss + contrastive_weight * contrastive_loss
        return total_loss, fm_loss, align_loss, contrastive_loss

    @torch.no_grad()
    def sample(self, z_img, z_text=None, z_img_tokens=None, **kwargs):
        style = self.projector(z_img, z_text)
        kwargs["style_prompt"] = style
        self._set_img_tokens(z_img_tokens)
        result = self.cfm.sample(**kwargs)
        self._set_img_tokens(None)
        return result


def load_base_cfm(dit_config_path: str, device: torch.device,
                  max_frames: int = 2048, pretrained_cache: str = "./weights") -> CFM:
    from huggingface_hub import hf_hub_download
    repo_id = "ASLP-lab/DiffRhythm-1_2" if max_frames == 2048 else "ASLP-lab/DiffRhythm-1_2-full"
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="cfm_model.pt",
                                cache_dir=pretrained_cache)

    with open(dit_config_path) as f:
        model_config = json.load(f)

    dit = DiT(**model_config["model"], max_frames=max_frames)
    for block in dit.transformer_blocks:
        block.self_attn.config._attn_implementation = "eager"
    cfm = CFM(transformer=dit, num_channels=model_config["model"]["mel_dim"],
              max_frames=max_frames)
    cfm = cfm.to(device)
    cfm = load_checkpoint(cfm, ckpt_path, device=device, use_ema=False)
    cfm = cfm.float()
    return cfm


def freeze_base(cfm: CFM):
    for p in cfm.parameters():
        p.requires_grad_(False)


def inject_lora(cfm: CFM, lora_cfg: dict) -> CFM:
    config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", [
            "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.o_proj",
        ]),
        bias=lora_cfg.get("bias", "none"),
    )
    cfm.transformer = get_peft_model(cfm.transformer, config)
    return cfm


def build_image_cond_model(cfg: dict, device: torch.device):
    """Build the full image-conditioned model: base CFM + LoRA + projector + cross-attention."""
    from src.image_cond.projector import ImageToMuLanProjector
    from src.image_cond.image_cross_attn import ImageConditioningAdapter

    model_cfg = cfg["model"]
    dit_config_path = model_cfg["dit_config"]
    max_frames = model_cfg.get("max_frames", 2048)
    pretrained_cache = cfg.get("pretrained_cache", "./weights")

    print("Loading base CFM model...")
    cfm = load_base_cfm(dit_config_path, device, max_frames, pretrained_cache)

    print("Freezing base weights...")
    freeze_base(cfm)

    use_lora = cfg.get("lora", {}).get("enabled", True)
    if use_lora:
        print("Injecting LoRA adapters...")
        cfm = inject_lora(cfm, cfg["lora"])
    else:
        print("Skipping LoRA (disabled in config)")

    print("Building projector...")
    proj_cfg = cfg["projector"]
    projector = ImageToMuLanProjector(
        d_img=proj_cfg["d_img"],
        d_text=proj_cfg.get("d_text", 512),
        d_mulan=proj_cfg["d_mulan"],
        hidden_dim=proj_cfg["hidden_dim"],
        num_layers=proj_cfg["num_layers"],
        dropout=proj_cfg["dropout"],
        l2_normalize=proj_cfg["l2_normalize"],
    ).to(device)

    # Build image cross-attention adapter
    img_cross_attn = None
    xattn_cfg = cfg.get("image_cross_attn")
    if xattn_cfg and xattn_cfg.get("enabled", True):
        print("Building image cross-attention adapter...")
        img_cross_attn = ImageConditioningAdapter(
            dit_dim=model_cfg.get("dit_dim", 2048),
            d_img=xattn_cfg.get("d_img", 768),
            num_heads=xattn_cfg.get("num_heads", 16),
            num_layers=xattn_cfg.get("num_cross_attn_layers", 8),
            total_dit_layers=16,
            dropout=xattn_cfg.get("dropout", 0.0),
        ).to(device)

    model = ImageCondCFM(cfm, projector, img_cross_attn)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    return model


def save_adapter(model: ImageCondCFM, save_dir: str):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    has_lora = hasattr(model.cfm.transformer, 'save_pretrained')
    if has_lora:
        model.cfm.transformer.save_pretrained(save_dir / "lora_adapter")
    torch.save(model.projector.state_dict(), save_dir / "projector.pt")
    if model.img_cross_attn is not None:
        torch.save(model.img_cross_attn.state_dict(), save_dir / "img_cross_attn.pt")
    print(f"Saved adapter to {save_dir}")


def load_adapter(model: ImageCondCFM, load_dir: str, device: torch.device):
    load_dir = Path(load_dir)
    adapter_path = load_dir / "lora_adapter"
    if adapter_path.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        weights_file = adapter_path / "adapter_model.safetensors"
        if weights_file.exists():
            lora_state = load_file(str(weights_file), device=str(device))
        else:
            lora_state = torch.load(adapter_path / "adapter_model.bin",
                                    map_location=device, weights_only=True)
        set_peft_model_state_dict(model.cfm.transformer, lora_state)
    proj_state = torch.load(load_dir / "projector.pt",
                            map_location=device, weights_only=True)
    model.projector.load_state_dict(proj_state)
    xattn_path = load_dir / "img_cross_attn.pt"
    if xattn_path.exists() and model.img_cross_attn is not None:
        xattn_state = torch.load(xattn_path, map_location=device, weights_only=True)
        model.img_cross_attn.load_state_dict(xattn_state)
    return model
