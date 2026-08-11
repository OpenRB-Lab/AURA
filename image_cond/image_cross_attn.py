"""Image cross-attention modules for ControlNet-style conditioning in DiT.

Injects trainable cross-attention layers into the DiT backbone.
Q comes from DiT hidden states, K/V from projected image patch tokens.
Output projections are zero-initialized for stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImageCrossAttention(nn.Module):
    """Single cross-attention layer: DiT hidden states attend to image patch tokens."""

    def __init__(self, dim: int = 2048, num_heads: int = 16,
                 d_img: int = 768, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(d_img, dim, bias=False)
        self.v_proj = nn.Linear(d_img, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Zero-init output so cross-attn starts as identity
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, img_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] DiT hidden states
            img_tokens: [B, S, d_img] image patch tokens (S=256 for SigLIP)
        Returns:
            [B, N, dim] cross-attention output (additive residual)
        """
        B, N, _ = x.shape
        S = img_tokens.shape[1]

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(img_tokens).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(img_tokens).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Manual attention (avoids cuDNN SDPA issues)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn = torch.matmul(attn_weights, v)
        attn = attn.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(attn)


class ImageConditioningAdapter(nn.Module):
    """Full adapter: projects image tokens + applies cross-attention at selected DiT layers."""

    def __init__(self, dit_dim: int = 2048, d_img: int = 768,
                 num_heads: int = 16, num_layers: int = 8,
                 total_dit_layers: int = 16, dropout: float = 0.0):
        super().__init__()
        self.total_dit_layers = total_dit_layers

        # Which DiT layers get cross-attention (every other)
        self.cross_attn_layers = list(range(0, total_dit_layers, total_dit_layers // num_layers))

        self.layer_norms = nn.ModuleDict()
        self.cross_attns = nn.ModuleDict()

        for idx in self.cross_attn_layers:
            self.layer_norms[str(idx)] = nn.LayerNorm(dit_dim)
            self.cross_attns[str(idx)] = ImageCrossAttention(
                dim=dit_dim, num_heads=num_heads,
                d_img=d_img, dropout=dropout,
            )

    def forward_layer(self, layer_idx: int, x: torch.Tensor,
                      img_tokens: torch.Tensor) -> torch.Tensor:
        key = str(layer_idx)
        if key in self.cross_attns:
            residual = self.cross_attns[key](self.layer_norms[key](x), img_tokens)
            return x + residual
        return x
