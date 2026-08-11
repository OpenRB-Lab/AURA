"""Projector: maps image (+ optional text) embeddings into MuQ-MuLan style space."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageToMuLanProjector(nn.Module):
    def __init__(self, d_img: int = 768, d_mulan: int = 512,
                 d_text: int = 512, hidden_dim: int = 1024,
                 num_layers: int = 2, dropout: float = 0.1,
                 l2_normalize: bool = True):
        super().__init__()
        self.l2_normalize = l2_normalize
        self.d_text = d_text

        input_dim = d_img + d_text
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else d_mulan
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z_img: torch.Tensor, z_text: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            z_img: [B, d_img] pooled image embedding
            z_text: [B, d_text] MuLan text embedding (optional, zeros if None)
        Returns:
            z_style: [B, d_mulan] projected embedding
        """
        if z_text is None:
            z_text = torch.zeros(z_img.shape[0], self.d_text,
                                 device=z_img.device, dtype=z_img.dtype)
        z = torch.cat([z_img, z_text], dim=-1)
        z = self.net(z)
        if self.l2_normalize:
            z = F.normalize(z, dim=-1)
        return z
