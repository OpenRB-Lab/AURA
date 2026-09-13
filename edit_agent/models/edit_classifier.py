"""Hierarchical semantic classifier over the 9 edit-token hidden states.

Forces the thinker's edit tokens to explicitly encode (kind, instrument):
pool = [h_kind ; mean(h_payload)] -> trunk -> two heads. Used as an
auxiliary loss in the joint stage (live h) and as a standalone probe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from edit_agent.tokens import EDIT_KINDS

INST_CLASSES = ["drums", "bass", "guitar", "keys", "strings", "brass_wind",
                "synth", "vocals", "sfx_other", "none"]
KIND_TO_ID = {k: i for i, k in enumerate(EDIT_KINDS)}
INST_TO_ID = {c: i for i, c in enumerate(INST_CLASSES)}


class EditSemanticClassifier(nn.Module):
    def __init__(self, d: int = 3584, hidden: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(2 * d),
                                   nn.Linear(2 * d, hidden), nn.GELU())
        self.head_kind = nn.Linear(hidden, len(EDIT_KINDS))
        self.head_inst = nn.Linear(hidden, len(INST_CLASSES))

    def forward(self, h):
        """h: [B, 9, d] -> (kind_logits [B,7], inst_logits [B,10])"""
        pooled = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=-1).float()
        z = self.trunk(pooled)
        return self.head_kind(z), self.head_inst(z)

    def loss(self, h, kind_label, inst_label):
        """labels: str or None; returns (loss, dict of accs). Unknown inst
        (label None) contributes only the kind term."""
        kl, il = self.forward(h)
        dev = h.device
        loss = torch.zeros((), device=dev)
        stats = {}
        if kind_label is not None and kind_label in KIND_TO_ID:
            t = torch.tensor([KIND_TO_ID[kind_label]], device=dev)
            loss = loss + F.cross_entropy(kl, t)
            stats["kind_ok"] = int(kl.argmax(-1).item() == t.item())
        if inst_label is not None and inst_label in INST_TO_ID:
            t = torch.tensor([INST_TO_ID[inst_label]], device=dev)
            loss = loss + F.cross_entropy(il, t)
            stats["inst_ok"] = int(il.argmax(-1).item() == t.item())
        return loss, stats
