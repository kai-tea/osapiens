"""Small MLP + Union-IoU-aligned loss for per-pixel deforestation scoring.

Why an MLP on Kaite's 402-dim feature vector instead of a temporal TCN on
raw monthly stacks: raw-stack loading is ~30 min / tile; in a 4 h wall-
clock window that drowns the whole budget. The 402-dim parquets are already
cached by src.data, so this trainer lives entirely in-memory on one GPU.
What we add over LightGBM v1:

  1. Soft-IoU loss aligned with the challenge's Union-IoU metric
     (BCE alone rewards confidence on easy pixels).
  2. Feature-group dropout on the AEF block — v1's post-mortem showed AEF
     was the top-importance feature but extrapolated catastrophically on
     out-of-distribution MGRS tiles. We randomly zero the AEF block per
     batch so the network cannot become AEF-dependent.
  3. Per-MGRS inverse-frequency sample weighting to fight the 0.44 F1
     regional gap (see io.per_region_weights).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    in_dim: int = 402
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.3
    aef_mask: tuple[bool, ...] | None = None  # True at AEF feature positions
    aef_drop_prob: float = 0.5                # chance a batch zeroes AEF


class PixelMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        d = cfg.in_dim
        for h in cfg.hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(cfg.dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        if cfg.aef_mask is not None:
            self.register_buffer(
                "_aef_mask",
                torch.tensor(cfg.aef_mask, dtype=torch.bool),
                persistent=False,
            )
        else:
            self._aef_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self._aef_mask is not None and self.cfg.aef_drop_prob > 0:
            if torch.rand((), device=x.device) < self.cfg.aef_drop_prob:
                x = x.clone()
                x[:, self._aef_mask] = 0.0
        return self.net(x).squeeze(-1)


def soft_iou_loss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable soft-IoU, weighted.

    IoU = sum(p*y) / (sum(p) + sum(y) - sum(p*y))
    Loss = 1 - IoU. Aligned with the leaderboard's Union-IoU primary metric.
    """
    p = torch.sigmoid(logits)
    pw = p * w
    yw = y * w
    inter = (pw * y).sum()
    union = pw.sum() + yw.sum() - inter
    return 1.0 - inter / (union + eps)


def bce_weighted(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, pos_weight: float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, y, weight=w, pos_weight=torch.tensor(pos_weight, device=logits.device)
    )


def combined_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    pos_weight: float,
    iou_weight: float = 0.5,
) -> torch.Tensor:
    return (1 - iou_weight) * bce_weighted(logits, y, w, pos_weight) + iou_weight * soft_iou_loss(logits, y, w)
