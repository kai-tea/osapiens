"""Small MLP model for the Mark 2 embedding-only baseline."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class SmallMLP(nn.Module):
    """A compact MLP that maps per-pixel embedding vectors to a binary logit."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Extension point: larger hidden layers can be added here later.
        # Extension point: an image branch for multimodal fusion can be joined here later.

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw logits for a batch of pixel features."""
        return self.network(features).squeeze(-1)


def build_mlp(input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> SmallMLP:
    """Create the default small MLP baseline."""
    return SmallMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)


def save_model_checkpoint(
    model: SmallMLP,
    path: Path,
    *,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
) -> None:
    """Save model weights and minimal architecture metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> SmallMLP:
    """Load a saved small MLP checkpoint for inference or evaluation."""
    checkpoint = torch.load(path, map_location=map_location)
    model = build_mlp(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model
