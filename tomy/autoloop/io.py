"""IO helpers for the autoloop — reuse Kaite's src/ utilities where possible."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    FEATURES_ROOT,
    feature_names,
    load_tile_features,
    load_tile_manifest,
)
from src.eval import load_fold_assignments

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

AEF_PREFIX = "aef_"


def aef_column_mask(columns: list[str]) -> np.ndarray:
    """Boolean mask selecting the AEF feature block (192 dims out of 402)."""
    return np.array([c.startswith(AEF_PREFIX) for c in columns], dtype=bool)


def load_train_frame(tile_ids: list[str]) -> pd.DataFrame:
    """Concatenate Kaite's cached feature parquets for the given train tiles.

    Drops rows with NaN in any feature column (rare: cloud-only pixels after
    interp). Adds an MGRS column for per-region sampling weights.
    """
    frames: list[pd.DataFrame] = []
    for tid in tile_ids:
        frames.append(load_tile_features(tid))
    df = pd.concat(frames, ignore_index=True)
    feats = feature_names()
    df = df.dropna(subset=feats).reset_index(drop=True)
    df["mgrs"] = df["region_id"].astype(str)
    return df


def per_region_weights(df: pd.DataFrame) -> np.ndarray:
    """Inverse MGRS-frequency weight, normalised to mean 1.

    Fights v1's 0.44 F1 regional gap — SE Asia tiles dominate row counts so
    unweighted training over-learns that biome.
    """
    counts = df["mgrs"].value_counts()
    w_per_region = (counts.mean() / counts).to_dict()
    return df["mgrs"].map(w_per_region).to_numpy().astype(np.float32)


def label_targets(df: pd.DataFrame, positive_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (y, soft_weight) arrays.

    y is the hard target from soft_target >= threshold; soft_weight combines
    Cini's sample_weight with the per-region inverse frequency.
    """
    y = (df["soft_target"].to_numpy() >= positive_threshold).astype(np.float32)
    w = df["sample_weight"].to_numpy().astype(np.float32)
    w_region = per_region_weights(df)
    return y, (w * w_region).astype(np.float32)


__all__ = [
    "REPO_ROOT",
    "aef_column_mask",
    "feature_names",
    "label_targets",
    "load_fold_assignments",
    "load_tile_manifest",
    "load_train_frame",
    "per_region_weights",
]
