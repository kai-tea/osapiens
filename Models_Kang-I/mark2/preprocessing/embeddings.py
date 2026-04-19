"""Transparent embedding preprocessing for the Mark 2 baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from ..utils.io import load_embedding
except ImportError:
    from utils.io import load_embedding


def sanitize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Replace NaN and infinite values with zeros in a deterministic way."""
    return np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def compute_channel_statistics(
    data_root: Path,
    tile_ids: list[str],
    split: str,
    year: int,
) -> dict[str, np.ndarray]:
    """Compute per-channel mean and standard deviation from the training split only."""
    if not tile_ids:
        raise ValueError("At least one tile is required to compute normalization statistics")

    channel_sum: np.ndarray | None = None
    channel_sum_sq: np.ndarray | None = None
    pixel_count = 0

    for tile_id in tile_ids:
        embedding, _ = load_embedding(data_root=data_root, tile_id=tile_id, split=split, year=year)
        embedding = sanitize_embedding(embedding)
        flattened = embedding.reshape(embedding.shape[0], -1).astype(np.float64, copy=False)

        if channel_sum is None:
            channel_sum = np.zeros(flattened.shape[0], dtype=np.float64)
            channel_sum_sq = np.zeros(flattened.shape[0], dtype=np.float64)

        channel_sum += flattened.sum(axis=1)
        channel_sum_sq += np.square(flattened).sum(axis=1)
        pixel_count += flattened.shape[1]

    if channel_sum is None or channel_sum_sq is None or pixel_count == 0:
        raise ValueError("Could not compute normalization statistics from the provided tiles")

    mean = channel_sum / pixel_count
    variance = np.maximum((channel_sum_sq / pixel_count) - np.square(mean), 0.0)
    std = np.sqrt(variance)

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "pixel_count": np.array([pixel_count], dtype=np.int64),
    }


def normalize_embedding(embedding: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply channel-wise standardization using precomputed training statistics."""
    safe_embedding = sanitize_embedding(embedding)
    normalized = (safe_embedding - mean[:, None, None]) / (std[:, None, None] + 1e-6)
    return normalized.astype(np.float32, copy=False)


def extract_pixel_features(normalized_embedding: np.ndarray) -> np.ndarray:
    """Return the raw normalized embedding vector for each pixel."""
    features = np.moveaxis(normalized_embedding, 0, -1)

    # Extension point: local spatial embedding features such as 3x3 mean/std can be
    # concatenated here later without changing the rest of the pipeline.
    return features.astype(np.float32, copy=False)


def save_normalization_statistics(
    output_path: Path,
    *,
    year: int,
    training_tiles: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    pixel_count: int,
) -> None:
    """Save normalization statistics so validation and test processing can reuse them."""
    payload = {
        "year": year,
        "training_tiles": training_tiles,
        "pixel_count": pixel_count,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def load_normalization_statistics(path: Path) -> dict[str, np.ndarray | int | list[str]]:
    """Load saved normalization statistics from disk."""
    payload = json.loads(path.read_text())
    return {
        "year": int(payload["year"]),
        "training_tiles": list(payload["training_tiles"]),
        "pixel_count": int(payload["pixel_count"]),
        "mean": np.asarray(payload["mean"], dtype=np.float32),
        "std": np.asarray(payload["std"], dtype=np.float32),
    }
