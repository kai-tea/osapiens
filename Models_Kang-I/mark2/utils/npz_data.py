"""Utilities for reading per-tile `.npz` artifacts for training and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def list_npz_files(directory: Path) -> list[Path]:
    """Return `.npz` files from a directory in deterministic order."""
    return sorted(directory.glob("*.npz"))


def load_tile_npz(path: Path) -> dict[str, np.ndarray]:
    """Load one saved tile artifact from disk."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def flatten_valid_pixels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Flatten features and labels, keeping only pixels where `valid_mask` is true."""
    tile = load_tile_npz(path)
    features = tile["features"].reshape(-1, tile["features"].shape[-1]).astype(np.float32, copy=False)
    labels = tile["labels"].reshape(-1).astype(np.int64, copy=False)
    valid_mask = tile["valid_mask"].reshape(-1).astype(bool, copy=False)

    keep_mask = valid_mask & ((labels == 0) | (labels == 1))
    return features[keep_mask], labels[keep_mask]


def infer_input_dim_from_npz(directory: Path) -> int:
    """Read the embedding channel count from the first `.npz` file in a directory."""
    files = list_npz_files(directory)
    if not files:
        raise FileNotFoundError(f"No .npz files found in {directory}")

    tile = load_tile_npz(files[0])
    return int(tile["features"].shape[-1])


def count_split_pixels(directory: Path) -> dict[str, int]:
    """Count valid pixels and class balance for a split of `.npz` tile files."""
    totals = {"pixels": 0, "positive": 0, "negative": 0}
    for path in list_npz_files(directory):
        _, labels = flatten_valid_pixels(path)
        totals["pixels"] += int(labels.shape[0])
        totals["positive"] += int(np.count_nonzero(labels == 1))
        totals["negative"] += int(np.count_nonzero(labels == 0))
    return totals


def iterate_training_batches(
    directory: Path,
    batch_size: int,
    epoch_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Yield deterministic shuffled mini-batches of valid pixel rows for one epoch."""
    rng = np.random.default_rng(epoch_seed)
    file_paths = list_npz_files(directory)
    if not file_paths:
        raise FileNotFoundError(f"No .npz files found in {directory}")

    file_order = rng.permutation(len(file_paths))
    for file_index in file_order:
        features, labels = flatten_valid_pixels(file_paths[int(file_index)])
        if labels.size == 0:
            continue

        pixel_order = rng.permutation(labels.shape[0])
        features = features[pixel_order]
        labels = labels[pixel_order]

        for start in range(0, labels.shape[0], batch_size):
            end = start + batch_size
            yield features[start:end], labels[start:end]


def iterate_eval_batches(directory: Path, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Yield deterministic evaluation batches from a split of `.npz` files."""
    file_paths = list_npz_files(directory)
    if not file_paths:
        raise FileNotFoundError(f"No .npz files found in {directory}")

    for path in file_paths:
        features, labels = flatten_valid_pixels(path)
        if labels.size == 0:
            continue

        for start in range(0, labels.shape[0], batch_size):
            end = start + batch_size
            yield features[start:end], labels[start:end]


def iter_prediction_inputs(directory: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Yield tile artifacts for prediction in deterministic order."""
    for path in list_npz_files(directory):
        yield path, load_tile_npz(path)
