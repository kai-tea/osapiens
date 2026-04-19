"""Transparent rule-based weak-label fusion for the Mark 2 baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from ..utils.io import read_single_band, reproject_to_match
except ImportError:
    from utils.io import read_single_band, reproject_to_match


def load_radd_mask(data_root: Path, tile_id: str, reference_meta: dict) -> np.ndarray:
    """Load the RADD weak label as a boolean deforestation mask."""
    path = data_root / "labels" / "train" / "radd" / f"radd_{tile_id}_labels.tif"
    raster, meta = read_single_band(path)
    return reproject_to_match(raster, meta, reference_meta) > 0


def load_gladl_mask(data_root: Path, tile_id: str, reference_meta: dict) -> np.ndarray:
    """Union all available GLAD-L alert rasters for a tile into one boolean mask."""
    pattern = f"gladl_{tile_id}_alert[0-9][0-9].tif"
    paths = sorted((data_root / "labels" / "train" / "gladl").glob(pattern))
    if not paths:
        return np.zeros((reference_meta["height"], reference_meta["width"]), dtype=bool)

    layers = []
    meta = None
    for path in paths:
        raster, meta = read_single_band(path)
        layers.append(raster)

    union = np.maximum.reduce(layers)
    return reproject_to_match(union, meta, reference_meta) > 0


def load_glads2_mask(data_root: Path, tile_id: str, reference_meta: dict) -> np.ndarray:
    """Load the GLAD-S2 weak label as a boolean deforestation mask."""
    path = data_root / "labels" / "train" / "glads2" / f"glads2_{tile_id}_alert.tif"
    if not path.exists():
        return np.zeros((reference_meta["height"], reference_meta["width"]), dtype=bool)

    raster, meta = read_single_band(path)
    return reproject_to_match(raster, meta, reference_meta) > 0


def load_weak_label_sources(data_root: Path, tile_id: str, reference_meta: dict) -> dict[str, np.ndarray]:
    """Load the three weak sources needed for the conservative label combiner."""
    return {
        "radd": load_radd_mask(data_root, tile_id, reference_meta),
        "gladl": load_gladl_mask(data_root, tile_id, reference_meta),
        "glads2": load_glads2_mask(data_root, tile_id, reference_meta),
    }


def combine_weak_labels(source_masks: dict[str, np.ndarray]) -> np.ndarray:
    """Combine weak sources into positive, negative, and uncertain labels."""
    votes = np.stack(
        [
            source_masks["radd"].astype(np.uint8),
            source_masks["gladl"].astype(np.uint8),
            source_masks["glads2"].astype(np.uint8),
        ],
        axis=0,
    ).sum(axis=0)

    labels = np.full(votes.shape, -1, dtype=np.int8)

    # Rule 1: require agreement from at least two sources before calling a pixel positive.
    labels[votes >= 2] = 1

    # Rule 2: only call a pixel negative when all three sources are silent.
    labels[votes == 0] = 0

    # Rule 3: a single positive source stays uncertain for this conservative v1 baseline.
    return labels


def build_combined_labels(data_root: Path, tile_id: str, reference_meta: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load source masks and return the combined weak labels for one tile."""
    source_masks = load_weak_label_sources(data_root, tile_id, reference_meta)

    # Extension point: weighted label fusion can be added here later.
    # Extension point: forest-map-informed relabeling can be added here later.
    labels = combine_weak_labels(source_masks)
    return labels, source_masks


def build_valid_training_mask(labels: np.ndarray) -> np.ndarray:
    """Return the training mask that keeps only non-uncertain labels."""
    return labels != -1


def summarize_label_counts(labels: np.ndarray) -> dict[str, int]:
    """Count positive, negative, and uncertain pixels for quick inspection."""
    return {
        "positive": int(np.count_nonzero(labels == 1)),
        "negative": int(np.count_nonzero(labels == 0)),
        "uncertain": int(np.count_nonzero(labels == -1)),
    }
