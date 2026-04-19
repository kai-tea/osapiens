"""Transparent rule-based weak-label fusion for the Mark 2 baseline."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds

try:
    from ..utils.io import read_single_band, reproject_to_match
except ImportError:
    from utils.io import read_single_band, reproject_to_match


JRC_DIR_CANDIDATES = (
    Path("data/external/jrc_gfc2020"),
    Path("~/jrc_gfc2020").expanduser(),
)
JRC_CRS = "EPSG:4326"


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


def _jrc_tile_name_candidates(lat_top: int, lon_left: int) -> list[str]:
    """Return possible JRC filenames for one 10x10 degree tile."""
    ns = "N" if lat_top >= 0 else "S"
    ew = "E" if lon_left >= 0 else "W"
    lat_token = f"{ns}{abs(lat_top):02d}"
    lon_value = abs(lon_left)
    lon_tokens = [f"{ew}{lon_value:02d}"]

    padded_lon_token = f"{ew}{lon_value:03d}"
    if padded_lon_token not in lon_tokens:
        lon_tokens.append(padded_lon_token)

    return [f"JRC_GFC2020_V3_{lat_token}_{lon_token}.tif" for lon_token in lon_tokens]


def _resolve_jrc_tile_path(jrc_root: Path, lat_top: int, lon_left: int) -> tuple[Path | None, list[str], str | None]:
    """Resolve one JRC tile path while supporting multiple filename conventions."""
    candidate_names = _jrc_tile_name_candidates(lat_top, lon_left)
    for candidate_name in candidate_names:
        candidate_path = jrc_root / candidate_name
        if candidate_path.exists():
            return candidate_path, candidate_names, candidate_name
    return None, candidate_names, None


def _tiles_covering_bbox_4326(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
) -> list[tuple[int, int]]:
    """Enumerate JRC 10x10 degree tiles covering a bounding box."""
    lat_top_min = int(math.ceil(lat_min / 10.0)) * 10
    lat_top_max = int(math.ceil(lat_max / 10.0)) * 10
    lon_left_min = int(math.floor(lon_min / 10.0)) * 10
    lon_left_max = int(math.floor(lon_max / 10.0)) * 10

    tiles = []
    for lat_top in range(lat_top_min, lat_top_max + 1, 10):
        for lon_left in range(lon_left_min, lon_left_max + 1, 10):
            tiles.append((lat_top, lon_left))
    return tiles


def _find_jrc_root() -> Path | None:
    """Return the first JRC directory that exists on disk."""
    for path in JRC_DIR_CANDIDATES:
        if path.exists():
            return path
    return None


def load_jrc_forest_mask(reference_meta: dict) -> tuple[np.ndarray | None, dict[str, object]]:
    """Load a boolean JRC forest mask aligned to the embedding grid if JRC is available."""
    jrc_root = _find_jrc_root()
    if jrc_root is None:
        return None, {
            "jrc_available": False,
            "jrc_root": None,
            "forest_fraction": None,
            "used_jrc_files": [],
            "missing_jrc_candidates": [],
        }

    left, bottom, right, top = array_bounds(
        reference_meta["height"],
        reference_meta["width"],
        reference_meta["transform"],
    )
    lon_min, lat_min, lon_max, lat_max = transform_bounds(
        reference_meta["crs"],
        JRC_CRS,
        left,
        bottom,
        right,
        top,
        densify_pts=21,
    )
    needed_tiles = _tiles_covering_bbox_4326(lon_min, lat_min, lon_max, lat_max)

    aligned_layers = []
    used_jrc_files: list[str] = []
    missing_jrc_candidates: list[dict[str, object]] = []
    for lat_top, lon_left in needed_tiles:
        path, candidate_names, matched_name = _resolve_jrc_tile_path(jrc_root, lat_top, lon_left)
        if path is None:
            missing_jrc_candidates.append(
                {
                    "lat_top": lat_top,
                    "lon_left": lon_left,
                    "candidate_names": candidate_names,
                }
            )
            continue
        raster, meta = read_single_band(path)
        aligned = reproject_to_match(raster, meta, reference_meta)
        aligned_layers.append(aligned > 0)
        used_jrc_files.append(matched_name if matched_name is not None else path.name)

    if not aligned_layers:
        return None, {
            "jrc_available": False,
            "jrc_root": str(jrc_root),
            "forest_fraction": None,
            "used_jrc_files": [],
            "missing_jrc_candidates": missing_jrc_candidates,
        }

    forest_mask = np.logical_or.reduce(aligned_layers)
    return forest_mask, {
        "jrc_available": True,
        "jrc_root": str(jrc_root),
        "forest_fraction": float(forest_mask.mean()),
        "used_jrc_files": used_jrc_files,
        "missing_jrc_candidates": missing_jrc_candidates,
    }


def apply_jrc_label_gating(
    labels: np.ndarray,
    forest_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, int | float | bool | str | None]]:
    """Gate labels to plausible forest pixels using JRC forest cover when available."""
    before_counts = summarize_label_counts(labels)
    if forest_mask is None:
        return labels, {
            "jrc_available": False,
            "forest_fraction": None,
            "positive_before": before_counts["positive"],
            "negative_before": before_counts["negative"],
            "uncertain_before": before_counts["uncertain"],
            "positive_after": before_counts["positive"],
            "negative_after": before_counts["negative"],
            "uncertain_after": before_counts["uncertain"],
            "positive_suppressed_outside_forest": 0,
            "negative_ignored_outside_forest": 0,
        }

    gated_labels = labels.copy()
    positive_outside_forest = (labels == 1) & (~forest_mask)
    negative_outside_forest = (labels == 0) & (~forest_mask)

    # Keep training focused on plausible forest pixels and ignore positives outside forest.
    gated_labels[positive_outside_forest] = -1

    # Remove easy non-forest negatives so the model is not dominated by trivial background.
    gated_labels[negative_outside_forest] = -1

    after_counts = summarize_label_counts(gated_labels)
    return gated_labels, {
        "jrc_available": True,
        "forest_fraction": float(forest_mask.mean()),
        "positive_before": before_counts["positive"],
        "negative_before": before_counts["negative"],
        "uncertain_before": before_counts["uncertain"],
        "positive_after": after_counts["positive"],
        "negative_after": after_counts["negative"],
        "uncertain_after": after_counts["uncertain"],
        "positive_suppressed_outside_forest": int(np.count_nonzero(positive_outside_forest)),
        "negative_ignored_outside_forest": int(np.count_nonzero(negative_outside_forest)),
    }


def build_combined_labels(data_root: Path, tile_id: str, reference_meta: dict) -> tuple[np.ndarray, dict[str, object]]:
    """Load weak labels, apply simple JRC gating, and return labels plus diagnostics."""
    source_masks = load_weak_label_sources(data_root, tile_id, reference_meta)
    labels = combine_weak_labels(source_masks)
    forest_mask, forest_summary = load_jrc_forest_mask(reference_meta)
    gated_labels, gating_summary = apply_jrc_label_gating(labels, forest_mask)

    # Extension point: weighted label fusion can be added here later.
    # Extension point: forest-map-informed relabeling can be made more selective later.
    return gated_labels, {
        "source_masks": source_masks,
        "forest_summary": forest_summary,
        "gating_summary": gating_summary,
    }


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
