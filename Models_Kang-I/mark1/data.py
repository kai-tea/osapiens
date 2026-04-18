from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


YEARS = (2020, 2021, 2022, 2023)
LABEL_SOURCES = ("radd", "gladl", "glads2")
GLADL_YEARS = (21, 22, 23, 24, 25)
TARGET_START_DATE = date(2021, 1, 1)
TARGET_END_DATE = date(2023, 12, 31)
RADD_REFERENCE_DATE = date(2014, 12, 31)
GLADS2_REFERENCE_DATE = date(2019, 1, 1)
RADD_TARGET_START_OFFSET = (TARGET_START_DATE - RADD_REFERENCE_DATE).days
RADD_TARGET_END_OFFSET = (TARGET_END_DATE - RADD_REFERENCE_DATE).days
GLADS2_TARGET_START_OFFSET = (TARGET_START_DATE - GLADS2_REFERENCE_DATE).days
GLADS2_TARGET_END_OFFSET = (TARGET_END_DATE - GLADS2_REFERENCE_DATE).days


def parse_tile_and_year(path: Path) -> tuple[str, int]:
    parts = path.stem.split("_")
    return "_".join(parts[:-1]), int(parts[-1])


def list_tiles(data_root: Path, split: str, years: Iterable[int] = YEARS) -> list[str]:
    """List AEF tiles that contain the required yearly embeddings for a split.

    The downloaded challenge data includes AEF embeddings through 2025, but the
    Mark 1 baseline intentionally keeps the original 2020-2023 feature window.
    """
    embedding_dir = data_root / "aef-embeddings" / split
    required_years = set(years)
    years_by_tile: dict[str, set[int]] = {}

    for path in sorted(embedding_dir.glob("*.tif*")):
        tile_id, year = parse_tile_and_year(path)
        years_by_tile.setdefault(tile_id, set()).add(year)

    return sorted(tile_id for tile_id, seen_years in years_by_tile.items() if required_years.issubset(seen_years))


def list_train_tiles(data_root: Path, years: Iterable[int] = YEARS) -> list[str]:
    return list_tiles(data_root, split="train", years=years)


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raster: {path}")

    with rasterio.open(path) as src:
        array = src.read()
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }

    return array, meta


def read_single_band(path: Path) -> tuple[np.ndarray, dict]:
    array, meta = read_raster(path)
    return array[0], meta


def _validate_embedding_meta(meta: dict, tile_id: str, year: int) -> None:
    if meta["count"] <= 0:
        raise ValueError(f"{tile_id}_{year}: embedding has no bands")
    if meta["height"] <= 0 or meta["width"] <= 0:
        raise ValueError(f"{tile_id}_{year}: embedding has invalid shape")
    if meta["crs"] is None:
        raise ValueError(f"{tile_id}_{year}: embedding CRS is missing")
    if meta["transform"] is None:
        raise ValueError(f"{tile_id}_{year}: embedding transform is missing")


def load_embeddings(
    data_root: Path,
    tile_id: str,
    split: str = "train",
    years: Iterable[int] = YEARS,
) -> tuple[dict[int, np.ndarray], dict]:
    embeddings: dict[int, np.ndarray] = {}
    reference_meta = None

    for year in years:
        path = data_root / "aef-embeddings" / split / f"{tile_id}_{year}.tiff"
        array, meta = read_raster(path)
        _validate_embedding_meta(meta, tile_id, year)

        if reference_meta is None:
            reference_meta = meta
        else:
            same_shape = meta["height"] == reference_meta["height"] and meta["width"] == reference_meta["width"]
            same_band_count = meta["count"] == reference_meta["count"]
            same_crs = meta["crs"] == reference_meta["crs"]
            same_transform = meta["transform"] == reference_meta["transform"]
            if not (same_shape and same_band_count and same_crs and same_transform):
                raise ValueError(f"{tile_id}_{year}: embedding metadata does not match the other yearly embeddings")

        embeddings[year] = array.astype(np.float32, copy=False)

    if reference_meta is None:
        raise FileNotFoundError(f"No embeddings found for tile '{tile_id}'")

    return embeddings, reference_meta


def build_valid_mask(embeddings: dict[int, np.ndarray], years: Iterable[int] = YEARS) -> np.ndarray:
    masks = [np.all(np.isfinite(embeddings[year]), axis=0) for year in years]
    return np.logical_and.reduce(masks)


def reproject_to_match(source: np.ndarray, source_meta: dict, reference_meta: dict) -> np.ndarray:
    aligned = np.zeros((reference_meta["height"], reference_meta["width"]), dtype=np.float32)
    reproject(
        source=source.astype(np.float32, copy=False),
        destination=aligned,
        src_transform=source_meta["transform"],
        src_crs=source_meta["crs"],
        dst_transform=reference_meta["transform"],
        dst_crs=reference_meta["crs"],
        resampling=Resampling.nearest,
    )
    return aligned


def load_weak_labels(data_root: Path, tile_id: str, reference_meta: dict) -> dict[str, np.ndarray]:
    labels_root = data_root / "labels" / "train"

    radd_raw, radd_meta = read_single_band(labels_root / "radd" / f"radd_{tile_id}_labels.tif")
    radd = reproject_to_match(radd_raw, radd_meta, reference_meta) > 0

    glads2_path = labels_root / "glads2" / f"glads2_{tile_id}_alert.tif"
    if glads2_path.exists():
        glads2_raw, glads2_meta = read_single_band(glads2_path)
        glads2 = reproject_to_match(glads2_raw, glads2_meta, reference_meta) > 0
    else:
        glads2 = np.zeros((reference_meta["height"], reference_meta["width"]), dtype=bool)

    gladl_layers = []
    gladl_meta = None
    for year_suffix in (21, 22, 23):
        gladl_path = labels_root / "gladl" / f"gladl_{tile_id}_alert{year_suffix}.tif"
        if gladl_path.exists():
            gladl_raw, gladl_meta = read_single_band(gladl_path)
            gladl_layers.append(gladl_raw)

    if gladl_layers:
        gladl_union = np.maximum.reduce(gladl_layers)
        gladl = reproject_to_match(gladl_union, gladl_meta, reference_meta) > 0
    else:
        gladl = np.zeros((reference_meta["height"], reference_meta["width"]), dtype=bool)

    return {"radd": radd, "gladl": gladl, "glads2": glads2}


def load_label_evidence(data_root: Path, tile_id: str, reference_meta: dict) -> dict[str, np.ndarray]:
    labels_root = data_root / "labels" / "train"
    height = reference_meta["height"]
    width = reference_meta["width"]

    def empty_bool() -> np.ndarray:
        return np.zeros((height, width), dtype=bool)

    strong_positive = empty_bool()
    weak_positive = empty_bool()
    future_alert = empty_bool()
    prior_alert = empty_bool()
    uncertain_alert = empty_bool()
    any_alert = empty_bool()
    strong_source_count = np.zeros((height, width), dtype=np.uint8)
    weak_source_count = np.zeros((height, width), dtype=np.uint8)

    # RADD: leading digit is confidence, remaining digits are days since 2014-12-31.
    radd_raw, radd_meta = read_single_band(labels_root / "radd" / f"radd_{tile_id}_labels.tif")
    radd = np.rint(reproject_to_match(radd_raw, radd_meta, reference_meta)).astype(np.int32, copy=False)
    radd_confidence = radd // 10_000
    radd_offset = radd % 10_000
    radd_has_alert = radd > 0
    any_alert |= radd_has_alert
    radd_strong = (radd_confidence == 3) & radd_has_alert & (radd_offset >= RADD_TARGET_START_OFFSET) & (radd_offset <= RADD_TARGET_END_OFFSET)
    radd_weak = (radd_confidence == 2) & radd_has_alert & (radd_offset >= RADD_TARGET_START_OFFSET) & (radd_offset <= RADD_TARGET_END_OFFSET)
    strong_positive |= radd_strong
    weak_positive |= radd_weak
    strong_source_count += radd_strong.astype(np.uint8)
    weak_source_count += radd_weak.astype(np.uint8)
    prior_alert |= radd_has_alert & (radd_offset < RADD_TARGET_START_OFFSET)
    future_alert |= radd_has_alert & (radd_offset > RADD_TARGET_END_OFFSET)

    # GLAD-S2: confidence stored in alert.tif, date in days since 2019-01-01 in alertDate.tif.
    glads2_alert_path = labels_root / "glads2" / f"glads2_{tile_id}_alert.tif"
    glads2_date_path = labels_root / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    if glads2_alert_path.exists() and glads2_date_path.exists():
        glads2_alert_raw, glads2_alert_meta = read_single_band(glads2_alert_path)
        glads2_date_raw, glads2_date_meta = read_single_band(glads2_date_path)
        glads2_alert = np.rint(reproject_to_match(glads2_alert_raw, glads2_alert_meta, reference_meta)).astype(np.int16, copy=False)
        glads2_date = np.rint(reproject_to_match(glads2_date_raw, glads2_date_meta, reference_meta)).astype(np.int32, copy=False)
        glads2_has_alert = glads2_alert > 0
        glads2_in_target = (glads2_date >= GLADS2_TARGET_START_OFFSET) & (glads2_date <= GLADS2_TARGET_END_OFFSET)
        any_alert |= glads2_has_alert
        glads2_strong = glads2_has_alert & glads2_in_target & (glads2_alert >= 3)
        glads2_weak = glads2_has_alert & glads2_in_target & (glads2_alert == 2)
        strong_positive |= glads2_strong
        weak_positive |= glads2_weak
        strong_source_count += glads2_strong.astype(np.uint8)
        weak_source_count += glads2_weak.astype(np.uint8)
        uncertain_alert |= glads2_has_alert & ((glads2_alert == 1) | ((glads2_alert == 2) & ~glads2_in_target))
        prior_alert |= glads2_has_alert & (glads2_date < GLADS2_TARGET_START_OFFSET)
        future_alert |= glads2_has_alert & (glads2_date > GLADS2_TARGET_END_OFFSET)

    # GLAD-L: yearly alert rasters with 2=probable, 3=confirmed.
    for year_suffix in GLADL_YEARS:
        alert_path = labels_root / "gladl" / f"gladl_{tile_id}_alert{year_suffix}.tif"
        date_path = labels_root / "gladl" / f"gladl_{tile_id}_alertDate{year_suffix}.tif"
        if not alert_path.exists() or not date_path.exists():
            continue

        alert_raw, alert_meta = read_single_band(alert_path)
        alert = np.rint(reproject_to_match(alert_raw, alert_meta, reference_meta)).astype(np.int16, copy=False)
        has_alert = alert > 0
        any_alert |= has_alert
        full_year = 2000 + year_suffix
        if 2021 <= full_year <= 2023:
            gladl_strong = has_alert & (alert == 3)
            gladl_weak = has_alert & (alert == 2)
            strong_positive |= gladl_strong
            weak_positive |= gladl_weak
            strong_source_count += gladl_strong.astype(np.uint8)
            weak_source_count += gladl_weak.astype(np.uint8)
        elif full_year < 2021:
            prior_alert |= has_alert
        else:
            future_alert |= has_alert

    return {
        "strong_positive": strong_positive,
        "weak_positive": weak_positive,
        "strong_source_count": strong_source_count,
        "weak_source_count": weak_source_count,
        "prior_alert": prior_alert,
        "future_alert": future_alert,
        "uncertain_alert": uncertain_alert,
        "any_alert": any_alert,
    }


def remove_tiny_blobs(mask: np.ndarray, min_blob_size: int = 4) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    cleaned = mask.copy()
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for row in range(height):
        for col in range(width):
            if not cleaned[row, col] or visited[row, col]:
                continue

            stack = [(row, col)]
            component = []
            visited[row, col] = True

            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))

                for d_row, d_col in neighbors:
                    next_row = current_row + d_row
                    next_col = current_col + d_col
                    inside = 0 <= next_row < height and 0 <= next_col < width
                    if inside and cleaned[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))

            if len(component) < min_blob_size:
                for component_row, component_col in component:
                    cleaned[component_row, component_col] = False

    return cleaned


def fuse_labels(
    weak_labels: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    apply_blob_filter: bool = False,
    min_blob_size: int = 4,
    positive_min_votes: int = 2,
    negative_max_votes: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    votes = np.stack([weak_labels[name].astype(np.uint8) for name in LABEL_SOURCES], axis=0).sum(axis=0)

    positive_mask = votes >= positive_min_votes
    if apply_blob_filter:
        positive_mask = remove_tiny_blobs(positive_mask, min_blob_size=min_blob_size)

    fused_labels = np.full(votes.shape, -1, dtype=np.int8)
    sample_weights = np.zeros(votes.shape, dtype=np.float32)
    fused_labels[valid_mask & (votes <= negative_max_votes)] = 0
    fused_labels[valid_mask & positive_mask] = 1
    sample_weights[fused_labels == 0] = 1.0
    sample_weights[fused_labels == 1] = 1.0
    train_mask = fused_labels >= 0

    return fused_labels, train_mask, sample_weights


def fuse_labels_confidence_date(
    label_evidence: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    apply_blob_filter: bool = False,
    min_blob_size: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strong_positive = label_evidence["strong_positive"]
    weak_positive = label_evidence["weak_positive"]
    strong_source_count = label_evidence["strong_source_count"].astype(np.float32, copy=False)
    weak_source_count = label_evidence["weak_source_count"].astype(np.float32, copy=False)
    prior_alert = label_evidence["prior_alert"]
    future_alert = label_evidence["future_alert"]
    uncertain_alert = label_evidence["uncertain_alert"]
    any_alert = label_evidence["any_alert"]

    positive_score = strong_source_count + 0.5 * weak_source_count
    conflict_mask = prior_alert | future_alert
    positive_mask = valid_mask & ~conflict_mask & ((strong_positive) | (positive_score >= 1.5))
    if apply_blob_filter:
        positive_mask = remove_tiny_blobs(positive_mask, min_blob_size=min_blob_size)

    negative_mask = valid_mask & ~any_alert & ~conflict_mask & ~uncertain_alert
    fused_labels = np.full(valid_mask.shape, -1, dtype=np.int8)
    sample_weights = np.zeros(valid_mask.shape, dtype=np.float32)

    fused_labels[negative_mask] = 0
    sample_weights[negative_mask] = 1.0

    fused_labels[positive_mask] = 1
    positive_weights = np.clip(0.5 + 0.2 * strong_source_count + 0.1 * weak_source_count, 0.6, 1.0)
    sample_weights[positive_mask] = positive_weights[positive_mask]

    train_mask = fused_labels >= 0
    return fused_labels, train_mask, sample_weights


def sample_training_tile(
    data_root: Path,
    tile_id: str,
    apply_blob_filter: bool = False,
    min_blob_size: int = 4,
    max_samples_per_tile: int | None = None,
    random_seed: int = 42,
    positive_min_votes: int = 2,
    negative_max_votes: int = 0,
    label_policy: str = "legacy_vote",
) -> dict:
    embeddings, embedding_meta = load_embeddings(data_root, tile_id)
    valid_mask = build_valid_mask(embeddings)
    if label_policy == "legacy_vote":
        weak_labels = load_weak_labels(data_root, tile_id, embedding_meta)
        fused_labels, train_mask, sample_weights = fuse_labels(
            weak_labels,
            valid_mask,
            apply_blob_filter=apply_blob_filter,
            min_blob_size=min_blob_size,
            positive_min_votes=positive_min_votes,
            negative_max_votes=negative_max_votes,
        )
    elif label_policy == "confidence_date":
        label_evidence = load_label_evidence(data_root, tile_id, embedding_meta)
        fused_labels, train_mask, sample_weights = fuse_labels_confidence_date(
            label_evidence,
            valid_mask,
            apply_blob_filter=apply_blob_filter,
            min_blob_size=min_blob_size,
        )
    else:
        raise ValueError(f"Unsupported label_policy: {label_policy}")

    rows, cols = np.where(train_mask)
    if max_samples_per_tile is not None and rows.size > max_samples_per_tile:
        rng = np.random.default_rng(random_seed)
        keep_indices = rng.choice(rows.size, size=max_samples_per_tile, replace=False)
        rows = rows[keep_indices]
        cols = cols[keep_indices]

    return {
        "tile_id": tile_id,
        "embeddings": embeddings,
        "embedding_meta": embedding_meta,
        "valid_mask": valid_mask,
        "rows": rows.astype(np.int32),
        "cols": cols.astype(np.int32),
        "labels": fused_labels[rows, cols].astype(np.int8),
        "sample_weights": sample_weights[rows, cols].astype(np.float32),
    }


def write_raster(path: Path, array: np.ndarray, reference_meta: dict, dtype: str, nodata: float | int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": reference_meta["height"],
        "width": reference_meta["width"],
        "count": 1,
        "dtype": dtype,
        "crs": reference_meta["crs"],
        "transform": reference_meta["transform"],
        "compress": "lzw",
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
