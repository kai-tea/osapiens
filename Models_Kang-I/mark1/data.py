from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


YEARS = (2020, 2021, 2022, 2023)
LABEL_SOURCES = ("radd", "gladl", "glads2")


def parse_tile_and_year(path: Path) -> tuple[str, int]:
    parts = path.stem.split("_")
    return "_".join(parts[:-1]), int(parts[-1])


def list_train_tiles(data_root: Path, years: Iterable[int] = YEARS) -> list[str]:
    embedding_dir = data_root / "aef-embeddings" / "train"
    required_years = set(years)
    years_by_tile: dict[str, set[int]] = {}

    for path in sorted(embedding_dir.glob("*.tif*")):
        tile_id, year = parse_tile_and_year(path)
        years_by_tile.setdefault(tile_id, set()).add(year)

    return sorted(tile_id for tile_id, seen_years in years_by_tile.items() if required_years.issubset(seen_years))


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


def load_embeddings(data_root: Path, tile_id: str, years: Iterable[int] = YEARS) -> tuple[dict[int, np.ndarray], dict]:
    embeddings: dict[int, np.ndarray] = {}
    reference_meta = None

    for year in years:
        path = data_root / "aef-embeddings" / "train" / f"{tile_id}_{year}.tiff"
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
) -> tuple[np.ndarray, np.ndarray]:
    votes = np.stack([weak_labels[name].astype(np.uint8) for name in LABEL_SOURCES], axis=0).sum(axis=0)

    positive_mask = votes >= 2
    if apply_blob_filter:
        positive_mask = remove_tiny_blobs(positive_mask, min_blob_size=min_blob_size)

    fused_labels = np.full(votes.shape, -1, dtype=np.int8)
    fused_labels[valid_mask & (votes == 0)] = 0
    fused_labels[valid_mask & positive_mask] = 1
    train_mask = fused_labels >= 0

    return fused_labels, train_mask


def sample_training_tile(
    data_root: Path,
    tile_id: str,
    apply_blob_filter: bool = False,
    min_blob_size: int = 4,
    max_samples_per_tile: int | None = None,
    random_seed: int = 42,
) -> dict:
    embeddings, embedding_meta = load_embeddings(data_root, tile_id)
    valid_mask = build_valid_mask(embeddings)
    weak_labels = load_weak_labels(data_root, tile_id, embedding_meta)
    fused_labels, train_mask = fuse_labels(
        weak_labels,
        valid_mask,
        apply_blob_filter=apply_blob_filter,
        min_blob_size=min_blob_size,
    )

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
