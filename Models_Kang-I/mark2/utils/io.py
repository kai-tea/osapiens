"""Lightweight I/O helpers for the Mark 2 embedding-only baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def parse_tile_and_year(path: Path) -> tuple[str, int]:
    """Extract the tile identifier and year from an embedding filename."""
    parts = path.stem.split("_")
    return "_".join(parts[:-1]), int(parts[-1])


def list_embedding_paths(data_root: Path, split: str) -> list[Path]:
    """Return all embedding rasters for a split in deterministic order."""
    embedding_dir = data_root / "aef-embeddings" / split
    return sorted(embedding_dir.glob("*.tif*"))


def list_tiles_for_year(data_root: Path, split: str, year: int) -> list[str]:
    """List tile ids that have an embedding raster for the requested year."""
    tile_ids = []
    for path in list_embedding_paths(data_root, split):
        tile_id, path_year = parse_tile_and_year(path)
        if path_year == year:
            tile_ids.append(tile_id)
    return sorted(tile_ids)


def available_years_by_split(data_root: Path, split: str) -> dict[str, set[int]]:
    """Map each tile id to the embedding years available for a split."""
    years_by_tile: dict[str, set[int]] = {}
    for path in list_embedding_paths(data_root, split):
        tile_id, year = parse_tile_and_year(path)
        years_by_tile.setdefault(tile_id, set()).add(year)
    return years_by_tile


def infer_latest_common_year(data_root: Path, splits: tuple[str, ...] = ("train", "test")) -> int:
    """Infer the latest embedding year shared by every tile in the requested splits."""
    common_years: set[int] | None = None
    for split in splits:
        years_by_tile = available_years_by_split(data_root, split)
        if not years_by_tile:
            raise FileNotFoundError(f"No embeddings found for split '{split}'")
        split_common_years = set.intersection(*(set(years) for years in years_by_tile.values()))
        common_years = split_common_years if common_years is None else common_years & split_common_years

    if not common_years:
        raise ValueError(f"No common embedding year found across splits: {splits}")

    return max(common_years)


def split_training_tiles(tile_ids: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    """Create a deterministic train/validation tile split from sorted tile ids."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if len(tile_ids) <= 1 or val_fraction == 0.0:
        return sorted(tile_ids), []

    sorted_tiles = sorted(tile_ids)
    validation_count = max(1, int(round(len(sorted_tiles) * val_fraction)))
    validation_count = min(validation_count, len(sorted_tiles) - 1)
    validation_tiles = sorted_tiles[-validation_count:]
    training_tiles = sorted_tiles[:-validation_count]
    return training_tiles, validation_tiles


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    """Read a raster and return its array data plus minimal metadata."""
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
    """Read the first raster band and keep the metadata."""
    array, meta = read_raster(path)
    return array[0], meta


def load_embedding(data_root: Path, tile_id: str, split: str, year: int) -> tuple[np.ndarray, dict]:
    """Load one embedding raster for a tile, split, and year."""
    path = data_root / "aef-embeddings" / split / f"{tile_id}_{year}.tiff"
    array, meta = read_raster(path)
    return array.astype(np.float32, copy=False), meta


def reproject_to_match(source: np.ndarray, source_meta: dict, reference_meta: dict) -> np.ndarray:
    """Reproject a single-band raster to the embedding grid."""
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
