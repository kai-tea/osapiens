"""JRC Global Forest Cover 2020 V3 loader.

Resamples the 10 m EPSG:4326 JRC raster onto a project tile's canonical UTM
grid. Used as the authoritative 2020-forest gate in place of the NDVI heuristic
in ``label_fusion.forest_mask_2020``.

Tile convention (confirmed against the live FTP on 2026-04-18):
    JRC_GFC2020_V3_<NS><lat>_<EW><lon>.tif — upper-left corner at (lat, lon),
    covering ``[lat - 10, lat]`` × ``[lon, lon + 10]`` in EPSG:4326.
    Pixel values: 1 = forest, 0 = non-forest.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.warp import Resampling, reproject, transform_bounds

from .data_loader import TileGrid, tile_grid

JRC_DIR = Path("data/external/jrc_gfc2020")
JRC_CRS = rasterio.crs.CRS.from_epsg(4326)


def _jrc_tile_name(lat_top: int, lon_left: int) -> str:
    ns = "N" if lat_top >= 0 else "S"
    ew = "E" if lon_left >= 0 else "W"
    return f"JRC_GFC2020_V3_{ns}{abs(lat_top):02d}_{ew}{abs(lon_left):02d}.tif"


def _tiles_covering_bbox_4326(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> list[tuple[int, int]]:
    """Enumerate (lat_top, lon_left) pairs covering a WGS84 bbox."""
    lat_top_min = int(math.ceil(lat_min / 10.0)) * 10
    lat_top_max = int(math.ceil(lat_max / 10.0)) * 10
    lon_left_min = int(math.floor(lon_min / 10.0)) * 10
    lon_left_max = int(math.floor(lon_max / 10.0)) * 10
    tiles: list[tuple[int, int]] = []
    for lat_top in range(lat_top_min, lat_top_max + 1, 10):
        for lon_left in range(lon_left_min, lon_left_max + 1, 10):
            tiles.append((lat_top, lon_left))
    return tiles


def _grid_bbox_4326(grid: TileGrid) -> tuple[float, float, float, float]:
    """Tile's UTM bbox transformed to EPSG:4326 (lon_min, lat_min, lon_max, lat_max)."""
    h, w = grid.shape
    left, top = grid.transform * (0, 0)
    right, bottom = grid.transform * (w, h)
    return transform_bounds(grid.crs, JRC_CRS, left, bottom, right, top, densify_pts=21)


@lru_cache(maxsize=64)
def load_jrc_on_tile_grid(tile_id: str, split: str = "train") -> np.ndarray | None:
    """JRC 2020 forest mask resampled onto the tile's canonical UTM grid.

    Returns (H, W) bool or ``None`` if no JRC tile covering this project tile
    exists on disk (e.g. ocean-only regions JRC does not publish, or the
    downloader hasn't run).
    """
    grid = tile_grid(tile_id, split)
    bbox4326 = _grid_bbox_4326(grid)
    needed = _tiles_covering_bbox_4326(*bbox4326)

    present: list[Path] = []
    for lat_top, lon_left in needed:
        p = JRC_DIR / _jrc_tile_name(lat_top, lon_left)
        if p.exists():
            present.append(p)

    if not present:
        return None

    # Open the JRC tile(s) and merge if more than one (project tile straddles a
    # 10° JRC boundary). ``rio_merge`` handles both cases with one call.
    srcs = [rasterio.open(p) for p in present]
    try:
        arr, src_transform = rio_merge(srcs, resampling=Resampling.nearest)
    finally:
        for s in srcs:
            s.close()
    # arr shape: (bands, H_src, W_src)
    src_band = arr[0].astype(np.uint8)

    # Reproject EPSG:4326 → tile UTM, nearest-neighbour (binary mask).
    dst = np.zeros(grid.shape, dtype=np.uint8)
    reproject(
        source=src_band,
        destination=dst,
        src_transform=src_transform,
        src_crs=JRC_CRS,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=Resampling.nearest,
    )
    # Treat any non-zero value as forest (defensive: V3 uses {0, 1}, but future
    # releases may introduce additional classes — anything non-zero = tree cover).
    return dst != 0


def jrc_coverage_on_disk(tile_id: str, split: str = "train") -> list[Path]:
    """Which JRC tiles on disk cover this project tile (for diagnostics)."""
    grid = tile_grid(tile_id, split)
    needed = _tiles_covering_bbox_4326(*_grid_bbox_4326(grid))
    return [JRC_DIR / _jrc_tile_name(lat, lon) for lat, lon in needed if (JRC_DIR / _jrc_tile_name(lat, lon)).exists()]
