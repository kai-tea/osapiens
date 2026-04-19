"""Small raster I/O helpers for Mark 2 prediction export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

try:
    from .io import read_raster
except ImportError:
    from io import read_raster


def load_reference_meta(path: Path) -> dict:
    """Read raster metadata needed to write aligned outputs."""
    _, meta = read_raster(path)
    return meta


def write_raster(path: Path, array: np.ndarray, reference_meta: dict, dtype: str, nodata: float | int | None = None) -> None:
    """Write a single-band GeoTIFF using a reference grid."""
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
