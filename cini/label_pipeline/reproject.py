"""Raster reprojection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .constants import REPROJECT_NODATA


@dataclass(frozen=True)
class ReferenceGrid:
    shape: tuple[int, int]
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    profile: dict


def load_reference_grid(s2_ref_path: str | Path) -> ReferenceGrid:
    with rasterio.open(s2_ref_path) as src:
        profile = src.profile.copy()
        return ReferenceGrid(
            shape=src.shape,
            transform=src.transform,
            crs=src.crs,
            profile=profile,
        )


def reproject_raster_to_grid(
    raster_path: str | Path,
    reference_grid: ReferenceGrid,
    fill_value: float = REPROJECT_NODATA,
) -> tuple[np.ndarray, np.ndarray]:
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        source = src.read(1).astype(np.float32)
        data = np.full(reference_grid.shape, fill_value, dtype=np.float32)
        reproject(
            source=source,
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=reference_grid.transform,
            dst_crs=reference_grid.crs,
            src_nodata=src.nodata,
            dst_nodata=fill_value,
            resampling=Resampling.nearest,
        )

        coverage_source = np.ones(src.shape, dtype=np.uint8)
        coverage = np.zeros(reference_grid.shape, dtype=np.uint8)
        reproject(
            source=coverage_source,
            destination=coverage,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=reference_grid.transform,
            dst_crs=reference_grid.crs,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    return data, coverage
