"""Forest-in-2020 masks from Hansen Global Forest Change (v1.11).

Downloads the two Hansen GFC layers we need — ``treecover2000`` and
``lossyear`` — from the public Earth Engine bucket, caches them under
``artifacts/masks_v1/hansen/``, and builds a per-S2-tile boolean
raster of "pixel was forest at end of 2020" on the reference S2 UTM
grid.

Forest-2020 rule (default threshold 25% canopy cover):

    forest_2020 = (treecover2000 >= threshold)
                  AND (lossyear == 0 OR lossyear >= 21)

``lossyear`` encodes the year of first detected loss minus 2000
(``lossyear == 20`` → loss during 2020; we exclude those as
not-forest-at-end-of-2020, keeping pixels lost in 2021+ and pixels
never lost).

NDVI fallback: if Hansen cannot be reached, callers should switch to
the documented NDVI-on-2020-S2 threshold and note the fallback in
their report. See ``#v2`` priority notes in ``reports/baseline_v1.md``.
"""

from __future__ import annotations

import logging
import math
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
MASKS_ROOT = REPO_ROOT / "artifacts" / "masks_v1" / "hansen"

HANSEN_BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11"
HANSEN_LAYERS = ("treecover2000", "lossyear")
LOSSYEAR_2020 = 20
DEFAULT_TREECOVER_THRESHOLD = 25


def hansen_tile_key(bounds_4326: tuple[float, float, float, float]) -> str:
    """Return the Hansen 10°×10° tile key (upper-left corner) for bounds.

    Hansen tiles are indexed by their top-left lat (e.g. ``10N``) and
    left-edge lon (e.g. ``080W``) at 10° resolution. A tile at key
    ``10N_080W`` covers latitudes ``0N-10N`` and longitudes ``80W-70W``.
    """
    minx, miny, maxx, maxy = bounds_4326
    top = math.ceil(maxy / 10) * 10
    left = math.floor(minx / 10) * 10
    lat_token = f"{abs(top):02d}{'N' if top >= 0 else 'S'}"
    lon_token = f"{abs(left):03d}{'E' if left >= 0 else 'W'}"
    return f"{lat_token}_{lon_token}"


def _hansen_url(layer: str, tile_key: str) -> str:
    return f"{HANSEN_BASE_URL}/Hansen_GFC-2023-v1.11_{layer}_{tile_key}.tif"


def _cached_path(layer: str, tile_key: str) -> Path:
    return MASKS_ROOT / f"{layer}_{tile_key}.tif"


def download_hansen(layer: str, tile_key: str, force: bool = False) -> Path:
    """Download one Hansen GFC layer tile into the local cache.

    Returns the cached path. Re-uses the on-disk copy unless ``force``.
    Raises on HTTP failure.
    """
    if layer not in HANSEN_LAYERS:
        raise ValueError(f"unsupported Hansen layer: {layer}")
    path = _cached_path(layer, tile_key)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = _hansen_url(layer, tile_key)
    logger.info("downloading %s -> %s", url, path)
    with urllib.request.urlopen(url, timeout=600) as resp, open(path, "wb") as fh:
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    return path


@lru_cache(maxsize=32)
def _ensure_hansen_pair(tile_key: str) -> tuple[Path, Path]:
    """Return cached paths to (treecover2000, lossyear) for one Hansen tile."""
    tc = download_hansen("treecover2000", tile_key)
    ly = download_hansen("lossyear", tile_key)
    return tc, ly


def _warp_layer_to_ref(path: Path, ref_profile: dict) -> np.ndarray:
    with rasterio.open(path) as src:
        src_data = src.read(1)
        dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=src_data.dtype)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=0,
        )
    return dst


def forest_mask_2020(
    s2_ref_profile: dict,
    treecover_threshold: int = DEFAULT_TREECOVER_THRESHOLD,
) -> np.ndarray:
    """Build a boolean ``(H, W)`` "forest at end of 2020" mask on the
    reference S2 UTM grid.

    Downloads (or reuses) the appropriate Hansen GFC tile(s), warps
    ``treecover2000`` and ``lossyear`` onto the reference grid (nearest
    neighbour), and applies the forest-2020 rule.
    """
    bounds_4326 = transform_bounds(
        s2_ref_profile["crs"],
        "EPSG:4326",
        *rasterio.transform.array_bounds(
            s2_ref_profile["height"], s2_ref_profile["width"], s2_ref_profile["transform"]
        ),
    )
    tile_key = hansen_tile_key(bounds_4326)
    tc_path, ly_path = _ensure_hansen_pair(tile_key)

    tc = _warp_layer_to_ref(tc_path, s2_ref_profile)
    ly = _warp_layer_to_ref(ly_path, s2_ref_profile)
    still_forest_at_eoy_2020 = (ly == 0) | (ly > LOSSYEAR_2020)
    return (tc >= treecover_threshold) & still_forest_at_eoy_2020
