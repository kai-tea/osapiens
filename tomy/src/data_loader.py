"""Tile-aware multimodal data loader.

Canonical per-tile grid = Sentinel-2 UTM grid (January of reference year).
Everything else (S1, AEF, labels) is reprojected onto that grid so feature
stacks and labels align pixel-for-pixel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

DATA_ROOT = Path("data/makeathon-challenge")
S2_DIR = DATA_ROOT / "sentinel-2"
S1_DIR = DATA_ROOT / "sentinel-1"
AEF_DIR = DATA_ROOT / "aef-embeddings"
LABELS_DIR = DATA_ROOT / "labels" / "train"
METADATA_DIR = DATA_ROOT / "metadata"

S2_BANDS = 12
AEF_BANDS = 64
REFERENCE_YEAR = 2020
REFERENCE_MONTH = 1


@dataclass(frozen=True)
class TileGrid:
    """Canonical raster grid for a tile (UTM, S2 resolution)."""

    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    shape: tuple[int, int]  # (H, W)

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]


def list_tiles(split: str = "train") -> list[str]:
    """Enumerate tile IDs present under sentinel-2/{split}/."""
    root = S2_DIR / split
    if not root.exists():
        return []
    tiles = set()
    for p in root.iterdir():
        if p.is_dir() and p.name.endswith("__s2_l2a"):
            tiles.add(p.name.removesuffix("__s2_l2a"))
    return sorted(tiles)


def _open(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return rasterio.open(path)


@lru_cache(maxsize=256)
def tile_grid(tile_id: str, split: str = "train") -> TileGrid:
    """Canonical grid for a tile: take the first available S2 scene."""
    tile_dir = S2_DIR / split / f"{tile_id}__s2_l2a"
    candidates = sorted(tile_dir.glob(f"{tile_id}__s2_l2a_*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No S2 scenes found for {tile_id} in {split}")
    # Prefer Jan 2020 if present, else first
    preferred = tile_dir / f"{tile_id}__s2_l2a_{REFERENCE_YEAR}_{REFERENCE_MONTH}.tif"
    path = preferred if preferred.exists() else candidates[0]
    with rasterio.open(path) as src:
        return TileGrid(transform=src.transform, crs=src.crs, shape=src.shape)


# ---------- Sentinel-2 ----------

_S2_FNAME_RE = re.compile(r"(?P<tile>.+?)__s2_l2a_(?P<year>\d{4})_(?P<month>\d{1,2})\.tif$")


def s2_paths(tile_id: str, year: int, split: str = "train") -> dict[int, Path]:
    """Map month -> path for S2 monthly scenes of a tile/year."""
    tile_dir = S2_DIR / split / f"{tile_id}__s2_l2a"
    out: dict[int, Path] = {}
    for p in tile_dir.glob(f"{tile_id}__s2_l2a_{year}_*.tif"):
        m = _S2_FNAME_RE.search(p.name)
        if m:
            out[int(m["month"])] = p
    return dict(sorted(out.items()))


def load_s2_month(path: Path) -> np.ndarray:
    """Return (12, H, W) float32 reflectance for one month."""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def s2_year_stack(
    tile_id: str,
    year: int,
    split: str = "train",
    bands: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Return (T, B, H, W) stack of monthly S2 scenes + the month list.

    `bands` is 1-indexed; defaults to all 12.
    """
    paths = s2_paths(tile_id, year, split)
    if not paths:
        g = tile_grid(tile_id, split)
        n_bands = S2_BANDS if bands is None else len(bands)
        return np.empty((0, n_bands, *g.shape), dtype=np.float32), []
    stacks = []
    months = []
    for month, path in paths.items():
        with rasterio.open(path) as src:
            arr = src.read(list(bands)) if bands else src.read()
        stacks.append(arr.astype(np.float32))
        months.append(month)
    return np.stack(stacks, axis=0), months


def s2_year_stats(tile_id: str, year: int, split: str = "train") -> np.ndarray:
    """Per-pixel yearly S2 statistics: (B*3, H, W) = [median, min, max] over months.

    Zero values are treated as missing (S2 nodata convention in this dataset).
    """
    stack, _ = s2_year_stack(tile_id, year, split)
    if stack.size == 0:
        g = tile_grid(tile_id, split)
        return np.zeros((S2_BANDS * 3, *g.shape), dtype=np.float32)
    masked = np.where(stack == 0, np.nan, stack)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(masked, axis=0)
        lo = np.nanmin(masked, axis=0)
        hi = np.nanmax(masked, axis=0)
    out = np.concatenate([med, lo, hi], axis=0)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


# ---------- Sentinel-1 ----------

_S1_FNAME_RE = re.compile(
    r"(?P<tile>.+?)__s1_rtc_(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<orbit>ascending|descending)\.tif$"
)


def s1_paths(tile_id: str, year: int, split: str = "train") -> list[tuple[int, str, Path]]:
    """Return list of (month, orbit, path) for a tile/year."""
    tile_dir = S1_DIR / split / f"{tile_id}__s1_rtc"
    out = []
    for p in tile_dir.glob(f"{tile_id}__s1_rtc_{year}_*.tif"):
        m = _S1_FNAME_RE.search(p.name)
        if m:
            out.append((int(m["month"]), m["orbit"], p))
    out.sort()
    return out


def _to_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear backscatter -> dB, mapping non-positive values to NaN."""
    return np.where(linear > 0, 10.0 * np.log10(np.clip(linear, 1e-8, None)), np.nan)


def s1_year_stats(tile_id: str, year: int, split: str = "train") -> np.ndarray:
    """Per-pixel yearly S1 VV (dB) stats: (4, H, W) = [median, mean, min, max].

    S1 is delivered at a coarser native grid than S2 (~30m vs 10m), so each
    scene is reprojected onto the canonical tile grid before stats are computed.
    """
    paths = s1_paths(tile_id, year, split)
    g = tile_grid(tile_id, split)
    if not paths:
        return np.zeros((4, *g.shape), dtype=np.float32)
    stacks = []
    for _, _, path in paths:
        with rasterio.open(path) as src:
            native = src.read(1).astype(np.float32)
            dst = np.zeros(g.shape, dtype=np.float32)
            reproject(
                source=native,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=g.transform,
                dst_crs=g.crs,
                resampling=Resampling.bilinear,
            )
        stacks.append(_to_db(dst))
    s = np.stack(stacks, axis=0)  # (T, H, W)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(s, axis=0)
            mean = np.nanmean(s, axis=0)
            lo = np.nanmin(s, axis=0)
            hi = np.nanmax(s, axis=0)
    out = np.stack([med, mean, lo, hi], axis=0)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


# ---------- AlphaEarth embeddings (EPSG:4326 → UTM) ----------


def aef_path(tile_id: str, year: int, split: str = "train") -> Path:
    return AEF_DIR / split / f"{tile_id}_{year}.tiff"


def load_aef_on_tile_grid(
    tile_id: str,
    year: int,
    split: str = "train",
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Load AEF embedding for (tile, year), reprojected onto the canonical UTM grid.

    Returns (64, H, W) float32.
    """
    g = tile_grid(tile_id, split)
    path = aef_path(tile_id, year, split)
    with rasterio.open(path) as src:
        src_data = src.read().astype(np.float32)  # (64, h, w) in EPSG:4326
        # AEF stores nodata as NaN on tile edges; sanitise before resampling so
        # bilinear interpolation near the edge doesn't smear NaN into valid pixels.
        src_data = np.nan_to_num(src_data, nan=0.0)
        dst = np.zeros((src_data.shape[0], *g.shape), dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=g.transform,
            dst_crs=g.crs,
            resampling=resampling,
        )
    return dst


def reproject_to_tile(
    src_path: Path,
    grid: TileGrid,
    resampling: Resampling = Resampling.nearest,
    band: int | None = None,
) -> np.ndarray:
    """Read a raster and reproject a single band (or all) onto the tile grid."""
    with rasterio.open(src_path) as src:
        data = src.read(band) if band else src.read()
        src_arr = data.astype(np.float32)
        if src_arr.ndim == 2:
            src_arr = src_arr[None]
        dst = np.zeros((src_arr.shape[0], *grid.shape), dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=resampling,
        )
    return dst[0] if band else dst


# ---------- Combined per-year feature stack ----------


def per_year_features(
    tile_id: str,
    year: int,
    split: str = "train",
    include_aef: bool = True,
    include_s2: bool = True,
    include_s1: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build the full feature stack for a (tile, year).

    Returns (features (C, H, W) float32, channel names).
    """
    parts: list[np.ndarray] = []
    names: list[str] = []

    if include_s2:
        s2 = s2_year_stats(tile_id, year, split)
        parts.append(s2)
        for stat in ("med", "min", "max"):
            for b in range(1, S2_BANDS + 1):
                names.append(f"s2_{stat}_B{b:02d}_{year}")

    if include_s1:
        s1 = s1_year_stats(tile_id, year, split)
        parts.append(s1)
        names.extend([f"s1_{s}_{year}" for s in ("med", "mean", "min", "max")])

    if include_aef:
        aef = load_aef_on_tile_grid(tile_id, year, split)
        parts.append(aef)
        names.extend([f"aef_{year}_b{i:02d}" for i in range(aef.shape[0])])

    feats = np.concatenate(parts, axis=0) if parts else np.zeros((0, 0, 0), dtype=np.float32)
    return feats, names


def year_over_year_features(
    tile_id: str,
    year: int,
    split: str = "train",
    baseline_year: int = REFERENCE_YEAR,
) -> tuple[np.ndarray, list[str]]:
    """Features designed for change detection: [feats_baseline, feats_year, aef_delta]."""
    base, base_names = per_year_features(tile_id, baseline_year, split)
    cur, cur_names = per_year_features(tile_id, year, split)
    # AEF delta (last AEF_BANDS channels of each)
    base_aef = base[-AEF_BANDS:]
    cur_aef = cur[-AEF_BANDS:]
    delta = cur_aef - base_aef
    delta_names = [f"aef_delta_{year}_vs_{baseline_year}_b{i:02d}" for i in range(AEF_BANDS)]
    feats = np.concatenate([base, cur, delta], axis=0)
    names = base_names + cur_names + delta_names
    return feats, names
