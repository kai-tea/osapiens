"""Per-tile data loading and feature extraction for the Kaite baseline.

Consumes the raw Sentinel-1, Sentinel-2, and AlphaEarth Foundations imagery
together with Cini's weak-label artifacts to produce per-pixel feature
vectors on each tile's Sentinel-2 UTM grid. The feature schema is a temporal
aggregation of the monthly stacks into two windows — ``baseline`` (2020) and
``change`` (2021-2025) — plus their delta, concatenated with baseline/latest
AEF embeddings.

Feature layout (402 dims total):

- S1 VV per orbit (ascending, descending): 2 streams × 5 stats × 3 windows = 30
- S2 per band (B01-B12): 12 bands × 5 stats × 3 windows = 180
- AEF (64 embedding dims): 64 × (baseline, latest, delta) = 192

Where ``stats = (mean, std, min, max, linear-trend)`` and
``windows = (baseline, change, delta=change-baseline)``.

Outputs per tile:

- ``artifacts/features_v1/{tile_id}.parquet`` — one row per sampled pixel with
  columns ``(tile_id, row, col, <402 feature names>, soft_target,
  sample_weight, hard_label, event_year)``.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "makeathon-challenge"
LABEL_ROOT = REPO_ROOT / "artifacts" / "labels_v1"
MODEL_INPUTS_ROOT = REPO_ROOT / "artifacts" / "model_inputs_v1"
FEATURES_ROOT = REPO_ROOT / "artifacts" / "features_v1"
TILE_MANIFEST_PATH = MODEL_INPUTS_ROOT / "tile_manifest.csv"

BASELINE_YEARS = frozenset({2020})
CHANGE_YEARS = frozenset({2021, 2022, 2023, 2024, 2025})
WINDOWS = ("baseline", "change", "delta")
STATS = ("mean", "std", "min", "max", "trend")
S2_BAND_NAMES = tuple(f"b{i:02d}" for i in range(1, 13))
S1_ORBITS = ("ascending", "descending")
AEF_DIMS = 64
ABSENT_DAYS_SENTINEL = -1
LABELPACK_NODATA = -9999.0


def feature_names() -> list[str]:
    """Return the canonical ordered feature names (402 entries).

    Order: S1 (orbits × stats × windows), then S2 (bands × stats × windows),
    then AEF (dims × {baseline, latest, delta}). Stable across runs so that
    training and inference use the same columns.
    """
    names: list[str] = []
    for orbit in S1_ORBITS:
        for stat in STATS:
            for window in WINDOWS:
                names.append(f"s1_{orbit[:3]}_{stat}_{window}")
    for band in S2_BAND_NAMES:
        for stat in STATS:
            for window in WINDOWS:
                names.append(f"s2_{band}_{stat}_{window}")
    for dim in range(AEF_DIMS):
        for window in ("baseline", "latest", "delta"):
            names.append(f"aef_e{dim:02d}_{window}")
    return names


@dataclass(frozen=True)
class TilePaths:
    """Resolved filesystem paths and metadata for a single tile."""

    tile_id: str
    split: str
    region_id: str
    fold_id: int | None
    s2_ref_path: Path
    s2_monthly: tuple[Path, ...]
    s1_monthly: tuple[Path, ...]
    aef_annual: tuple[Path, ...]
    labelpack_path: Path | None


def load_tile_manifest(manifest_path: Path = TILE_MANIFEST_PATH) -> pd.DataFrame:
    """Load Cini's tile manifest CSV."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Tile manifest not found at {manifest_path}. "
            "Regenerate Cini's model_inputs_v1 artifacts first."
        )
    return pd.read_csv(manifest_path)


def resolve_tile_paths(tile_id: str, manifest: pd.DataFrame | None = None) -> TilePaths:
    """Resolve all modality paths for a tile from Cini's tile manifest."""
    manifest = manifest if manifest is not None else load_tile_manifest()
    rows = manifest[manifest["tile_id"] == tile_id]
    if rows.empty:
        raise KeyError(f"Tile '{tile_id}' not found in manifest")
    row = rows.iloc[0]

    s2_dir = REPO_ROOT / row["s2_dir_path"]
    s1_dir = REPO_ROOT / row["s1_dir_path"]
    s2_monthly = tuple(sorted(s2_dir.glob("*.tif"), key=_parse_s2_year_month))
    s1_monthly = tuple(sorted(s1_dir.glob("*.tif"), key=_parse_s1_sort_key))

    split = row["split"]
    aef_dir = DATA_ROOT / "aef-embeddings" / split
    aef_annual = tuple(sorted(aef_dir.glob(f"{tile_id}_*.tiff"), key=_parse_aef_year))

    labelpack_path = LABEL_ROOT / "tiles" / f"{tile_id}_labelpack.tif"
    labelpack_path = labelpack_path if labelpack_path.exists() else None

    fold_id = int(row["fold_id"]) if pd.notna(row["fold_id"]) else None

    return TilePaths(
        tile_id=tile_id,
        split=split,
        region_id=str(row["region_id"]),
        fold_id=fold_id,
        s2_ref_path=REPO_ROOT / row["s2_ref_path"],
        s2_monthly=s2_monthly,
        s1_monthly=s1_monthly,
        aef_annual=aef_annual,
        labelpack_path=labelpack_path,
    )


def _parse_s2_year_month(path: Path) -> tuple[int, int]:
    stem = path.stem
    parts = stem.rsplit("_", 2)
    return int(parts[-2]), int(parts[-1])


def _parse_s1_sort_key(path: Path) -> tuple[int, int, int]:
    stem = path.stem
    parts = stem.rsplit("_", 3)
    year, month, orbit = int(parts[-3]), int(parts[-2]), parts[-1]
    orbit_rank = 0 if orbit == "ascending" else 1
    return year, month, orbit_rank


def _parse_aef_year(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _month_index(year: int, month: int) -> int:
    """Months elapsed since 2020-01 as a float-compatible index."""
    return (year - 2020) * 12 + (month - 1)


def _warp_to_ref(
    path: Path,
    ref_profile: dict,
    resampling: Resampling,
    nodata_fill: float | None,
) -> np.ndarray:
    """Open a raster and reproject it onto the S2 reference grid.

    Returns ``(B, H_ref, W_ref)`` float32 with nodata converted to NaN.
    Works for rasters already in the ref CRS (resamples if resolution
    differs) and for rasters in a different CRS (e.g. AEF EPSG:4326).
    """
    with rasterio.open(path) as src:
        src_data = src.read().astype(np.float32)
        dst = np.full(
            (src.count, ref_profile["height"], ref_profile["width"]),
            np.nan,
            dtype=np.float32,
        )
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    if nodata_fill is not None:
        dst[dst == nodata_fill] = np.nan
    return dst


def _read_band_stack(
    paths: Iterable[Path],
    ref_profile: dict,
    resampling: Resampling,
    nodata_fill: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read rasters, warp to the reference grid, and stack on axis 0.

    Returns ``(stack, times)`` where ``stack`` has shape
    ``(T, B, H_ref, W_ref)`` with nodata converted to NaN and ``times``
    is a float month index relative to 2020-01.
    """
    arrays: list[np.ndarray] = []
    times: list[float] = []
    for p in paths:
        arrays.append(_warp_to_ref(p, ref_profile, resampling, nodata_fill))
        if "_s2_l2a_" in p.name:
            y, m = _parse_s2_year_month(p)
        else:
            parts = p.stem.rsplit("_", 3)
            y, m = int(parts[-3]), int(parts[-2])
        times.append(float(_month_index(y, m)))
    stack = np.stack(arrays, axis=0)
    times_arr = np.asarray(times, dtype=np.float32)
    return stack, times_arr


def _linear_trend_per_band(stack: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-pixel linear regression slope of ``stack`` over time axis.

    ``stack`` has shape ``(T, H, W)`` and may contain NaN for missing
    observations. Pixels without ≥2 valid observations receive NaN.
    """
    t_b = t.reshape(-1, 1, 1)
    t_masked = np.where(np.isnan(stack), np.nan, t_b)
    t_mean = np.nanmean(t_masked, axis=0)
    x_mean = np.nanmean(stack, axis=0)
    t_c = t_masked - t_mean
    x_c = stack - x_mean
    num = np.nansum(t_c * x_c, axis=0)
    den = np.nansum(t_c * t_c, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(den > 0, num / den, np.nan)
    return slope.astype(np.float32)


def _window_stats(stack: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute (mean, std, min, max, trend) for a ``(T, B, H, W)`` stack.

    Returns array of shape ``(B, 5, H, W)`` in the canonical stat order.
    All-NaN pixels (no valid observation in the window) propagate as NaN
    — ``lightgbm`` treats them natively as missing.
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)
        mn = np.nanmin(stack, axis=0)
        mx = np.nanmax(stack, axis=0)
        B = stack.shape[1]
        trend = np.empty_like(mean)
        for b in range(B):
            trend[b] = _linear_trend_per_band(stack[:, b], t)
    return np.stack([mean, std, mn, mx, trend], axis=1).astype(np.float32)


def _empty_stats(bands: int, height: int, width: int) -> np.ndarray:
    return np.full((bands, len(STATS), height, width), np.nan, dtype=np.float32)


def _read_s2_window(paths: list[Path], ref_profile: dict) -> tuple[np.ndarray, np.ndarray]:
    if not paths:
        return np.empty((0, 12, 0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return _read_band_stack(paths, ref_profile, Resampling.bilinear, nodata_fill=0.0)


def _read_s1_orbit_window(paths: list[Path], ref_profile: dict) -> tuple[np.ndarray, np.ndarray]:
    if not paths:
        return np.empty((0, 1, 0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return _read_band_stack(paths, ref_profile, Resampling.bilinear, nodata_fill=None)


def _stats_block(
    paths: list[Path],
    reader,
    ref_profile: dict,
    bands: int,
) -> np.ndarray:
    height = ref_profile["height"]
    width = ref_profile["width"]
    if not paths:
        return _empty_stats(bands, height, width)
    stack, t = reader(paths, ref_profile)
    return _window_stats(stack, t)


def _extract_s2_features(
    paths: TilePaths,
    ref_profile: dict,
) -> tuple[np.ndarray, list[str]]:
    baseline_paths = [p for p in paths.s2_monthly if _parse_s2_year_month(p)[0] in BASELINE_YEARS]
    change_paths = [p for p in paths.s2_monthly if _parse_s2_year_month(p)[0] in CHANGE_YEARS]
    baseline = _stats_block(baseline_paths, _read_s2_window, ref_profile, 12)
    change = _stats_block(change_paths, _read_s2_window, ref_profile, 12)
    delta = change - baseline
    features = np.stack([baseline, change, delta], axis=2)  # (12, 5, 3, H, W)
    H, W = ref_profile["height"], ref_profile["width"]
    features = features.reshape(12 * len(STATS) * len(WINDOWS), H, W)
    names = [
        f"s2_{band}_{stat}_{window}"
        for band in S2_BAND_NAMES
        for stat in STATS
        for window in WINDOWS
    ]
    return features, names


def _extract_s1_features(
    paths: TilePaths,
    ref_profile: dict,
) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    H, W = ref_profile["height"], ref_profile["width"]
    for orbit in S1_ORBITS:
        orbit_paths = [p for p in paths.s1_monthly if p.stem.endswith(f"_{orbit}")]
        baseline_paths = [p for p in orbit_paths if _parse_s1_sort_key(p)[0] in BASELINE_YEARS]
        change_paths = [p for p in orbit_paths if _parse_s1_sort_key(p)[0] in CHANGE_YEARS]
        baseline = _stats_block(baseline_paths, _read_s1_orbit_window, ref_profile, 1)
        change = _stats_block(change_paths, _read_s1_orbit_window, ref_profile, 1)
        delta = change - baseline
        block = np.stack([baseline, change, delta], axis=2)  # (1, 5, 3, H, W)
        blocks.append(block.reshape(len(STATS) * len(WINDOWS), H, W))
        names.extend(
            f"s1_{orbit[:3]}_{stat}_{window}" for stat in STATS for window in WINDOWS
        )
    return np.concatenate(blocks, axis=0), names


def _extract_aef_features(
    paths: TilePaths,
    ref_profile: dict,
) -> tuple[np.ndarray, list[str]]:
    if not paths.aef_annual:
        raise FileNotFoundError(f"No AEF rasters for tile {paths.tile_id}")
    aef_by_year: dict[int, Path] = {_parse_aef_year(p): p for p in paths.aef_annual}
    baseline_year = 2020 if 2020 in aef_by_year else min(aef_by_year)
    latest_year = max(aef_by_year)
    baseline = _warp_to_ref(aef_by_year[baseline_year], ref_profile, Resampling.nearest, None)
    latest = _warp_to_ref(aef_by_year[latest_year], ref_profile, Resampling.nearest, None)
    delta = latest - baseline
    per_window = np.stack([baseline, latest, delta], axis=1)  # (64, 3, H, W)
    H, W = ref_profile["height"], ref_profile["width"]
    features = per_window.reshape(AEF_DIMS * 3, H, W)
    names = [
        f"aef_e{dim:02d}_{window}"
        for dim in range(AEF_DIMS)
        for window in ("baseline", "latest", "delta")
    ]
    return features, names


def _read_ref_profile(s2_ref_path: Path) -> dict:
    with rasterio.open(s2_ref_path) as src:
        profile = src.profile.copy()
    return profile


def _read_labelpack(path: Path) -> dict[str, np.ndarray]:
    """Load Cini's 18-band labelpack and return bands keyed by name."""
    from rasterio.enums import MaskFlags  # noqa: F401

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    names = (
        "train_mask",
        "hard_label",
        "seed_mask",
        "soft_target",
        "sample_weight",
        "obs_count",
        "radd_obs",
        "radd_alert",
        "radd_conf",
        "radd_days",
        "gladl_obs",
        "gladl_alert",
        "gladl_conf",
        "gladl_days",
        "glads2_obs",
        "glads2_alert",
        "glads2_conf",
        "glads2_days",
    )
    return {n: data[i] for i, n in enumerate(names)}


def _compute_event_year(labelpack: dict[str, np.ndarray]) -> np.ndarray:
    """Earliest positive alert year across sources. ``-1`` if no event."""
    days_stack = np.stack(
        [labelpack["radd_days"], labelpack["gladl_days"], labelpack["glads2_days"]], axis=0
    )
    present = days_stack >= 0  # ABSENT_DAYS_SENTINEL == -1
    masked = np.where(present, days_stack, np.inf)
    min_days = masked.min(axis=0)
    has_event = np.isfinite(min_days)
    with np.errstate(invalid="ignore"):
        year = np.where(has_event, 2020 + np.floor(min_days / 365.25), -1)
    return year.astype(np.int16)


def build_tile_features(
    paths: TilePaths,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray], dict]:
    """Compute the full per-pixel feature stack for a tile.

    Returns ``(features, names, labels, ref_profile)`` where ``features``
    has shape ``(F, H, W)`` and ``labels`` is a dict of per-pixel
    auxiliary arrays (soft_target, sample_weight, hard_label, event_year,
    train_mask, region_id). For test tiles with no labelpack the labels
    dict contains only ``train_mask`` set to all ones.
    """
    ref_profile = _read_ref_profile(paths.s2_ref_path)
    H, W = ref_profile["height"], ref_profile["width"]

    logger.info("tile %s: extracting S1 features", paths.tile_id)
    s1_feats, s1_names = _extract_s1_features(paths, ref_profile)
    logger.info("tile %s: extracting S2 features", paths.tile_id)
    s2_feats, s2_names = _extract_s2_features(paths, ref_profile)
    logger.info("tile %s: extracting AEF features", paths.tile_id)
    aef_feats, aef_names = _extract_aef_features(paths, ref_profile)

    features = np.concatenate([s1_feats, s2_feats, aef_feats], axis=0)
    names = s1_names + s2_names + aef_names

    assert names == feature_names(), (
        f"feature name ordering drifted; got {len(names)} names but schema expects "
        f"{len(feature_names())}"
    )

    labels: dict[str, np.ndarray] = {}
    if paths.labelpack_path is not None:
        pack = _read_labelpack(paths.labelpack_path)
        labels["train_mask"] = (pack["train_mask"] == 1.0).astype(np.uint8)
        labels["soft_target"] = pack["soft_target"].astype(np.float32)
        labels["sample_weight"] = pack["sample_weight"].astype(np.float32)
        labels["hard_label"] = pack["hard_label"].astype(np.int16)
        labels["event_year"] = _compute_event_year(pack)
    else:
        labels["train_mask"] = np.ones((H, W), dtype=np.uint8)

    return features, names, labels, ref_profile


def _stratified_pixel_sample(
    train_mask: np.ndarray,
    soft_target: np.ndarray | None,
    max_pixels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a flat index array of sampled pixels inside ``train_mask``.

    For train tiles (``soft_target`` present) samples are stratified by
    the soft-target binarisation at 0.5 so the positive rate is preserved
    even after aggressive downsampling. For test tiles (``soft_target``
    is ``None``) samples are uniform.
    """
    flat_mask = train_mask.reshape(-1).astype(bool)
    eligible = np.flatnonzero(flat_mask)
    if eligible.size <= max_pixels or max_pixels <= 0:
        return eligible

    if soft_target is None:
        return rng.choice(eligible, size=max_pixels, replace=False)

    y = (soft_target.reshape(-1)[eligible] >= 0.5)
    pos_idx = eligible[y]
    neg_idx = eligible[~y]
    if pos_idx.size == 0 or neg_idx.size == 0:
        return rng.choice(eligible, size=max_pixels, replace=False)
    n_pos = min(pos_idx.size, max_pixels // 2)
    n_neg = max_pixels - n_pos
    n_neg = min(n_neg, neg_idx.size)
    n_pos = min(pos_idx.size, max_pixels - n_neg)
    chosen = np.concatenate(
        [
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=n_neg, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return chosen


def build_tile_dataset(
    paths: TilePaths,
    max_pixels_per_tile: int | None,
    seed: int,
) -> pd.DataFrame:
    """Materialise a sampled per-pixel dataframe for one tile.

    Rows are the tile's eligible pixels — ``train_mask == 1`` for train
    tiles, all pixels for test tiles. Columns are ``(tile_id, row, col,
    region_id, fold_id, <402 feature names>, soft_target, sample_weight,
    hard_label, event_year)``. ``soft_target``, ``sample_weight``,
    ``hard_label``, ``event_year`` are absent for test tiles.
    """
    features, names, labels, _ = build_tile_features(paths)
    F, H, W = features.shape

    rng = np.random.default_rng(seed)
    soft = labels.get("soft_target")
    sample_idx = _stratified_pixel_sample(
        labels["train_mask"], soft, max_pixels_per_tile or -1, rng
    )

    rows, cols = np.unravel_index(sample_idx, (H, W))
    feature_matrix = features.reshape(F, -1)[:, sample_idx].T.astype(np.float32)

    df = pd.DataFrame(feature_matrix, columns=names)
    df.insert(0, "col", cols.astype(np.int32))
    df.insert(0, "row", rows.astype(np.int32))
    df.insert(0, "fold_id", paths.fold_id if paths.fold_id is not None else -1)
    df.insert(0, "region_id", paths.region_id)
    df.insert(0, "tile_id", paths.tile_id)

    if soft is not None:
        df["soft_target"] = soft.reshape(-1)[sample_idx].astype(np.float32)
        df["sample_weight"] = labels["sample_weight"].reshape(-1)[sample_idx].astype(np.float32)
        df["hard_label"] = labels["hard_label"].reshape(-1)[sample_idx].astype(np.int16)
        df["event_year"] = labels["event_year"].reshape(-1)[sample_idx].astype(np.int16)
    return df


def write_tile_features(
    tile_id: str,
    max_pixels_per_tile: int | None,
    seed: int,
    output_root: Path = FEATURES_ROOT,
    manifest: pd.DataFrame | None = None,
) -> Path:
    """Compute and persist one tile's sampled feature parquet."""
    paths = resolve_tile_paths(tile_id, manifest=manifest)
    df = build_tile_dataset(paths, max_pixels_per_tile, seed)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{tile_id}.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("tile %s: wrote %d rows -> %s", tile_id, len(df), output_path)
    return output_path


def load_tile_features(tile_id: str, features_root: Path = FEATURES_ROOT) -> pd.DataFrame:
    """Read back a tile's cached feature parquet."""
    path = features_root / f"{tile_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features not found for tile {tile_id} at {path}")
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiles",
        nargs="*",
        help="Tile IDs to process (default: all train + test tiles in the manifest)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "test"],
        choices=["train", "test"],
        help="Which splits to process when --tiles is not supplied",
    )
    parser.add_argument(
        "--max-pixels-per-tile",
        type=int,
        default=100_000,
        help="Subsample cap per tile (set to 0 for all pixels, use with care)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FEATURES_ROOT,
        help="Destination directory for per-tile parquet files",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    manifest = load_tile_manifest()
    if args.tiles:
        tile_ids = list(args.tiles)
    else:
        tile_ids = manifest.loc[manifest["split"].isin(args.splits), "tile_id"].tolist()

    logger.info("processing %d tiles: %s", len(tile_ids), tile_ids)
    max_px = args.max_pixels_per_tile if args.max_pixels_per_tile > 0 else None
    for tile_id in tile_ids:
        write_tile_features(
            tile_id,
            max_pixels_per_tile=max_px,
            seed=args.seed,
            output_root=args.output_root,
            manifest=manifest,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
