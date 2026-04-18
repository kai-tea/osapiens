"""Cini-free imagery change detection.

Direct physical signal of deforestation:
  NDVI drop + S1 backscatter drop, gated by Hansen forest-2020.

No weak labels. No Cini fusion. If it lost vegetation AND was forest in 2020,
it's deforestation. The only per-pixel supervision comes from Hansen GFC
(independent of the three weak sources Cini combines).

Sentinel-2 L2A band convention (per src.data.S2_BAND_NAMES b01..b12):
  index 3 = B04 (red, ~665 nm), index 7 = B08 (NIR, ~842 nm).
NDVI = (NIR - red) / (NIR + red).

Deforestation signatures:
  - NDVI: forest ≥ 0.6 before, ≤ 0.3 after → drop > 0.3 (large, unambiguous).
  - S1 VV: forest 0.08–0.15 linear before (≈ -11 to -8 dB), 0.04–0.06 after
    (bare soil / crop). A 40 %+ relative drop is a strong signal.
The functions here use *relative* drops so they work on both linear- and
dB-scaled inputs without needing to know which the dataset ships.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.data import (
    BASELINE_YEARS,
    CHANGE_YEARS,
    _parse_s1_sort_key,
    _parse_s2_year_month,
    _read_s1_orbit_window,
    _read_s2_window,
    resolve_tile_paths,
)

logger = logging.getLogger(__name__)

S2_RED_IDX = 3   # B04 in 1-indexed b01..b12
S2_NIR_IDX = 7   # B08

# Pixel-level thresholds. Tuned for sensitivity on the ungated leaderboard
# framing: we'd rather over-cover and let Hansen gate clean up than miss
# positives entirely.
NDVI_BASELINE_MIN = 0.50     # baseline NDVI to call a pixel "was forest"
NDVI_DROP_MIN = 0.25         # absolute drop from baseline to flag change
S1_REL_DROP_MIN = 0.30       # relative VV drop: (base - curr)/base
EPS = 1e-6


def _ndvi(stack: np.ndarray) -> np.ndarray:
    """Return (T, H, W) NDVI from a (T, 12, H, W) S2 stack."""
    red = stack[:, S2_RED_IDX]
    nir = stack[:, S2_NIR_IDX]
    denom = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(np.abs(denom) > EPS, (nir - red) / denom, np.nan)
    return ndvi.astype(np.float32)


def ndvi_change_mask(
    tile_id: str,
    manifest,
    baseline_threshold: float = NDVI_BASELINE_MIN,
    drop_threshold: float = NDVI_DROP_MIN,
) -> tuple[np.ndarray, dict]:
    """(H, W) boolean mask where NDVI shows a large drop from a forested baseline.

    True where ``NDVI_mean(2020) >= baseline_threshold`` AND
    ``NDVI_mean(2020) - NDVI_mean(2023-2025) >= drop_threshold``.
    Pixels with fewer than 2 valid observations in a window are False.
    """
    paths = resolve_tile_paths(tile_id, manifest=manifest)
    baseline_paths = [p for p in paths.s2_monthly if _parse_s2_year_month(p)[0] in BASELINE_YEARS]
    change_paths = [p for p in paths.s2_monthly if _parse_s2_year_month(p)[0] in CHANGE_YEARS]

    import rasterio

    with rasterio.open(paths.s2_ref_path) as src:
        ref_profile = src.profile.copy()

    if not baseline_paths or not change_paths:
        H, W = ref_profile["height"], ref_profile["width"]
        return np.zeros((H, W), dtype=bool), {"skipped": "missing S2 window"}

    logger.info("tile %s: reading %d baseline + %d change S2 rasters", tile_id,
                len(baseline_paths), len(change_paths))
    base_stack, _ = _read_s2_window(baseline_paths, ref_profile)
    curr_stack, _ = _read_s2_window(change_paths, ref_profile)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_ndvi = np.nanmean(_ndvi(base_stack), axis=0)
        curr_ndvi = np.nanmean(_ndvi(curr_stack), axis=0)

    delta = base_ndvi - curr_ndvi
    mask = (base_ndvi >= baseline_threshold) & (delta >= drop_threshold)
    mask = np.where(np.isnan(base_ndvi) | np.isnan(curr_ndvi), False, mask)

    stats = {
        "n_baseline_rasters": len(baseline_paths),
        "n_change_rasters": len(change_paths),
        "baseline_ndvi_mean": float(np.nanmean(base_ndvi)),
        "current_ndvi_mean": float(np.nanmean(curr_ndvi)),
        "mean_delta": float(np.nanmean(delta)),
        "positive_fraction": float(mask.mean()),
    }
    return mask.astype(bool), stats


def s1_backscatter_drop_mask(
    tile_id: str,
    manifest,
    rel_drop_threshold: float = S1_REL_DROP_MIN,
) -> tuple[np.ndarray, dict]:
    """(H, W) boolean mask where VV backscatter (ascending orbit) dropped.

    True where ``(VV_base - VV_curr) / VV_base >= rel_drop_threshold``.
    Uses the ascending orbit mean; descending is redundant and slower.
    """
    paths = resolve_tile_paths(tile_id, manifest=manifest)
    asc_paths = [p for p in paths.s1_monthly if p.stem.endswith("_ascending")]
    baseline = [p for p in asc_paths if _parse_s1_sort_key(p)[0] in BASELINE_YEARS]
    change = [p for p in asc_paths if _parse_s1_sort_key(p)[0] in CHANGE_YEARS]

    import rasterio

    with rasterio.open(paths.s2_ref_path) as src:
        ref_profile = src.profile.copy()

    if not baseline or not change:
        H, W = ref_profile["height"], ref_profile["width"]
        return np.zeros((H, W), dtype=bool), {"skipped": "missing S1 window"}

    logger.info("tile %s: reading %d baseline + %d change S1 rasters", tile_id,
                len(baseline), len(change))
    base_stack, _ = _read_s1_orbit_window(baseline, ref_profile)
    curr_stack, _ = _read_s1_orbit_window(change, ref_profile)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base_vv = np.nanmean(base_stack[:, 0], axis=0)
        curr_vv = np.nanmean(curr_stack[:, 0], axis=0)
    rel_drop = (base_vv - curr_vv) / (np.abs(base_vv) + EPS)
    mask = rel_drop >= rel_drop_threshold
    mask = np.where(np.isnan(base_vv) | np.isnan(curr_vv), False, mask)

    stats = {
        "n_baseline_rasters": len(baseline),
        "n_change_rasters": len(change),
        "baseline_vv_mean": float(np.nanmean(base_vv)),
        "current_vv_mean": float(np.nanmean(curr_vv)),
        "mean_rel_drop": float(np.nanmean(rel_drop)),
        "positive_fraction": float(mask.mean()),
    }
    return mask.astype(bool), stats


def imagery_change_binary(
    tile_id: str,
    manifest,
    require_s1_agreement: bool = False,
) -> tuple[np.ndarray, dict, dict]:
    """Combine NDVI-drop and Hansen-forest, optionally AND with S1 drop.

    Returns ``(binary, ref_profile, stats)``. Binary is ``False`` outside
    Hansen forest-2020 so downstream gate/refusal rules still apply; the
    Hansen gate is already baked in here — callers shouldn't re-apply it.
    """
    from src.masks import forest_mask_2020

    import rasterio

    paths = resolve_tile_paths(tile_id, manifest=manifest)
    with rasterio.open(paths.s2_ref_path) as src:
        ref_profile = src.profile.copy()

    ndvi_mask, ndvi_stats = ndvi_change_mask(tile_id, manifest)
    hansen = forest_mask_2020(ref_profile)
    binary = ndvi_mask & hansen

    combined_stats = {"ndvi": ndvi_stats, "require_s1_agreement": require_s1_agreement}
    if require_s1_agreement:
        s1_mask, s1_stats = s1_backscatter_drop_mask(tile_id, manifest)
        binary = binary & s1_mask
        combined_stats["s1"] = s1_stats

    combined_stats["final_positive_fraction"] = float(binary.mean())
    return binary.astype(bool), ref_profile, combined_stats
