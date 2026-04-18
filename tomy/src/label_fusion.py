"""Weak-label decoding and fusion.

Three noisy sources — RADD, GLAD-L, GLAD-S2 — are decoded into a common
(alert_binary, alert_date, confidence) representation, reprojected onto the
canonical tile grid, and fused.

Challenge constraint: only events after 2020-01-01 count, and only on pixels
that were forest in 2020.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling

from .data_loader import (
    LABELS_DIR,
    TileGrid,
    reproject_to_tile,
    s2_year_stack,
    tile_grid,
)

POST2020 = date(2020, 1, 1)
RADD_EPOCH = date(2014, 12, 31)
GLADS2_EPOCH = date(2019, 1, 1)


@dataclass
class DecodedLabels:
    """Aligned to the canonical tile grid."""

    alert: np.ndarray  # uint8 binary (0/1): any alert
    date_days: np.ndarray  # int32 days since 1970-01-01 (0 = no alert)
    confidence: np.ndarray  # float32 in [0, 1]
    source: str


def _days_since_epoch(d: date) -> int:
    return (d - date(1970, 1, 1)).days


# ---------- RADD ----------


def decode_radd(raster_val: np.ndarray) -> DecodedLabels:
    """Decode the RADD integer encoding.

    leading digit 2 = low-conf, 3 = high-conf; remaining digits = days since 2014-12-31.
    0 = no alert.
    """
    v = raster_val.astype(np.int64)
    alert = (v > 0).astype(np.uint8)

    leading = v // 10000  # 2 or 3 (or 0)
    day_offset = v % 10000
    valid = alert.astype(bool)

    confidence = np.where(leading == 3, 1.0, np.where(leading == 2, 0.5, 0.0)).astype(np.float32)

    epoch_days = _days_since_epoch(RADD_EPOCH)
    date_days = np.zeros_like(v, dtype=np.int32)
    date_days[valid] = epoch_days + day_offset[valid].astype(np.int32)

    return DecodedLabels(alert=alert, date_days=date_days, confidence=confidence, source="radd")


# ---------- GLAD-L (Landsat) ----------


def decode_gladl(alert: np.ndarray, alert_date: np.ndarray, year_2digit: int) -> DecodedLabels:
    """alert: 0/2/3. alert_date: day-of-year within 20YY."""
    full_year = 2000 + year_2digit
    a = alert.astype(np.int32)
    has_alert = (a > 0).astype(np.uint8)

    confidence = np.where(a == 3, 1.0, np.where(a == 2, 0.5, 0.0)).astype(np.float32)

    epoch_days = _days_since_epoch(date(full_year, 1, 1))
    date_days = np.zeros_like(a, dtype=np.int32)
    valid = (has_alert == 1) & (alert_date > 0)
    date_days[valid] = epoch_days + alert_date[valid].astype(np.int32) - 1

    return DecodedLabels(alert=has_alert, date_days=date_days, confidence=confidence, source=f"gladl{year_2digit:02d}")


# ---------- GLAD-S2 ----------


def decode_glads2(alert: np.ndarray, alert_date: np.ndarray) -> DecodedLabels:
    """alert: 0=none, 1=recent-only, 2=low, 3=medium, 4=high.
    alert_date: days since 2019-01-01.
    """
    a = alert.astype(np.int32)
    has_alert = (a > 0).astype(np.uint8)
    conf_lut = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    confidence = conf_lut[np.clip(a, 0, 4)]

    epoch_days = _days_since_epoch(GLADS2_EPOCH)
    date_days = np.zeros_like(a, dtype=np.int32)
    valid = (has_alert == 1) & (alert_date > 0)
    date_days[valid] = epoch_days + alert_date[valid].astype(np.int32)

    return DecodedLabels(alert=has_alert, date_days=date_days, confidence=confidence, source="glads2")


# ---------- Loading + reprojection onto tile grid ----------


def load_radd_on_grid(tile_id: str, grid: TileGrid) -> DecodedLabels | None:
    path = LABELS_DIR / "radd" / f"radd_{tile_id}_labels.tif"
    if not path.exists():
        return None
    raw = reproject_to_tile(path, grid, resampling=Resampling.nearest, band=1)
    return decode_radd(raw)


def _merge_decoded(items: list[DecodedLabels], source: str) -> DecodedLabels | None:
    """Collapse multiple DecodedLabels (e.g. per-year GLAD-L) into one.

    alert  = OR across years
    date   = earliest valid day
    conf   = max confidence across years that flagged the pixel
    """
    if not items:
        return None
    H, W = items[0].alert.shape
    alert = np.zeros((H, W), dtype=np.uint8)
    confidence = np.zeros((H, W), dtype=np.float32)
    date_days = np.full((H, W), np.iinfo(np.int32).max, dtype=np.int32)
    for it in items:
        hit = it.alert.astype(bool)
        alert |= it.alert
        confidence = np.maximum(confidence, np.where(hit, it.confidence, 0.0))
        date_days = np.where(hit, np.minimum(date_days, it.date_days), date_days)
    date_days = np.where(alert.astype(bool), date_days, 0).astype(np.int32)
    return DecodedLabels(alert=alert, date_days=date_days, confidence=confidence, source=source)


def load_gladl_on_grid(tile_id: str, grid: TileGrid) -> DecodedLabels | None:
    """Load all per-year GLAD-L files and merge into one DecodedLabels.

    Merging is essential: otherwise two GLAD-L alert years on the same pixel
    would each count as a separate source during weak-label fusion.
    """
    root = LABELS_DIR / "gladl"
    per_year: list[DecodedLabels] = []
    for alert_path in sorted(root.glob(f"gladl_{tile_id}_alert[0-9][0-9].tif")):
        yy = int(alert_path.stem[-2:])
        date_path = root / f"gladl_{tile_id}_alertDate{yy:02d}.tif"
        if not date_path.exists():
            continue
        alert = reproject_to_tile(alert_path, grid, resampling=Resampling.nearest, band=1)
        adate = reproject_to_tile(date_path, grid, resampling=Resampling.nearest, band=1)
        per_year.append(decode_gladl(alert.astype(np.int32), adate.astype(np.int32), yy))
    return _merge_decoded(per_year, source="gladl")


def load_glads2_on_grid(tile_id: str, grid: TileGrid) -> DecodedLabels | None:
    root = LABELS_DIR / "glads2"
    ap = root / f"glads2_{tile_id}_alert.tif"
    dp = root / f"glads2_{tile_id}_alertDate.tif"
    if not (ap.exists() and dp.exists()):
        return None
    alert = reproject_to_tile(ap, grid, resampling=Resampling.nearest, band=1)
    adate = reproject_to_tile(dp, grid, resampling=Resampling.nearest, band=1)
    return decode_glads2(alert.astype(np.int32), adate.astype(np.int32))


def load_all_labels(tile_id: str, split: str = "train") -> dict[str, DecodedLabels]:
    """Load every available weak-label source, collapsed to one entry per source.

    Keys are the source names (``radd``, ``gladl``, ``glads2``). Missing sources
    are omitted from the returned dict.
    """
    grid = tile_grid(tile_id, split)
    sources: dict[str, DecodedLabels] = {}
    if (r := load_radd_on_grid(tile_id, grid)) is not None:
        sources["radd"] = r
    if (g := load_gladl_on_grid(tile_id, grid)) is not None:
        sources["gladl"] = g
    if (s := load_glads2_on_grid(tile_id, grid)) is not None:
        sources["glads2"] = s
    return sources


# ---------- 2020 forest mask ----------


def forest_mask_2020(
    tile_id: str,
    split: str = "train",
    ndvi_threshold: float = 0.6,
) -> np.ndarray:
    """Bool mask: pixels that look like forest in 2020 (median NDVI > threshold).

    Uses Sentinel-2 B04 (red) and B08 (NIR) medians across 2020.
    """
    stack, _ = s2_year_stack(tile_id, 2020, split, bands=(4, 8))
    if stack.size == 0:
        g = tile_grid(tile_id, split)
        return np.zeros(g.shape, dtype=bool)
    masked = np.where(stack == 0, np.nan, stack.astype(np.float32))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        red_med = np.nanmedian(masked[:, 0], axis=0)
        nir_med = np.nanmedian(masked[:, 1], axis=0)
    denom = nir_med + red_med
    ndvi = np.where(denom > 0, (nir_med - red_med) / np.where(denom == 0, 1, denom), 0.0)
    return np.nan_to_num(ndvi, nan=0.0) > ndvi_threshold


# ---------- Fusion ----------


def fuse_post2020(
    tile_id: str,
    split: str = "train",
    gate_by_forest: bool = True,
    min_sources: int = 2,
) -> dict[str, np.ndarray]:
    """Fuse all weak-label sources into a post-2020 deforestation training target.

    Returns a dict with:
      - `any_alert`:       uint8 {0, 1} — at least one source flagged the pixel post-2020
      - `confident_pos`:   uint8 — ≥ ``min_sources`` sources agree post-2020 (training positives)
      - `confident_neg`:   uint8 — no source flagged AND pixel was forest in 2020 (training negatives)
      - `soft`:            float32 in [0, 1] — mean confidence across sources that flagged it
      - `event_days`:      int32 — earliest post-2020 alert date (days since 1970-01-01), 0 otherwise
      - `source_count`:    uint8 — number of sources agreeing post-2020
    """
    sources = load_all_labels(tile_id, split)
    grid = tile_grid(tile_id, split)
    H, W = grid.shape

    post_cutoff_days = _days_since_epoch(POST2020)
    stack_alert = []
    stack_conf = []
    stack_date = []

    for dl in sources.values():
        post = (dl.alert.astype(bool)) & (dl.date_days >= post_cutoff_days)
        stack_alert.append(post.astype(np.uint8))
        stack_conf.append(np.where(post, dl.confidence, 0.0).astype(np.float32))
        d = np.where(post, dl.date_days, np.iinfo(np.int32).max).astype(np.int32)
        stack_date.append(d)

    if not stack_alert:
        return {
            "any_alert": np.zeros((H, W), dtype=np.uint8),
            "confident_pos": np.zeros((H, W), dtype=np.uint8),
            "confident_neg": np.zeros((H, W), dtype=np.uint8),
            "soft": np.zeros((H, W), dtype=np.float32),
            "event_days": np.zeros((H, W), dtype=np.int32),
            "source_count": np.zeros((H, W), dtype=np.uint8),
        }

    alerts = np.stack(stack_alert, axis=0)
    confs = np.stack(stack_conf, axis=0)
    dates = np.stack(stack_date, axis=0)

    source_count = alerts.sum(axis=0).astype(np.uint8)
    any_alert = (source_count > 0).astype(np.uint8)
    confident_pos = (source_count >= min_sources).astype(np.uint8)

    # Mean confidence over sources that actually fired
    denom = np.clip(alerts.sum(axis=0), 1, None)
    soft = (confs.sum(axis=0) / denom).astype(np.float32)
    soft = np.where(any_alert == 1, soft, 0.0)

    # Earliest post-2020 alert day
    event_days = dates.min(axis=0).astype(np.int32)
    event_days = np.where(any_alert == 1, event_days, 0)

    if gate_by_forest:
        forest = forest_mask_2020(tile_id, split)
        confident_pos = (confident_pos & forest.astype(np.uint8)).astype(np.uint8)
        any_alert = (any_alert & forest.astype(np.uint8)).astype(np.uint8)
        confident_neg = ((source_count == 0) & forest).astype(np.uint8)
    else:
        confident_neg = (source_count == 0).astype(np.uint8)

    return {
        "any_alert": any_alert,
        "confident_pos": confident_pos,
        "confident_neg": confident_neg,
        "soft": soft,
        "event_days": event_days,
        "source_count": source_count,
    }
