"""No-train heuristic candidate GeoJSONs (safety net).

Each variant polygonises a Hansen-GFC-based rule. These run in minutes and
do not depend on training succeeding, so the autoloop always produces at
least a handful of shippable GeoJSONs even if the model crashes.

Current leaderboard best (2026-04-18 16:39): 31.20% — Hansen
tc>=25% lossyear 21-25 with no time_step. We tweak the lossyear window
and treecover threshold to explore the recall/FPR trade-off, plus add
per-polygon time_step from the mode lossyear which the original
submit_heuristic dropped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from src.data import load_tile_manifest, resolve_tile_paths
from src.masks import LOSSYEAR_2020
from src.predict import (
    REFUSE_ABOVE_DEFAULT,
    REFUSE_BELOW_DEFAULT,
    SubmissionRefusedError,
    _enforce_refusal_rule,
    _write_binary_raster,
)
from src.submit_heuristic import heuristic_binary_raster

from .change_detect import imagery_change_binary
from .io import REPO_ROOT
from .predict import (
    RASTER_ROOT,
    SUBMISSION_ROOT,
    _polygons_with_time_step,
    combine_submission_geojsons,
    hansen_lossyear_raster,
    morphological_clean,
)

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    tag: str
    kind: str  # "hansen" | "ndvi" | "ndvi_s1"
    treecover_threshold: int = 25
    earliest_loss_year: int = LOSSYEAR_2020 + 1
    latest_loss_year: int = 25
    morph_open: int = 1
    min_area_ha: float = 0.5


DEFAULT_VARIANTS = (
    # --- Hansen-lossyear family (depends on GFC only; Cini-free) ---
    # Kaite's shipped baseline, refreshed with time_step.
    Variant("hansen_tc25_ly21to25_ts", "hansen", 25, LOSSYEAR_2020 + 1, 25, morph_open=1),
    # Higher recall (loosen canopy cover).
    Variant("hansen_tc15_ly21to25_ts", "hansen", 15, LOSSYEAR_2020 + 1, 25, morph_open=1),
    # Higher precision (tighter canopy).
    Variant("hansen_tc35_ly21to25_ts", "hansen", 35, LOSSYEAR_2020 + 1, 25, morph_open=1),
    # Recent-only (closer to the hidden-label window the grader likely uses).
    Variant("hansen_tc25_ly22to25_ts", "hansen", 25, LOSSYEAR_2020 + 2, 25, morph_open=1),
    Variant("hansen_tc25_ly23to25_ts", "hansen", 25, LOSSYEAR_2020 + 3, 25, morph_open=1),
    # No morphological cleanup — one polygon extra, costs FPR.
    Variant("hansen_tc25_ly21to25_raw", "hansen", 25, LOSSYEAR_2020 + 1, 25, morph_open=0),
    # --- Imagery change-detection family (Cini-free, label-free) ---
    # Pure NDVI drop on Hansen-gated forest.
    Variant("ndvi_drop_hansen", "ndvi", 25, morph_open=1),
    # NDVI drop AND S1 VV drop — tighter, lower FPR.
    Variant("ndvi_s1_drop_hansen", "ndvi_s1", 25, morph_open=1),
)


def _compute_variant_binary(
    variant: Variant, tile_id: str, manifest: pd.DataFrame
) -> tuple[np.ndarray, dict]:
    """Return (H, W) boolean mask + ref_profile for the chosen variant kind."""
    paths = resolve_tile_paths(tile_id, manifest=manifest)
    with rasterio.open(paths.s2_ref_path) as src:
        ref_profile = src.profile.copy()

    if variant.kind == "hansen":
        binary, _ff, _lf = heuristic_binary_raster(
            ref_profile,
            treecover_threshold=variant.treecover_threshold,
            earliest_loss_year=variant.earliest_loss_year,
            latest_loss_year=variant.latest_loss_year,
        )
    elif variant.kind == "ndvi":
        binary, _rp, _stats = imagery_change_binary(tile_id, manifest, require_s1_agreement=False)
    elif variant.kind == "ndvi_s1":
        binary, _rp, _stats = imagery_change_binary(tile_id, manifest, require_s1_agreement=True)
    else:
        raise ValueError(f"unknown variant kind: {variant.kind}")
    return binary.astype(bool), ref_profile


def emit_variant(
    variant: Variant,
    tile_ids: list[str],
    manifest: pd.DataFrame,
    refuse_above: float = REFUSE_ABOVE_DEFAULT,
    refuse_below: float = REFUSE_BELOW_DEFAULT,
) -> dict:
    """Produce one variant's per-tile rasters + GeoJSONs + combined merge."""
    tile_summaries: list[dict] = []
    for tid in tile_ids:
        binary, ref_profile = _compute_variant_binary(variant, tid, manifest)
        if variant.morph_open > 0:
            binary = morphological_clean(binary.astype(np.uint8), variant.morph_open).astype(bool)

        positive_fraction = float(binary.mean())
        raster_path = RASTER_ROOT / variant.tag / f"{tid}_binary.tif"
        _write_binary_raster(binary.astype(np.uint8), ref_profile, raster_path)

        try:
            _enforce_refusal_rule(tid, positive_fraction, refuse_above, refuse_below)
        except SubmissionRefusedError as err:
            logger.warning("variant %s / tile %s refused: %s", variant.tag, tid, err)
            tile_summaries.append(
                {"tile_id": tid, "refused": True, "positive_fraction": positive_fraction}
            )
            continue

        lossyear_hw = hansen_lossyear_raster(ref_profile)
        geojson_path = SUBMISSION_ROOT / variant.tag / f"{tid}.geojson"
        try:
            gj = _polygons_with_time_step(
                raster_path, lossyear_hw, variant.min_area_ha, geojson_path
            )
            n_polygons = len(gj.get("features", []))
        except ValueError as err:
            logger.warning("variant %s / tile %s: no polygons (%s)", variant.tag, tid, err)
            n_polygons = 0

        tile_summaries.append(
            {
                "tile_id": tid,
                "positive_fraction": positive_fraction,
                "n_polygons": int(n_polygons),
                "geojson_path": str(geojson_path.relative_to(REPO_ROOT)),
            }
        )

    combined_path = combine_submission_geojsons(variant.tag)
    return {
        "tag": variant.tag,
        "kind": variant.kind,
        "treecover_threshold": variant.treecover_threshold,
        "earliest_loss_year": variant.earliest_loss_year,
        "latest_loss_year": variant.latest_loss_year,
        "morph_open": variant.morph_open,
        "combined_geojson": str(combined_path.relative_to(REPO_ROOT)),
        "tiles": tile_summaries,
    }


def emit_all_variants(tile_ids: list[str] | None = None) -> list[dict]:
    manifest = load_tile_manifest()
    if tile_ids is None:
        tile_ids = manifest.loc[manifest["split"] == "test", "tile_id"].tolist()
    summaries = []
    for variant in DEFAULT_VARIANTS:
        logger.info("building variant: %s", variant.tag)
        summaries.append(emit_variant(variant, tile_ids, manifest))
    return summaries
