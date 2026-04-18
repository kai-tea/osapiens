"""Test-tile inference → Hansen-gated binary raster → GeoJSON with time_step.

Applies the binding constraints from TRAINING_PROTOCOL.md:
- Hansen GFC 2020 forest gate (treecover >= 25%, lossyear != 20 or never lost)
- Refusal rule: positive_fraction ∉ [5e-5, 0.10] → raise/skip tile
- min_area_ha = 0.5 on polygonisation

Per-polygon time_step: mode of Hansen lossyear pixels inside the polygon,
converted to YYMM with mid-year month (06). Polygons with no Hansen
lossyear coverage leave time_step absent — safer than guessing given
Year Accuracy penalises wrong years by area.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape

from src.data import resolve_tile_paths
from src.masks import (
    DEFAULT_TREECOVER_THRESHOLD,
    LOSSYEAR_2020,
    _ensure_hansen_pair,
    _warp_layer_to_ref,
    forest_mask_2020,
    hansen_tile_key,
)
from src.predict import (
    REFUSE_ABOVE_DEFAULT,
    REFUSE_BELOW_DEFAULT,
    SubmissionRefusedError,
    _enforce_refusal_rule,
    _write_binary_raster,
)
from submission_utils import raster_to_geojson

from .io import REPO_ROOT
from .train import (
    PixelMLP,
    _predict_tile_full_raster,
)

logger = logging.getLogger(__name__)

SUBMISSION_ROOT = REPO_ROOT / "submission" / "autoloop"
RASTER_ROOT = REPO_ROOT / "artifacts" / "predictions_autoloop"


def hansen_lossyear_raster(ref_profile: dict) -> np.ndarray:
    """Return an (H, W) array of Hansen lossyear values (0 = never lost)."""
    bounds = rasterio.warp.transform_bounds(
        ref_profile["crs"],
        "EPSG:4326",
        *rasterio.transform.array_bounds(
            ref_profile["height"], ref_profile["width"], ref_profile["transform"]
        ),
    )
    tile_key = hansen_tile_key(bounds)
    _tc_path, ly_path = _ensure_hansen_pair(tile_key)
    return _warp_layer_to_ref(ly_path, ref_profile)


def morphological_clean(binary: np.ndarray, open_radius: int = 1) -> np.ndarray:
    """One round of binary open + close to remove salt-and-pepper noise."""
    if open_radius <= 0:
        return binary.astype(np.uint8)
    st = ndimage.generate_binary_structure(2, 1)
    opened = ndimage.binary_opening(binary.astype(bool), structure=st, iterations=open_radius)
    closed = ndimage.binary_closing(opened, structure=st, iterations=open_radius)
    return closed.astype(np.uint8)


def _polygons_with_time_step(
    binary_path: Path,
    lossyear_hw: np.ndarray,
    min_area_ha: float,
    geojson_out: Path,
) -> dict:
    """Polygonise + compute per-polygon mode lossyear → YYMM time_step.

    Reuses submission_utils.raster_to_geojson's geometry logic inline so we
    can annotate each polygon with time_step before writing. Never invents
    a year: polygons without any Hansen lossyear pixel keep time_step=null.
    """
    with rasterio.open(binary_path) as src:
        data = src.read(1).astype(np.uint8)
        transform = src.transform
        crs = src.crs

    if data.sum() == 0:
        raise ValueError(f"empty binary raster: {binary_path}")

    geoms: list = []
    time_steps: list[int | None] = []
    poly_shapes = [
        (shape(geom), value)
        for geom, value in shapes(data, mask=data, transform=transform)
        if value == 1
    ]
    if not poly_shapes:
        raise ValueError(f"no foreground polygons in {binary_path}")

    # Build a (row, col) mask per polygon and fetch the mode lossyear.
    from rasterio.features import rasterize

    for poly, _val in poly_shapes:
        mask = rasterize(
            [(poly, 1)], out_shape=data.shape, transform=transform, fill=0, dtype=np.uint8
        ).astype(bool)
        ly_vals = lossyear_hw[mask]
        ly_positive = ly_vals[(ly_vals > LOSSYEAR_2020)]
        if ly_positive.size > 0:
            vals, counts = np.unique(ly_positive, return_counts=True)
            mode_ly = int(vals[np.argmax(counts)])
            year = 2000 + mode_ly
            yymm = int(f"{year % 100:02d}06")  # mid-year fallback
            time_steps.append(yymm)
        else:
            time_steps.append(None)
        geoms.append(poly)

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs).to_crs("EPSG:4326")
    utm = gdf.estimate_utm_crs()
    gdf_utm = gdf.to_crs(utm)
    keep = (gdf_utm.area / 10_000) >= min_area_ha
    gdf = gdf.loc[keep].reset_index(drop=True)
    kept_time_steps = [t for t, k in zip(time_steps, keep.tolist()) if k]
    gdf["time_step"] = kept_time_steps

    if gdf.empty:
        raise ValueError(f"all polygons < {min_area_ha} ha in {binary_path}")

    geojson = json.loads(gdf.to_json())
    # drop null properties.time_step entries to match the example schema.
    for feat in geojson["features"]:
        if feat["properties"].get("time_step") is None:
            feat["properties"].pop("time_step", None)

    geojson_out.parent.mkdir(parents=True, exist_ok=True)
    geojson_out.write_text(json.dumps(geojson))
    return geojson


def predict_test_tile(
    tile_id: str,
    model: PixelMLP,
    manifest: pd.DataFrame,
    threshold: float,
    candidate_tag: str,
    treecover_threshold: int = DEFAULT_TREECOVER_THRESHOLD,
    min_area_ha: float = 0.5,
    morph_open: int = 1,
    refuse_above: float = REFUSE_ABOVE_DEFAULT,
    refuse_below: float = REFUSE_BELOW_DEFAULT,
    extra_union: np.ndarray | None = None,
) -> dict:
    """Produce one tile's GeoJSON under submission/autoloop/<candidate_tag>/."""
    raster_dir = RASTER_ROOT / candidate_tag
    subm_dir = SUBMISSION_ROOT / candidate_tag

    proba_hw, ref_profile, _labels = _predict_tile_full_raster(model, tile_id, manifest)
    forest = forest_mask_2020(ref_profile, treecover_threshold=treecover_threshold)
    binary = ((proba_hw >= threshold) & forest).astype(np.uint8)
    if extra_union is not None:
        binary = np.maximum(binary, (extra_union & forest).astype(np.uint8))
    binary = morphological_clean(binary, open_radius=morph_open)

    positive_fraction = float(binary.mean())
    raster_path = raster_dir / f"{tile_id}_binary.tif"
    _write_binary_raster(binary, ref_profile, raster_path)

    _enforce_refusal_rule(tile_id, positive_fraction, refuse_above, refuse_below)

    lossyear_hw = hansen_lossyear_raster(ref_profile)
    geojson_path = subm_dir / f"{tile_id}.geojson"
    try:
        gj = _polygons_with_time_step(raster_path, lossyear_hw, min_area_ha, geojson_path)
        n_polygons = len(gj.get("features", []))
    except ValueError as err:
        logger.warning("tile %s (%s): no polygons (%s)", tile_id, candidate_tag, err)
        n_polygons = 0

    return {
        "tile_id": tile_id,
        "candidate": candidate_tag,
        "threshold": float(threshold),
        "positive_fraction": positive_fraction,
        "raster_path": str(raster_path.relative_to(REPO_ROOT)),
        "geojson_path": str(geojson_path.relative_to(REPO_ROOT)),
        "n_polygons": int(n_polygons),
    }


def emit_union_variant(
    union_tag: str,
    source_tags: list[str],
    tile_ids: list[str],
    manifest: pd.DataFrame,
    morph_open: int = 1,
    min_area_ha: float = 0.5,
) -> dict:
    """Union the per-tile binary rasters from multiple candidates into a new variant.

    Reads the already-written ``artifacts/predictions_autoloop/<tag>/<tile>_binary.tif``
    for each source tag, logical-ORs them, re-applies morphological cleanup
    and Hansen gate (via the source rasters' existing gating), polygonises
    with lossyear-mode time_step, and emits a combined GeoJSON. Sources
    that don't have a raster for a given tile are skipped.
    """
    from src.data import resolve_tile_paths as _rt
    from src.masks import forest_mask_2020

    out_subm = SUBMISSION_ROOT / union_tag
    out_raster = RASTER_ROOT / union_tag
    tile_summaries: list[dict] = []

    for tid in tile_ids:
        union = None
        ref_profile = None
        used_sources: list[str] = []
        for tag in source_tags:
            p = RASTER_ROOT / tag / f"{tid}_binary.tif"
            if not p.exists():
                continue
            with rasterio.open(p) as src:
                arr = src.read(1).astype(bool)
                if ref_profile is None:
                    ref_profile = src.profile.copy()
            union = arr if union is None else (union | arr)
            used_sources.append(tag)
        if union is None:
            tile_summaries.append({"tile_id": tid, "skipped": "no source rasters"})
            continue

        # Re-apply Hansen gate defensively (each source already did so, but
        # union could include pixels dilated by morphological close in one).
        forest = forest_mask_2020(ref_profile)
        union = union & forest
        union = morphological_clean(union.astype(np.uint8), open_radius=morph_open).astype(bool)

        positive_fraction = float(union.mean())
        raster_path = out_raster / f"{tid}_binary.tif"
        _write_binary_raster(union.astype(np.uint8), ref_profile, raster_path)

        try:
            _enforce_refusal_rule(
                tid, positive_fraction, REFUSE_ABOVE_DEFAULT, REFUSE_BELOW_DEFAULT
            )
        except SubmissionRefusedError as err:
            logger.warning("union %s / tile %s refused: %s", union_tag, tid, err)
            tile_summaries.append({"tile_id": tid, "refused": True})
            continue

        lossyear_hw = hansen_lossyear_raster(ref_profile)
        geojson_path = out_subm / f"{tid}.geojson"
        try:
            gj = _polygons_with_time_step(raster_path, lossyear_hw, min_area_ha, geojson_path)
            n_polygons = len(gj.get("features", []))
        except ValueError as err:
            logger.warning("union %s / tile %s: no polygons (%s)", union_tag, tid, err)
            n_polygons = 0

        tile_summaries.append(
            {
                "tile_id": tid,
                "sources": used_sources,
                "positive_fraction": positive_fraction,
                "n_polygons": int(n_polygons),
            }
        )

    combined_path = combine_submission_geojsons(union_tag)
    return {
        "tag": union_tag,
        "sources": source_tags,
        "combined_geojson": str(combined_path.relative_to(REPO_ROOT)),
        "tiles": tile_summaries,
    }


def combine_submission_geojsons(candidate_tag: str) -> Path:
    """Merge the per-tile GeoJSONs under submission/autoloop/<tag>/ into one.

    Strips every property other than a valid YYMM ``time_step`` so the
    combined file matches leaderboard.md: each feature has either
    ``{"time_step": <int>}`` or ``{}``. Merging also drops any feature
    whose geometry is not Polygon or MultiPolygon.
    """
    def _is_valid_yymm(v) -> bool:
        if isinstance(v, bool) or not isinstance(v, int):
            return False
        return 1 <= v <= 9912 and 1 <= (v % 100) <= 12

    in_dir = SUBMISSION_ROOT / candidate_tag
    out_path = SUBMISSION_ROOT / f"{candidate_tag}.geojson"
    features: list[dict] = []
    for gj_path in sorted(in_dir.glob("*.geojson")):
        gj = json.loads(gj_path.read_text())
        for feat in gj.get("features", []):
            geom = feat.get("geometry") or {}
            if geom.get("type") not in ("Polygon", "MultiPolygon"):
                continue
            props = feat.get("properties") or {}
            ts = props.get("time_step")
            clean_props: dict = {"time_step": ts} if _is_valid_yymm(ts) else {}
            features.append({"type": "Feature", "geometry": geom, "properties": clean_props})
    payload = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(payload))
    return out_path
