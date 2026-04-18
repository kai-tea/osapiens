"""Hansen-GFC-based heuristic submission (no ML model).

Produces a ship-ready submission using only Hansen Global Forest Change
(v1.11) layers. For each test tile:

- Download ``treecover2000`` and ``lossyear`` Hansen layers for the
  overlapping 10°×10° tile (cached under ``artifacts/masks_v1/hansen/``).
- Predict positive where ``treecover2000 >= T`` and ``lossyear ∈ [21, N]``
  — i.e. the pixel was at least ``T``% canopy in 2000 and Hansen
  detected loss in 2021 or later (post-2020).
- Apply the same submission refusal rule as ``src.predict`` to guard
  against pathological tiles (``positive_fraction`` outside
  ``[refuse_below, refuse_above]`` raises).
- Write a binary GeoTIFF + submission GeoJSON via
  ``submission_utils.raster_to_geojson``.

This is the current ship candidate per Kaite's v1 review: the ML model
lost to naive weak-label baselines on ungated CV, so until v2 clears
the ≥3 F1-point bar against a region-aware heuristic we ship Hansen.

Why Hansen and not ``copy_radd`` / ``majority_2of3``: the challenge
ships weak labels only for the train split, so RADD/GLAD rasters for
the test tiles would require fetching from external public sources.
Hansen gives a consensus-independent, globally uniform deforestation
signal using the same data we already download for the forest gate.
Note: this is a single-source approximation of what Kaite specified
(majority_2of3 in full-coverage regions, copy_radd elsewhere); swapping
it in becomes trivial once external RADD/GLAD downloads land.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

from submission_utils import raster_to_geojson

from .data import load_tile_manifest, resolve_tile_paths
from .masks import (
    DEFAULT_TREECOVER_THRESHOLD,
    LOSSYEAR_2020,
    _ensure_hansen_pair,
    _warp_layer_to_ref,
)
from .predict import (
    REFUSE_ABOVE_DEFAULT,
    REFUSE_BELOW_DEFAULT,
    SubmissionRefusedError,
    _enforce_refusal_rule,
    _write_binary_raster,
)
from .masks import hansen_tile_key

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_ROOT = REPO_ROOT / "artifacts" / "predictions_heuristic_v1"
SUBMISSION_ROOT = REPO_ROOT / "submission" / "heuristic_v1"


def heuristic_binary_raster(
    s2_ref_profile: dict,
    treecover_threshold: int = DEFAULT_TREECOVER_THRESHOLD,
    earliest_loss_year: int = LOSSYEAR_2020 + 1,
    latest_loss_year: int = 25,
) -> tuple[np.ndarray, float, float]:
    """Return ``(binary_mask, forest_fraction_2020, lossyear_fraction)``.

    ``binary_mask`` is boolean ``(H, W)`` on the reference S2 UTM grid
    with ``True`` where the pixel was ≥ ``treecover_threshold`` canopy
    in 2000 and lost between ``earliest_loss_year`` and
    ``latest_loss_year`` inclusive (both expressed as years since 2000).
    """
    bounds_4326 = rasterio.warp.transform_bounds(
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

    was_forest = tc >= treecover_threshold
    lost_post_2020 = (ly >= earliest_loss_year) & (ly <= latest_loss_year)
    binary = was_forest & lost_post_2020
    return binary, float(was_forest.mean()), float(lost_post_2020.mean())


def submit_tile(
    tile_id: str,
    raster_output_dir: Path,
    submission_output_dir: Path,
    min_area_ha: float,
    treecover_threshold: int,
    refuse_above: float,
    refuse_below: float,
    manifest: pd.DataFrame | None = None,
) -> dict:
    paths = resolve_tile_paths(tile_id, manifest=manifest)
    with rasterio.open(paths.s2_ref_path) as src:
        ref_profile = src.profile.copy()

    binary, forest_frac, lossyear_frac = heuristic_binary_raster(
        ref_profile, treecover_threshold=treecover_threshold
    )
    positive_fraction = float(binary.mean())
    logger.info(
        "tile %s: forest_2020=%.3f, lossyear_post2020=%.3f, positive=%.4f",
        tile_id,
        forest_frac,
        lossyear_frac,
        positive_fraction,
    )

    raster_path = raster_output_dir / f"{tile_id}_binary.tif"
    _write_binary_raster(binary.astype(np.uint8), ref_profile, raster_path)

    _enforce_refusal_rule(tile_id, positive_fraction, refuse_above, refuse_below)

    geojson_path = submission_output_dir / f"{tile_id}.geojson"
    geojson: dict | None = None
    n_polygons = 0
    try:
        geojson = raster_to_geojson(raster_path, geojson_path, min_area_ha=min_area_ha)
        n_polygons = len(geojson.get("features", []))
    except ValueError as err:
        logger.warning("tile %s: no submission polygons (%s)", tile_id, err)

    return {
        "tile_id": tile_id,
        "region_id": paths.region_id,
        "n_pixels": int(binary.size),
        "forest_fraction_2020": forest_frac,
        "lossyear_post2020_fraction": lossyear_frac,
        "positive_fraction": positive_fraction,
        "raster_path": str(raster_path.relative_to(REPO_ROOT)),
        "geojson_path": str(geojson_path.relative_to(REPO_ROOT))
        if geojson is not None
        else None,
        "n_polygons": int(n_polygons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiles", nargs="*", help="Test tile IDs (default: all test tiles in the manifest)"
    )
    parser.add_argument(
        "--raster-output-dir", type=Path, default=PREDICTIONS_ROOT
    )
    parser.add_argument(
        "--submission-output-dir", type=Path, default=SUBMISSION_ROOT
    )
    parser.add_argument("--min-area-ha", type=float, default=0.5)
    parser.add_argument(
        "--treecover-threshold", type=int, default=DEFAULT_TREECOVER_THRESHOLD
    )
    parser.add_argument("--refuse-above", type=float, default=REFUSE_ABOVE_DEFAULT)
    parser.add_argument("--refuse-below", type=float, default=REFUSE_BELOW_DEFAULT)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    manifest = load_tile_manifest()
    tile_ids = (
        args.tiles
        if args.tiles
        else manifest.loc[manifest["split"] == "test", "tile_id"].tolist()
    )

    summaries: list[dict] = []
    for tile_id in tile_ids:
        try:
            summary = submit_tile(
                tile_id=tile_id,
                raster_output_dir=args.raster_output_dir,
                submission_output_dir=args.submission_output_dir,
                min_area_ha=args.min_area_ha,
                treecover_threshold=args.treecover_threshold,
                refuse_above=args.refuse_above,
                refuse_below=args.refuse_below,
                manifest=manifest,
            )
        except SubmissionRefusedError as err:
            logger.error("%s", err)
            summaries.append({"tile_id": tile_id, "refused": True, "reason": str(err)})
            if not args.keep_going:
                raise
            continue
        summaries.append(summary)

    summary_path = args.submission_output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "strategy": "hansen_lossyear_post2020",
                "treecover_threshold": args.treecover_threshold,
                "refuse_above": args.refuse_above,
                "refuse_below": args.refuse_below,
                "tiles": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    logger.info("wrote submission summary -> %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
