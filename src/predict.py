"""Test-set inference for the Kaite v1 baseline.

Loads the final LightGBM model saved by ``src.train``, extracts features
for each test tile (full raster, no subsampling), binarises the
prediction at the threshold written by ``src.eval``
(``reports/baseline_results_thresholds.json``, default key ``mean``),
and writes one GeoJSON per tile under ``submission/baseline_v1/`` via
:func:`submission_utils.raster_to_geojson`.

Also writes a per-tile binary GeoTIFF under
``artifacts/predictions_v1/`` so downstream visualisations can overlay
the rasterised prediction.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import rasterio

from submission_utils import raster_to_geojson

from .data import build_tile_features, feature_names, load_tile_manifest, resolve_tile_paths
from .train import MODELS_ROOT, DEFAULT_MODEL_NAME

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_ROOT = REPO_ROOT / "artifacts" / "predictions_v1"
SUBMISSION_ROOT = REPO_ROOT / "submission" / "baseline_v1"
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "reports" / "baseline_results_thresholds.json"


def load_model(model_dir: Path, name: str) -> tuple[lgb.Booster, dict]:
    model_path = model_dir / f"{name}.lgb"
    metadata_path = model_dir / f"{name}.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first with `python -m src.train`."
        )
    booster = lgb.Booster(model_file=str(model_path))
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return booster, metadata


def load_threshold(path: Path, key: str) -> float:
    if not path.exists():
        raise FileNotFoundError(
            f"Threshold sidecar not found at {path}. Run `python -m src.eval` first."
        )
    payload = json.loads(path.read_text())
    if key not in payload:
        raise KeyError(f"Threshold key '{key}' not in {path}; found keys={sorted(payload)}")
    value = float(payload[key])
    logger.info("using threshold=%.4f from %s[%s]", value, path, key)
    return value


def _write_binary_raster(
    binary: np.ndarray,
    ref_profile: dict,
    output_path: Path,
) -> None:
    profile = ref_profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(binary.astype(np.uint8), 1)


def predict_tile(
    tile_id: str,
    booster: lgb.Booster,
    threshold: float,
    raster_output_dir: Path,
    submission_output_dir: Path,
    min_area_ha: float,
    manifest: pd.DataFrame | None = None,
) -> dict:
    """Produce a binary prediction raster and submission GeoJSON for one tile."""
    paths = resolve_tile_paths(tile_id, manifest=manifest)
    logger.info("tile %s: extracting features", tile_id)
    features, names, _labels, ref_profile = build_tile_features(paths)
    if names != feature_names():
        raise RuntimeError(
            f"tile {tile_id}: feature name drift detected between data.py and inference"
        )

    F, H, W = features.shape
    X = features.reshape(F, -1).T.astype(np.float32)
    logger.info("tile %s: predicting %d pixels", tile_id, X.shape[0])
    proba = booster.predict(X)
    binary = (proba >= threshold).astype(np.uint8).reshape(H, W)
    positive_fraction = float(binary.mean())

    raster_path = raster_output_dir / f"{tile_id}_binary.tif"
    _write_binary_raster(binary, ref_profile, raster_path)

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
        "n_pixels": int(H * W),
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
        "--tiles",
        nargs="*",
        help="Test tile IDs (default: all tiles with split==test in the manifest)",
    )
    parser.add_argument("--model-dir", type=Path, default=MODELS_ROOT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--thresholds-path",
        type=Path,
        default=DEFAULT_THRESHOLDS_PATH,
        help="JSON sidecar written by src.eval",
    )
    parser.add_argument(
        "--threshold-key",
        default="mean",
        choices=["mean", "median", "min", "max"],
        help="Which aggregate from the eval sidecar to use as the submission threshold",
    )
    parser.add_argument(
        "--threshold-override",
        type=float,
        default=None,
        help="Use this threshold directly instead of reading the eval sidecar",
    )
    parser.add_argument(
        "--raster-output-dir", type=Path, default=PREDICTIONS_ROOT
    )
    parser.add_argument(
        "--submission-output-dir", type=Path, default=SUBMISSION_ROOT
    )
    parser.add_argument("--min-area-ha", type=float, default=0.5)
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
        tile_ids = manifest.loc[manifest["split"] == "test", "tile_id"].tolist()

    booster, model_metadata = load_model(args.model_dir, args.model_name)
    if model_metadata.get("feature_list") and model_metadata["feature_list"] != feature_names():
        raise RuntimeError(
            "Model was trained with a different feature list than the current data.py schema"
        )
    threshold = (
        args.threshold_override
        if args.threshold_override is not None
        else load_threshold(args.thresholds_path, args.threshold_key)
    )

    summaries: list[dict] = []
    for tile_id in tile_ids:
        summary = predict_tile(
            tile_id=tile_id,
            booster=booster,
            threshold=threshold,
            raster_output_dir=args.raster_output_dir,
            submission_output_dir=args.submission_output_dir,
            min_area_ha=args.min_area_ha,
            manifest=manifest,
        )
        logger.info(
            "tile %s: positive_fraction=%.4f polygons=%d",
            tile_id,
            summary["positive_fraction"],
            summary["n_polygons"],
        )
        summaries.append(summary)

    summary_path = args.submission_output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "model": str(args.model_dir / f"{args.model_name}.lgb"),
                "threshold": threshold,
                "threshold_key": (
                    "override" if args.threshold_override is not None else args.threshold_key
                ),
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
