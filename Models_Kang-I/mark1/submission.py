from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import rasterio

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from submission_utils import raster_to_geojson
try:
    from .data import remove_tiny_blobs, write_raster
except ImportError:
    from data import remove_tiny_blobs, write_raster


def export_tile_submission(
    binary_raster_path: str | Path,
    output_path: str | Path,
    min_area_ha: float = 0.5,
    allow_empty: bool = True,
) -> dict:
    binary_raster_path = Path(binary_raster_path)
    output_path = Path(output_path)
    try:
        geojson = raster_to_geojson(binary_raster_path, output_path=output_path, min_area_ha=min_area_ha)
        export_mode = "polygonized"
    except ValueError:
        if not allow_empty:
            raise
        geojson = {"type": "FeatureCollection", "features": []}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(geojson))
        export_mode = "empty_feature_collection"

    return {
        "tile_id": _infer_tile_id(binary_raster_path),
        "binary_raster": str(binary_raster_path),
        "geojson_path": str(output_path),
        "polygon_count": len(geojson.get("features", [])),
        "min_area_ha": min_area_ha,
        "export_mode": export_mode,
    }


def export_submission_set(
    prediction_outputs: list[dict],
    submission_dir: str | Path,
    min_area_ha: float = 0.5,
    allow_empty: bool = True,
) -> list[dict]:
    submission_dir = Path(submission_dir)
    manifests = []

    for prediction_output in prediction_outputs:
        tile_id = prediction_output["tile_id"]
        binary_raster_path = Path(prediction_output["binary_raster"])
        geojson_path = submission_dir / f"pred_{tile_id}.geojson"
        manifests.append(
            export_tile_submission(
                binary_raster_path=binary_raster_path,
                output_path=geojson_path,
                min_area_ha=min_area_ha,
                allow_empty=allow_empty,
            )
        )

    return manifests


def threshold_probability_raster(
    probability_raster_path: str | Path,
    output_path: str | Path,
    threshold: float,
    apply_blob_filter: bool = False,
    min_blob_size: int = 4,
) -> dict:
    probability_raster_path = Path(probability_raster_path)
    output_path = Path(output_path)

    with rasterio.open(probability_raster_path) as src:
        probability_map = src.read(1)
        reference_meta = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }

    valid_mask = np.isfinite(probability_map)
    binary_map = np.full(probability_map.shape, 255, dtype=np.uint8)
    binary_map[valid_mask] = (probability_map[valid_mask] >= threshold).astype(np.uint8)
    if apply_blob_filter:
        foreground = binary_map == 1
        foreground = remove_tiny_blobs(foreground, min_blob_size=min_blob_size)
        binary_map[valid_mask] = foreground[valid_mask].astype(np.uint8)
    write_raster(output_path, binary_map, reference_meta, dtype="uint8", nodata=255)

    return {
        "threshold": threshold,
        "binary_raster": str(output_path),
        "positive_pixel_count": int((binary_map == 1).sum()),
        "valid_pixel_count": int(valid_mask.sum()),
        "blob_filter_applied": apply_blob_filter,
        "min_blob_size": min_blob_size if apply_blob_filter else None,
    }


def export_thresholded_prediction_set(
    prediction_outputs: list[dict],
    output_dir: str | Path,
    threshold: float,
    min_area_ha: float = 0.5,
    allow_empty: bool = True,
    apply_blob_filter: bool = False,
    min_blob_size: int = 4,
) -> list[dict]:
    output_dir = Path(output_dir)
    binary_dir = output_dir / "binary_rasters"
    submission_dir = output_dir / "geojson"
    manifests = []

    for prediction_output in prediction_outputs:
        tile_id = prediction_output["tile_id"]
        threshold_result = threshold_probability_raster(
            probability_raster_path=prediction_output["probability_raster"],
            output_path=binary_dir / f"{tile_id}_binary.tif",
            threshold=threshold,
            apply_blob_filter=apply_blob_filter,
            min_blob_size=min_blob_size,
        )
        submission_result = export_tile_submission(
            binary_raster_path=threshold_result["binary_raster"],
            output_path=submission_dir / f"pred_{tile_id}.geojson",
            min_area_ha=min_area_ha,
            allow_empty=allow_empty,
        )
        manifests.append(
            {
                "tile_id": tile_id,
                "threshold": threshold,
                "probability_raster": prediction_output["probability_raster"],
                **threshold_result,
                **submission_result,
            }
        )

    return manifests


def summarize_submission_outputs(submission_outputs: list[dict]) -> dict:
    return {
        "tile_count": len(submission_outputs),
        "polygon_tile_count": sum(1 for row in submission_outputs if row.get("polygon_count", 0) > 0),
        "empty_tile_count": sum(1 for row in submission_outputs if row.get("polygon_count", 0) == 0),
        "total_polygon_count": sum(int(row.get("polygon_count", 0)) for row in submission_outputs),
        "total_positive_pixels": sum(int(row.get("positive_pixel_count", 0)) for row in submission_outputs),
    }


def _infer_tile_id(binary_raster_path: Path) -> str:
    suffix = "_binary"
    stem = binary_raster_path.stem
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem
