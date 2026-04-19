"""Generate challenge-format submission artifacts from Mark 2 test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from submission_utils import raster_to_geojson

try:
    from ..utils.io import parse_tile_and_year
    from ..utils.npz_data import list_npz_files, load_tile_npz
    from ..utils.prediction import load_selected_threshold
    from ..utils.raster import load_reference_meta, write_raster
except ImportError:
    from utils.io import parse_tile_and_year
    from utils.npz_data import list_npz_files, load_tile_npz
    from utils.prediction import load_selected_threshold
    from utils.raster import load_reference_meta, write_raster


DEFAULT_DATA_ROOT = Path("data/makeathon-challenge")
DEFAULT_INPUT_DIR = Path("Models_Kang-I/mark2/outputs/predictions/mlp_test")
DEFAULT_OUTPUT_DIR = Path("Models_Kang-I/mark2/outputs/submission/mlp_test")
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_AREA_HA = 0.5


def export_tile_submission(
    probability_map: np.ndarray,
    reference_meta: dict,
    output_dir: Path,
    tile_id: str,
    threshold: float,
    min_area_ha: float,
) -> dict[str, object]:
    """Threshold one tile, write a binary raster, and polygonize it to GeoJSON."""
    binary_dir = output_dir / "binary_rasters"
    geojson_dir = output_dir / "geojson"
    binary_path = binary_dir / f"{tile_id}_binary.tif"
    geojson_path = geojson_dir / f"pred_{tile_id}.geojson"

    binary_map = (probability_map >= threshold).astype(np.uint8)
    write_raster(binary_path, binary_map, reference_meta, dtype="uint8", nodata=255)

    try:
        geojson = raster_to_geojson(binary_path, output_path=geojson_path, min_area_ha=min_area_ha)
        export_mode = "polygonized"
    except ValueError:
        geojson = {"type": "FeatureCollection", "features": []}
        geojson_path.parent.mkdir(parents=True, exist_ok=True)
        geojson_path.write_text(json.dumps(geojson))
        export_mode = "empty_feature_collection"

    return {
        "tile_id": tile_id,
        "threshold": threshold,
        "binary_raster": str(binary_path),
        "geojson_path": str(geojson_path),
        "positive_pixel_count": int(np.count_nonzero(binary_map == 1)),
        "polygon_count": len(geojson.get("features", [])),
        "min_area_ha": min_area_ha,
        "export_mode": export_mode,
    }


def resolve_threshold(threshold: float | None, threshold_report: Path | None) -> float:
    """Resolve the threshold either from CLI or from a saved validation report."""
    if threshold_report is not None:
        return load_selected_threshold(threshold_report)
    if threshold is None:
        return DEFAULT_THRESHOLD
    return float(threshold)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for submission generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT, help="Challenge data root directory.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with test prediction `.npz` files.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for binary rasters and GeoJSON.")
    parser.add_argument("--threshold", type=float, default=None, help="Explicit probability threshold for binarization.")
    parser.add_argument("--threshold_report", type=Path, default=None, help="Optional validation report JSON with selected threshold.")
    parser.add_argument("--min_area_ha", type=float, default=DEFAULT_MIN_AREA_HA, help="Minimum polygon area in hectares.")
    return parser


def main() -> None:
    """Create GeoJSON submission files from saved test prediction probabilities."""
    args = build_argument_parser().parse_args()
    chosen_threshold = resolve_threshold(args.threshold, args.threshold_report)
    manifests: list[dict[str, object]] = []

    for prediction_path in list_npz_files(args.input_dir):
        tile = load_tile_npz(prediction_path)
        if "probabilities" not in tile:
            raise ValueError(f"Prediction file {prediction_path} does not contain a probability map")

        tile_id = str(tile["tile_id"][0]) if "tile_id" in tile else parse_tile_and_year(prediction_path)[0]
        year = int(tile["year"][0]) if "year" in tile else parse_tile_and_year(prediction_path)[1]
        reference_path = args.data_root / "aef-embeddings" / "test" / f"{tile_id}_{year}.tiff"
        reference_meta = load_reference_meta(reference_path)

        manifests.append(
            export_tile_submission(
                probability_map=tile["probabilities"].astype(np.float32, copy=False),
                reference_meta=reference_meta,
                output_dir=args.output_dir,
                tile_id=tile_id,
                threshold=chosen_threshold,
                min_area_ha=args.min_area_ha,
            )
        )

    manifest_path = args.output_dir / "submission_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "threshold": chosen_threshold,
                "min_area_ha": args.min_area_ha,
                "tile_count": len(manifests),
                "tiles": manifests,
            },
            indent=2,
        )
    )
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
