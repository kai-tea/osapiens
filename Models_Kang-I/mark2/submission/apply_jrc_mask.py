"""Apply a JRC forest mask to existing binary prediction rasters and export one merged submission GeoJSON."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


DEFAULT_INPUT_PREDICTION_DIR = Path("~/osapiens/artifacts/predictions_autoloop/autoloop_recall").expanduser()
DEFAULT_JRC_DIR = Path("~/jrc_gfc2020").expanduser()
DEFAULT_OUTPUT_MASKED_DIR = Path("~/osapiens/artifacts/predictions_autoloop/autoloop_recall_jrc").expanduser()
DEFAULT_OUTPUT_SUBMISSION_DIR = Path("~/osapiens/artifacts/submission/autoloop_recall_jrc").expanduser()
DEFAULT_MERGED_OUTPUT_PATH = (
    Path("~/osapiens/artifacts/submission/autoloop_recall_jrc/submission_autoloop_recall_jrc.geojson").expanduser()
)
DEFAULT_MIN_AREA_HA = 0.5
DEFAULT_TOP_VALUE_COUNT = 10


def run_command(args: list[str], capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a shell command and raise a clear error if it fails."""
    result = subprocess.run(args, check=False, text=True, capture_output=capture_output)
    if result.returncode != 0:
        joined = " ".join(shlex.quote(part) for part in args)
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(f"Command failed: {joined}\n{stderr}")
    return result


def list_prediction_rasters(input_dir: Path) -> list[Path]:
    """List binary prediction rasters in deterministic order."""
    return sorted(path for path in input_dir.glob("*.tif") if path.is_file())


def list_jrc_rasters(jrc_dir: Path) -> list[Path]:
    """List JRC rasters recursively in deterministic order."""
    return sorted(path for path in jrc_dir.rglob("*.tif*") if path.is_file())


def infer_tile_id(path: Path) -> str:
    """Infer a tile id from a binary raster filename."""
    suffix = "_binary"
    stem = path.stem
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def get_raster_info(path: Path) -> dict:
    """Read GDAL JSON metadata for a raster."""
    result = run_command(["gdalinfo", "-json", str(path)], capture_output=True)
    return json.loads(result.stdout)


def get_raster_bounds(info: dict) -> tuple[float, float, float, float]:
    """Compute raster bounds from GDAL metadata."""
    transform = info["geoTransform"]
    width = int(info["size"][0])
    height = int(info["size"][1])
    xmin = float(transform[0])
    ymax = float(transform[3])
    xmax = xmin + float(transform[1]) * width
    ymin = ymax + float(transform[5]) * height
    return xmin, ymin, xmax, ymax


def get_raster_wkt(info: dict) -> str:
    """Extract raster CRS WKT from GDAL metadata."""
    coordinate_system = info.get("coordinateSystem") or {}
    wkt = coordinate_system.get("wkt")
    if not wkt:
        raise ValueError("Could not read raster CRS WKT from gdalinfo output")
    return wkt


def build_jrc_vrt(jrc_paths: list[Path], vrt_path: Path) -> None:
    """Build a VRT mosaic covering all JRC rasters."""
    if not jrc_paths:
        raise FileNotFoundError("No JRC rasters found")
    run_command(["gdalbuildvrt", str(vrt_path), *[str(path) for path in jrc_paths]])


def warp_jrc_to_prediction_grid(jrc_vrt_path: Path, prediction_info: dict, output_path: Path) -> None:
    """Warp the JRC mosaic to exactly match one prediction raster grid."""
    xmin, ymin, xmax, ymax = get_raster_bounds(prediction_info)
    width = int(prediction_info["size"][0])
    height = int(prediction_info["size"][1])
    prediction_wkt = get_raster_wkt(prediction_info)

    run_command(
        [
            "gdalwarp",
            "-overwrite",
            "-r",
            "near",
            "-t_srs",
            prediction_wkt,
            "-te",
            str(xmin),
            str(ymin),
            str(xmax),
            str(ymax),
            "-ts",
            str(width),
            str(height),
            "-dstnodata",
            "0",
            str(jrc_vrt_path),
            str(output_path),
        ]
    )


def read_raster_values(path: Path) -> np.ndarray:
    """Read raster cell values through GDAL XYZ output into a flat NumPy array."""
    result = run_command(["gdal_translate", "-of", "XYZ", str(path), "/vsistdout/"], capture_output=True)
    values = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        values.append(float(line.rsplit(maxsplit=1)[-1]))
    return np.asarray(values, dtype=np.float32)


def count_positive_pixels(path: Path) -> int:
    """Count pixels equal to 1 in a binary raster."""
    values = read_raster_values(path)
    return int(np.count_nonzero(values == 1))


def parse_forest_values(raw_values: str | None) -> list[float]:
    """Parse an explicit comma-separated list of JRC values to keep as forest."""
    if not raw_values:
        return []
    return [float(value.strip()) for value in raw_values.split(",") if value.strip()]


def validate_mask_mode(
    *,
    inspect_only: bool,
    forest_values: list[float],
    forest_min_value: float | None,
) -> None:
    """Validate that the user picked exactly one masking mode when not inspecting."""
    using_values = bool(forest_values)
    using_threshold = forest_min_value is not None

    if using_values and using_threshold:
        raise ValueError("Use either --forest_values or --forest_min_value, not both")
    if not inspect_only and not using_values and not using_threshold:
        raise ValueError("Provide exactly one of --forest_values or --forest_min_value, or run with --inspect_only")


def summarize_values(values: np.ndarray, top_k: int = DEFAULT_TOP_VALUE_COUNT) -> dict[str, object]:
    """Summarize raster values for inspection mode."""
    unique_values, counts = np.unique(values, return_counts=True)
    order = np.argsort(-counts)
    total = int(values.size)

    top_entries = []
    for index in order[:top_k]:
        count = int(counts[index])
        percentage = 100.0 * count / total if total else 0.0
        top_entries.append(
            {
                "value": float(unique_values[index]),
                "count": count,
                "percent": percentage,
            }
        )

    return {
        "pixel_count": total,
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
        "unique_value_count": int(unique_values.size),
        "top_values": top_entries,
    }


def format_top_values(top_values: list[dict[str, object]]) -> str:
    """Format top value summaries in a compact, readable way."""
    if not top_values:
        return "none"
    parts = []
    for row in top_values:
        parts.append(f"{row['value']}: {row['count']} ({row['percent']:.2f}%)")
    return "; ".join(parts)


def print_inspection_summary(tile_id: str, summary: dict[str, object]) -> None:
    """Print a concise per-tile inspection summary."""
    print(
        f"{tile_id}: "
        f"min={summary['min']} "
        f"max={summary['max']} "
        f"unique={summary['unique_value_count']} "
        f"top={format_top_values(summary['top_values'])}"
    )


def combine_value_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    """Combine per-tile inspection summaries into one global view."""
    if not summaries:
        return {
            "pixel_count": 0,
            "min": 0.0,
            "max": 0.0,
            "unique_value_count": 0,
            "top_values": [],
        }

    all_values = []
    for row in summaries:
        all_values.append(np.asarray(row["values"], dtype=np.float32))
    concatenated = np.concatenate(all_values)
    return summarize_values(concatenated)


def build_gdal_calc_expression(forest_values: list[float], forest_min_value: float | None) -> str:
    """Build a GDAL calc expression for masking predictions outside forest."""
    if forest_values:
        keep_terms = [f"(B=={value})" for value in forest_values]
        keep_expression = "|".join(keep_terms)
        return f"((A==1)*({keep_expression}))"
    if forest_min_value is not None:
        return f"((A==1)*(B>={forest_min_value}))"
    raise ValueError("No mask rule provided for GDAL calc expression")


def apply_forest_mask(
    prediction_raster: Path,
    aligned_jrc_raster: Path,
    output_raster: Path,
    forest_values: list[float],
    forest_min_value: float | None,
) -> None:
    """Apply the chosen JRC forest mask to a binary prediction raster."""
    output_raster.parent.mkdir(parents=True, exist_ok=True)
    calc_expression = build_gdal_calc_expression(forest_values, forest_min_value)
    run_command(
        [
            "gdal_calc.py",
            "-A",
            str(prediction_raster),
            "-B",
            str(aligned_jrc_raster),
            "--calc",
            calc_expression,
            "--NoDataValue=255",
            "--type=Byte",
            "--overwrite",
            "--outfile",
            str(output_raster),
        ]
    )


def export_tile_submission(masked_raster: Path, output_geojson: Path, min_area_ha: float) -> dict[str, object]:
    """Convert one masked raster into challenge-format GeoJSON."""
    try:
        from submission_utils import raster_to_geojson
    except ModuleNotFoundError as exc:
        if exc.name == "geopandas":
            raise ModuleNotFoundError("geopandas required for submission export") from exc
        raise

    try:
        geojson = raster_to_geojson(masked_raster, output_path=output_geojson, min_area_ha=min_area_ha)
        export_mode = "polygonized"
    except ValueError:
        geojson = {"type": "FeatureCollection", "features": []}
        output_geojson.parent.mkdir(parents=True, exist_ok=True)
        output_geojson.write_text(json.dumps(geojson))
        export_mode = "empty_feature_collection"

    return {
        "geojson": geojson,
        "geojson_path": str(output_geojson),
        "polygon_count": len(geojson.get("features", [])),
        "min_area_ha": min_area_ha,
        "export_mode": export_mode,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for JRC masking and submission export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_prediction_dir", type=Path, default=DEFAULT_INPUT_PREDICTION_DIR)
    parser.add_argument("--jrc_dir", type=Path, default=DEFAULT_JRC_DIR)
    parser.add_argument("--output_masked_dir", type=Path, default=DEFAULT_OUTPUT_MASKED_DIR)
    parser.add_argument("--output_submission_dir", type=Path, default=DEFAULT_OUTPUT_SUBMISSION_DIR)
    parser.add_argument("--merged_output_path", type=Path, default=DEFAULT_MERGED_OUTPUT_PATH)
    parser.add_argument(
        "--forest_values",
        type=str,
        default=None,
        help="Comma-separated exact JRC values to keep as forest, for example '1' or '1,2'.",
    )
    parser.add_argument(
        "--forest_min_value",
        type=float,
        default=None,
        help="Keep pixels where aligned JRC values are greater than or equal to this threshold.",
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Inspect aligned JRC values per tile and stop before masking or submission export.",
    )
    parser.add_argument("--min_area_ha", type=float, default=DEFAULT_MIN_AREA_HA)
    return parser


def main() -> None:
    """Apply JRC masking to binary predictions and export submission GeoJSON."""
    args = build_argument_parser().parse_args()
    input_prediction_dir = args.input_prediction_dir.expanduser()
    jrc_dir = args.jrc_dir.expanduser()
    output_masked_dir = args.output_masked_dir.expanduser()
    output_submission_dir = args.output_submission_dir.expanduser()
    merged_output_path = args.merged_output_path.expanduser()
    forest_values = parse_forest_values(args.forest_values)
    forest_min_value = args.forest_min_value

    validate_mask_mode(
        inspect_only=args.inspect_only,
        forest_values=forest_values,
        forest_min_value=forest_min_value,
    )

    prediction_rasters = list_prediction_rasters(input_prediction_dir)
    if not prediction_rasters:
        raise FileNotFoundError(f"No prediction rasters found in {input_prediction_dir}")

    jrc_rasters = list_jrc_rasters(jrc_dir)
    if not jrc_rasters:
        raise FileNotFoundError(f"No JRC rasters found in {jrc_dir}")

    output_masked_dir.mkdir(parents=True, exist_ok=True)
    output_submission_dir.mkdir(parents=True, exist_ok=True)
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    inspection_rows: list[dict[str, object]] = []
    merged_features: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="mark2_jrc_mask_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        jrc_vrt_path = temp_dir / "jrc_mosaic.vrt"
        build_jrc_vrt(jrc_rasters, jrc_vrt_path)

        for prediction_raster in prediction_rasters:
            tile_id = infer_tile_id(prediction_raster)
            prediction_info = get_raster_info(prediction_raster)
            aligned_jrc_raster = temp_dir / f"{tile_id}_jrc_aligned.tif"
            warp_jrc_to_prediction_grid(jrc_vrt_path, prediction_info, aligned_jrc_raster)

            jrc_values = read_raster_values(aligned_jrc_raster)
            summary = summarize_values(jrc_values)
            print_inspection_summary(tile_id, summary)
            inspection_rows.append({"tile_id": tile_id, "values": jrc_values})

            if args.inspect_only:
                manifest_rows.append(
                    {
                        "tile_id": tile_id,
                        "prediction_raster": str(prediction_raster),
                        "jrc_summary": summary,
                    }
                )
                continue

            masked_raster = output_masked_dir / prediction_raster.name
            apply_forest_mask(
                prediction_raster=prediction_raster,
                aligned_jrc_raster=aligned_jrc_raster,
                output_raster=masked_raster,
                forest_values=forest_values,
                forest_min_value=forest_min_value,
            )

            input_positive_pixels = count_positive_pixels(prediction_raster)
            output_positive_pixels = count_positive_pixels(masked_raster)
            removed_positive_pixels = input_positive_pixels - output_positive_pixels
            print(
                f"{tile_id}: "
                f"input_positive_pixels={input_positive_pixels} "
                f"output_positive_pixels={output_positive_pixels} "
                f"removed_positive_pixels={removed_positive_pixels}"
            )

            intermediate_geojson_path = output_submission_dir / "per_tile_geojson" / f"pred_{tile_id}.geojson"
            submission_result = export_tile_submission(
                masked_raster=masked_raster,
                output_geojson=intermediate_geojson_path,
                min_area_ha=args.min_area_ha,
            )
            merged_features.extend(submission_result["geojson"].get("features", []))

            manifest_rows.append(
                {
                    "tile_id": tile_id,
                    "prediction_raster": str(prediction_raster),
                    "masked_raster": str(masked_raster),
                    "input_positive_pixels": input_positive_pixels,
                    "output_positive_pixels": output_positive_pixels,
                    "removed_positive_pixels": removed_positive_pixels,
                    "forest_values": forest_values if forest_values else None,
                    "forest_min_value": forest_min_value,
                    "jrc_summary": summary,
                    "geojson_path": submission_result["geojson_path"],
                    "polygon_count": submission_result["polygon_count"],
                    "min_area_ha": submission_result["min_area_ha"],
                    "export_mode": submission_result["export_mode"],
                }
            )

    global_summary = combine_value_summaries(inspection_rows)
    print(
        f"GLOBAL: "
        f"min={global_summary['min']} "
        f"max={global_summary['max']} "
        f"unique={global_summary['unique_value_count']} "
        f"top={format_top_values(global_summary['top_values'])}"
    )

    merged_geojson = {"type": "FeatureCollection", "features": merged_features}
    if not args.inspect_only:
        merged_output_path.write_text(json.dumps(merged_geojson))

    manifest_path = output_submission_dir / "jrc_mask_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "inspect_only": args.inspect_only,
                "forest_values": forest_values if forest_values else None,
                "forest_min_value": forest_min_value,
                "global_jrc_summary": global_summary,
                "merged_output_path": str(merged_output_path),
                "merged_feature_count": len(merged_features),
                "tile_count": len(prediction_rasters),
                "tiles": manifest_rows,
            },
            indent=2,
        )
    )

    if not args.inspect_only:
        print(f"merged_output_path={merged_output_path}")
        print(f"total_feature_count={len(merged_features)}")
        print(f"tile_count={len(prediction_rasters)}")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
