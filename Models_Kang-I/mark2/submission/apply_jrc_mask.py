"""Apply a JRC forest mask, build a few safe submission candidates, and export the best one."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
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
DEFAULT_CANDIDATE_COMPONENTS = "0,5,20"
DEFAULT_MAX_TILE_POSITIVE_FRACTION = 0.10
DEFAULT_TIME_STEP = 2506
DEFAULT_DEBUG_REPORT_NAME = "submission_debug_report.json"


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


def get_raster_pixel_count(info: dict) -> int:
    """Return total pixel count from GDAL metadata."""
    return int(info["size"][0]) * int(info["size"][1])


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


def parse_component_candidates(raw_values: str) -> list[int]:
    """Parse candidate connected-component sizes."""
    values = [int(value.strip()) for value in raw_values.split(",") if value.strip()]
    if not values:
        raise ValueError("candidate component list must not be empty")
    return sorted(dict.fromkeys(values))


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


def describe_masking_mode(forest_values: list[float], forest_min_value: float | None) -> dict[str, object]:
    """Return an explicit description of the active masking mode."""
    if forest_values:
        return {
            "mode": "forest_values",
            "forest_values": forest_values,
            "forest_min_value": None,
            "description": f"keep JRC pixels with exact values in {forest_values}",
        }
    if forest_min_value is not None:
        return {
            "mode": "forest_min_value",
            "forest_values": None,
            "forest_min_value": forest_min_value,
            "description": f"keep JRC pixels with value >= {forest_min_value}",
        }
    return {
        "mode": "inspect_only",
        "forest_values": None,
        "forest_min_value": None,
        "description": "inspection only; no masking applied",
    }


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


def build_candidate_raster(
    masked_raster: Path,
    output_raster: Path,
    min_component_pixels: int,
) -> None:
    """Create one candidate raster from the masked base raster."""
    output_raster.parent.mkdir(parents=True, exist_ok=True)
    if min_component_pixels <= 0:
        shutil.copyfile(masked_raster, output_raster)
        return
    run_command(
        [
            "gdal_sieve.py",
            "-st",
            str(min_component_pixels),
            "-8",
            str(masked_raster),
            str(output_raster),
        ]
    )


def export_tile_submission(masked_raster: Path, output_geojson: Path, min_area_ha: float, time_step: int) -> dict[str, object]:
    """Convert one masked raster into challenge-format GeoJSON with a deterministic YYMM time_step."""
    try:
        from submission_utils import raster_to_geojson
    except ModuleNotFoundError as exc:
        if exc.name == "geopandas":
            raise ModuleNotFoundError("geopandas required for submission export") from exc
        raise

    try:
        geojson = raster_to_geojson(masked_raster, output_path=None, min_area_ha=min_area_ha)
        export_mode = "polygonized"
    except ValueError:
        geojson = {"type": "FeatureCollection", "features": []}
        export_mode = "empty_feature_collection"

    for feature in geojson.get("features", []):
        feature["properties"] = {"time_step": int(time_step)}

    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    output_geojson.write_text(json.dumps(geojson))
    return {
        "geojson": geojson,
        "geojson_path": str(output_geojson),
        "polygon_count": len(geojson.get("features", [])),
        "min_area_ha": min_area_ha,
        "export_mode": export_mode,
        "time_step": int(time_step),
    }


def build_candidate_name(min_component_pixels: int) -> str:
    """Create a readable candidate tag."""
    return "masked_raw" if min_component_pixels <= 0 else f"masked_sieve_{min_component_pixels}"


def candidate_rank_key(summary: dict[str, object]) -> tuple[int, int, int, float]:
    """Transparent selection key: prefer safe candidates, then stronger cleanup."""
    return (
        -int(bool(summary["is_safe"])),
        -int(summary["min_component_pixels"]),
        -int(summary["total_feature_count"]),
        float(summary["max_tile_positive_fraction"]),
    )


def build_candidate_reason(summary: dict[str, object]) -> list[str]:
    """Explain why a candidate is safe or unsafe."""
    reasons = []
    if int(summary["total_feature_count"]) <= 0:
        reasons.append("no polygons after polygonization")
    if int(summary["dense_tile_count"]) > 0:
        reasons.append("one or more tiles exceed max_tile_positive_fraction")
    if int(summary["empty_tile_count"]) >= int(summary["tile_count"]):
        reasons.append("all tiles are empty")
    if not reasons:
        reasons.append("passes current safety checks")
    return reasons


def select_best_candidate(candidate_summaries: list[dict[str, object]]) -> dict[str, object]:
    """Choose the safest candidate by simple explicit rules."""
    if not candidate_summaries:
        raise ValueError("No candidate summaries available for selection")
    return max(candidate_summaries, key=candidate_rank_key)


def rank_candidates(candidate_summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return candidates sorted by the exact selection order."""
    return sorted(candidate_summaries, key=candidate_rank_key, reverse=True)


def sanity_check_merged_geojson(geojson: dict[str, object]) -> dict[str, object]:
    """Run a lightweight structural sanity check on the merged submission object."""
    features = geojson.get("features", [])
    geometry_type_counts: dict[str, int] = {}
    missing_properties_count = 0
    missing_or_null_time_step_count = 0
    unique_time_steps: set[int] = set()

    if isinstance(features, list):
        for feature in features:
            geometry = feature.get("geometry", {}) if isinstance(feature, dict) else {}
            geometry_type = geometry.get("type", "missing") if isinstance(geometry, dict) else "missing"
            geometry_type_counts[geometry_type] = geometry_type_counts.get(geometry_type, 0) + 1

            properties = feature.get("properties") if isinstance(feature, dict) else None
            if not isinstance(properties, dict):
                missing_properties_count += 1
                missing_or_null_time_step_count += 1
                continue

            time_step = properties.get("time_step")
            if time_step is None:
                missing_or_null_time_step_count += 1
            else:
                unique_time_steps.add(int(time_step))

    return {
        "top_level_type": geojson.get("type"),
        "is_feature_collection": geojson.get("type") == "FeatureCollection",
        "total_feature_count": len(features) if isinstance(features, list) else 0,
        "geometry_type_counts": geometry_type_counts,
        "missing_properties_count": missing_properties_count,
        "missing_or_null_time_step_count": missing_or_null_time_step_count,
        "unique_time_steps": sorted(unique_time_steps),
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
    parser.add_argument(
        "--candidate_min_component_pixels",
        type=str,
        default=DEFAULT_CANDIDATE_COMPONENTS,
        help="Comma-separated connected-component size candidates, for example '0,5,20'.",
    )
    parser.add_argument(
        "--max_tile_positive_fraction",
        type=float,
        default=DEFAULT_MAX_TILE_POSITIVE_FRACTION,
        help="Reject candidates with any tile above this positive-pixel fraction.",
    )
    parser.add_argument(
        "--time_step",
        type=int,
        default=DEFAULT_TIME_STEP,
        help="Deterministic YYMM time_step assigned to all exported polygons. Default 2506.",
    )
    parser.add_argument("--min_area_ha", type=float, default=DEFAULT_MIN_AREA_HA)
    return parser


def main() -> None:
    """Apply JRC masking to binary predictions, build candidate submissions, and export the best safe one."""
    args = build_argument_parser().parse_args()
    input_prediction_dir = args.input_prediction_dir.expanduser()
    jrc_dir = args.jrc_dir.expanduser()
    output_masked_dir = args.output_masked_dir.expanduser()
    output_submission_dir = args.output_submission_dir.expanduser()
    merged_output_path = args.merged_output_path.expanduser()
    debug_report_path = output_submission_dir / DEFAULT_DEBUG_REPORT_NAME
    forest_values = parse_forest_values(args.forest_values)
    forest_min_value = args.forest_min_value
    candidate_components = parse_component_candidates(args.candidate_min_component_pixels)
    masking_mode = describe_masking_mode(forest_values, forest_min_value)
    warnings: list[str] = []

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

    print(f"masking_mode={masking_mode['mode']}")
    print(f"masking_description={masking_mode['description']}")
    if masking_mode["forest_values"] is not None:
        print(f"forest_values={masking_mode['forest_values']}")
    if masking_mode["forest_min_value"] is not None:
        print(f"forest_min_value={masking_mode['forest_min_value']}")

    manifest_rows: list[dict[str, object]] = []
    inspection_rows: list[dict[str, object]] = []
    all_candidate_tile_rows: list[dict[str, object]] = []
    candidate_states = {
        build_candidate_name(min_component_pixels): {
            "candidate_name": build_candidate_name(min_component_pixels),
            "min_component_pixels": min_component_pixels,
            "merged_features": [],
            "tile_rows": [],
            "tile_count": len(prediction_rasters),
            "dense_tile_count": 0,
            "empty_tile_count": 0,
            "total_positive_pixels": 0,
            "total_feature_count": 0,
            "total_pixel_count": 0,
            "max_tile_positive_fraction": 0.0,
        }
        for min_component_pixels in candidate_components
    }

    with tempfile.TemporaryDirectory(prefix="mark2_jrc_mask_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        jrc_vrt_path = temp_dir / "jrc_mosaic.vrt"
        build_jrc_vrt(jrc_rasters, jrc_vrt_path)

        for prediction_raster in prediction_rasters:
            tile_id = infer_tile_id(prediction_raster)
            prediction_info = get_raster_info(prediction_raster)
            tile_pixel_count = get_raster_pixel_count(prediction_info)
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

            base_masked_raster = output_masked_dir / "base_masked" / prediction_raster.name
            apply_forest_mask(
                prediction_raster=prediction_raster,
                aligned_jrc_raster=aligned_jrc_raster,
                output_raster=base_masked_raster,
                forest_values=forest_values,
                forest_min_value=forest_min_value,
            )
            input_positive_pixels = count_positive_pixels(prediction_raster)

            for min_component_pixels in candidate_components:
                candidate_name = build_candidate_name(min_component_pixels)
                candidate_state = candidate_states[candidate_name]
                candidate_raster = output_masked_dir / candidate_name / prediction_raster.name
                build_candidate_raster(
                    masked_raster=base_masked_raster,
                    output_raster=candidate_raster,
                    min_component_pixels=min_component_pixels,
                )

                output_positive_pixels = count_positive_pixels(candidate_raster)
                removed_positive_pixels = input_positive_pixels - output_positive_pixels
                positive_fraction = output_positive_pixels / tile_pixel_count if tile_pixel_count else 0.0
                intermediate_geojson_path = (
                    output_submission_dir / "candidates" / candidate_name / "per_tile_geojson" / f"pred_{tile_id}.geojson"
                )
                submission_result = export_tile_submission(
                    masked_raster=candidate_raster,
                    output_geojson=intermediate_geojson_path,
                    min_area_ha=args.min_area_ha,
                    time_step=args.time_step,
                )

                candidate_state["merged_features"].extend(submission_result["geojson"].get("features", []))
                candidate_state["total_feature_count"] += int(submission_result["polygon_count"])
                candidate_state["total_positive_pixels"] += int(output_positive_pixels)
                candidate_state["total_pixel_count"] += int(tile_pixel_count)
                candidate_state["max_tile_positive_fraction"] = max(
                    float(candidate_state["max_tile_positive_fraction"]),
                    float(positive_fraction),
                )
                if positive_fraction > args.max_tile_positive_fraction:
                    candidate_state["dense_tile_count"] += 1
                if output_positive_pixels == 0 or submission_result["polygon_count"] == 0:
                    candidate_state["empty_tile_count"] += 1

                tile_row = {
                    "tile_id": tile_id,
                    "prediction_raster": str(prediction_raster),
                    "candidate_name": candidate_name,
                    "masked_raster": str(candidate_raster),
                    "input_positive_pixels": input_positive_pixels,
                    "output_positive_pixels": output_positive_pixels,
                    "removed_positive_pixels": removed_positive_pixels,
                    "positive_fraction": positive_fraction,
                    "forest_values": forest_values if forest_values else None,
                    "forest_min_value": forest_min_value,
                    "jrc_summary": summary,
                    "geojson_path": submission_result["geojson_path"],
                    "polygon_count": submission_result["polygon_count"],
                    "min_area_ha": submission_result["min_area_ha"],
                    "export_mode": submission_result["export_mode"],
                    "time_step": submission_result["time_step"],
                    "min_component_pixels": min_component_pixels,
                }
                candidate_state["tile_rows"].append(tile_row)
                all_candidate_tile_rows.append(tile_row)
                print(
                    f"{tile_id} [{candidate_name}]: "
                    f"input_positive_pixels={input_positive_pixels} "
                    f"output_positive_pixels={output_positive_pixels} "
                    f"removed_positive_pixels={removed_positive_pixels} "
                    f"positive_fraction={positive_fraction:.6f} "
                    f"polygon_count={submission_result['polygon_count']}"
                )

    global_summary = combine_value_summaries(inspection_rows)
    print(
        f"GLOBAL: "
        f"min={global_summary['min']} "
        f"max={global_summary['max']} "
        f"unique={global_summary['unique_value_count']} "
        f"top={format_top_values(global_summary['top_values'])}"
    )
    if (
        forest_min_value is not None
        and forest_min_value > 1
        and float(global_summary["min"]) >= 0.0
        and float(global_summary["max"]) <= 1.0
        and int(global_summary["unique_value_count"]) <= 2
    ):
        warning = (
            "WARNING: forest_min_value is greater than 1, but inspected JRC values look binary 0/1. "
            "This masking rule is suspicious for the current data."
        )
        warnings.append(warning)
        print(warning)

    candidate_summaries: list[dict[str, object]] = []
    for candidate_name, candidate_state in candidate_states.items():
        aggregate_positive_fraction = (
            candidate_state["total_positive_pixels"] / candidate_state["total_pixel_count"]
            if candidate_state["total_pixel_count"]
            else 0.0
        )
        is_safe = (
            candidate_state["total_feature_count"] > 0
            and candidate_state["dense_tile_count"] == 0
            and candidate_state["empty_tile_count"] < candidate_state["tile_count"]
        )
        rank_key = candidate_rank_key(
            {
                "is_safe": is_safe,
                "min_component_pixels": candidate_state["min_component_pixels"],
                "total_feature_count": candidate_state["total_feature_count"],
                "max_tile_positive_fraction": candidate_state["max_tile_positive_fraction"],
            }
        )
        candidate_summary = {
            "candidate_name": candidate_name,
            "min_component_pixels": candidate_state["min_component_pixels"],
            "tile_count": candidate_state["tile_count"],
            "dense_tile_count": candidate_state["dense_tile_count"],
            "empty_tile_count": candidate_state["empty_tile_count"],
            "total_positive_pixels": candidate_state["total_positive_pixels"],
            "aggregate_positive_fraction": aggregate_positive_fraction,
            "total_feature_count": candidate_state["total_feature_count"],
            "max_tile_positive_fraction": candidate_state["max_tile_positive_fraction"],
            "is_safe": is_safe,
            "reasons": build_candidate_reason(
                {
                    "total_feature_count": candidate_state["total_feature_count"],
                    "dense_tile_count": candidate_state["dense_tile_count"],
                    "empty_tile_count": candidate_state["empty_tile_count"],
                    "tile_count": candidate_state["tile_count"],
                }
            ),
            "rank_key": rank_key,
        }
        candidate_summaries.append(candidate_summary)
        print(
            f"CANDIDATE {candidate_name}: "
            f"is_safe={is_safe} "
            f"features={candidate_summary['total_feature_count']} "
            f"aggregate_positive_fraction={aggregate_positive_fraction:.6f} "
            f"empty_tiles={candidate_summary['empty_tile_count']} "
            f"dense_tiles={candidate_summary['dense_tile_count']} "
            f"rank_key={rank_key} "
            f"reasons={'; '.join(candidate_summary['reasons'])}"
        )

    ranked_candidates = rank_candidates(candidate_summaries)
    print("candidate_ranking:")
    for rank_index, candidate_summary in enumerate(ranked_candidates, start=1):
        print(
            f"  rank={rank_index} "
            f"name={candidate_summary['candidate_name']} "
            f"rank_key={candidate_summary['rank_key']} "
            f"is_safe={candidate_summary['is_safe']} "
            f"reasons={'; '.join(candidate_summary['reasons'])}"
        )
    print("selection_logic=prefer safe candidates, then stronger connected-component cleanup, then more features, then lower max tile density")

    selected_candidate = None
    selection_explanation = None
    final_geojson_sanity = None
    if not args.inspect_only:
        selected_candidate = select_best_candidate(candidate_summaries)
        selected_state = candidate_states[selected_candidate["candidate_name"]]
        merged_geojson = {"type": "FeatureCollection", "features": selected_state["merged_features"]}
        merged_output_path.write_text(json.dumps(merged_geojson))
        final_geojson_sanity = sanity_check_merged_geojson(merged_geojson)
        manifest_rows = selected_state["tile_rows"]
        selected_rank_index = next(
            index
            for index, candidate_summary in enumerate(ranked_candidates, start=1)
            if candidate_summary["candidate_name"] == selected_candidate["candidate_name"]
        )
        selection_explanation = {
            "selected_candidate_name": selected_candidate["candidate_name"],
            "selected_rank": selected_rank_index,
            "selected_rank_key": selected_candidate["rank_key"],
            "selected_reasons": selected_candidate["reasons"],
            "full_ranking_order": [candidate_summary["candidate_name"] for candidate_summary in ranked_candidates],
        }
        print(f"selected_candidate={selected_candidate['candidate_name']}")
        print(f"selected_rank={selected_rank_index}")
        print(f"selected_rank_key={selected_candidate['rank_key']}")
        print(f"selected_reasons={'; '.join(selected_candidate['reasons'])}")
        print(f"merged_output_path={merged_output_path}")
        print(f"total_feature_count={len(selected_state['merged_features'])}")
        print(f"tile_count={len(prediction_rasters)}")
        raw_candidate_summary = next(
            (candidate_summary for candidate_summary in candidate_summaries if candidate_summary["candidate_name"] == "masked_raw"),
            None,
        )
        if raw_candidate_summary is not None and selected_candidate["candidate_name"] != "masked_raw":
            selected_positive_pixels = int(selected_candidate["total_positive_pixels"])
            raw_positive_pixels = int(raw_candidate_summary["total_positive_pixels"])
            if raw_positive_pixels > 0 and selected_positive_pixels < 0.5 * raw_positive_pixels:
                warning = (
                    "CAUTION: selected candidate keeps less than 50% of the raw candidate's positive pixels. "
                    "This may indicate over-pruning."
                )
                warnings.append(warning)
                print(warning)
        print(
            f"final_geojson_sanity="
            f"type={final_geojson_sanity['top_level_type']} "
            f"is_feature_collection={final_geojson_sanity['is_feature_collection']} "
            f"features={final_geojson_sanity['total_feature_count']} "
            f"geometry_types={final_geojson_sanity['geometry_type_counts']} "
            f"missing_properties={final_geojson_sanity['missing_properties_count']} "
            f"missing_or_null_time_step={final_geojson_sanity['missing_or_null_time_step_count']} "
            f"unique_time_steps={final_geojson_sanity['unique_time_steps']}"
        )

    manifest_path = output_submission_dir / "jrc_mask_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "masking_mode": masking_mode,
                "inspect_only": args.inspect_only,
                "warnings": warnings,
                "forest_values": forest_values if forest_values else None,
                "forest_min_value": forest_min_value,
                "time_step": args.time_step,
                "global_jrc_summary": global_summary,
                "merged_output_path": str(merged_output_path),
                "selected_candidate": selected_candidate,
                "selection_explanation": selection_explanation,
                "final_geojson_sanity": final_geojson_sanity,
                "candidate_summaries": candidate_summaries,
                "tiles": manifest_rows,
            },
            indent=2,
        )
    )
    debug_report_path.write_text(
        json.dumps(
            {
                "masking_mode": masking_mode,
                "warnings": warnings,
                "global_jrc_summary": global_summary,
                "candidate_summaries": candidate_summaries,
                "candidate_ranking": ranked_candidates,
                "selected_candidate": selected_candidate,
                "selection_explanation": selection_explanation,
                "final_geojson_sanity": final_geojson_sanity,
                "all_candidate_tile_rows": all_candidate_tile_rows,
                "selected_candidate_tile_rows": manifest_rows,
                "merged_output_path": str(merged_output_path),
                "manifest_path": str(manifest_path),
            },
            indent=2,
        )
    )
    print(f"saved {manifest_path}")
    print(f"saved {debug_report_path}")


if __name__ == "__main__":
    main()
