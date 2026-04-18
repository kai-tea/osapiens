"""Batch labelpack construction and artifact writing."""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    ABSENT_DAYS,
    AMBIGUOUS_HARD_LABEL,
    LABELPACK_BAND_NAMES,
    OUTPUT_NODATA,
    UNDEFINED_FLOAT,
)
from .decoders import decode_gladl, decode_glads2, decode_radd
from .splits import load_split_assignments


def _ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _parse_s2_filename(path: Path) -> tuple[int, int]:
    match = re.search(r"_(\d{4})_(\d+)\.tif$", path.name)
    if match is None:
        raise ValueError(f"Could not parse year/month from Sentinel-2 filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def _validate_label_inventory(data_root: Path) -> None:
    labels_root = data_root / "labels" / "train"
    required_sources = {
        "RADD": labels_root / "radd",
        "GLAD-L": labels_root / "gladl",
        "GLAD-S2": labels_root / "glads2",
    }
    for source_name, source_dir in required_sources.items():
        _ensure_exists(source_dir, f"{source_name} label directory")
        if not any(source_dir.glob("*.tif")):
            raise FileNotFoundError(f"{source_name} label directory is empty: {source_dir}")


def _discover_train_tiles(data_root: Path) -> dict[str, Path]:
    s2_train_root = data_root / "sentinel-2" / "train"
    _ensure_exists(s2_train_root, "Sentinel-2 train directory")

    tiles: dict[str, Path] = {}
    for tile_dir in sorted(s2_train_root.glob("*__s2_l2a")):
        if not tile_dir.is_dir():
            continue
        tile_id = tile_dir.name.replace("__s2_l2a", "")
        tiffs = sorted(tile_dir.glob("*.tif"), key=_parse_s2_filename)
        if not tiffs:
            raise FileNotFoundError(f"No Sentinel-2 TIFFs found for tile {tile_id}")
        tiles[tile_id] = tiffs[0]
    if not tiles:
        raise FileNotFoundError(f"No train tiles discovered in {s2_train_root}")
    return tiles


def _resolve_label_paths(data_root: Path, tile_id: str) -> dict:
    labels_root = data_root / "labels" / "train"
    radd_path = labels_root / "radd" / f"radd_{tile_id}_labels.tif"

    gladl_dir = labels_root / "gladl"
    gladl_paths: dict[int, dict[str, Path]] = {}
    for alert_path in sorted(gladl_dir.glob(f"gladl_{tile_id}_alert*.tif")):
        if "alertDate" in alert_path.name:
            continue
        year_suffix = alert_path.stem.split("alert")[-1]
        if not year_suffix.isdigit():
            continue
        year = int(year_suffix)
        if year < 20:
            continue
        date_path = gladl_dir / f"gladl_{tile_id}_alertDate{year_suffix}.tif"
        _ensure_exists(date_path, "GLAD-L alert date raster")
        gladl_paths[year] = {"alert": alert_path, "date": date_path}

    glads2_alert = labels_root / "glads2" / f"glads2_{tile_id}_alert.tif"
    glads2_date = labels_root / "glads2" / f"glads2_{tile_id}_alertDate.tif"
    glads2_paths = None
    if glads2_alert.exists() and glads2_date.exists():
        glads2_paths = {"alert": glads2_alert, "date": glads2_date}
    elif glads2_alert.exists() != glads2_date.exists():
        raise FileNotFoundError(
            f"Incomplete GLAD-S2 raster pair for tile {tile_id}: "
            f"alert={glads2_alert.exists()} date={glads2_date.exists()}"
        )

    return {
        "radd": radd_path if radd_path.exists() else None,
        "gladl": gladl_paths,
        "glads2": glads2_paths,
    }


def synthesize_labels(
    *,
    radd_obs: np.ndarray,
    radd_alert: np.ndarray,
    radd_conf: np.ndarray,
    gladl_obs: np.ndarray,
    gladl_alert: np.ndarray,
    gladl_conf: np.ndarray,
    glads2_obs: np.ndarray,
    glads2_alert: np.ndarray,
    glads2_conf: np.ndarray,
) -> dict[str, np.ndarray]:
    obs_count = (radd_obs + gladl_obs + glads2_obs).astype(np.uint8)
    train_mask = (obs_count >= 2).astype(np.uint8)

    soft_target = np.full(obs_count.shape, UNDEFINED_FLOAT, dtype=np.float32)
    conf_sum = radd_conf + gladl_conf + glads2_conf
    valid_train = train_mask == 1
    soft_target[valid_train] = conf_sum[valid_train] / obs_count[valid_train]

    strong_vote_radd = ((radd_alert == 1) & (radd_conf >= 0.5)).astype(np.uint8)
    strong_vote_gladl = ((gladl_alert == 1) & (gladl_conf >= 0.5)).astype(np.uint8)
    strong_vote_glads2 = ((glads2_alert == 1) & (glads2_conf >= 0.5)).astype(np.uint8)
    strong_vote_sum = (strong_vote_radd + strong_vote_gladl + strong_vote_glads2).astype(np.uint8)

    hard_label = np.full(obs_count.shape, AMBIGUOUS_HARD_LABEL, dtype=np.uint8)
    positive = (strong_vote_sum >= 2) & (obs_count >= 2)
    total_alerts = (radd_alert + gladl_alert + glads2_alert).astype(np.uint8)
    negative = (total_alerts == 0) & (obs_count == 3)

    hard_label[positive] = 1
    hard_label[negative] = 0

    sample_weight = np.full(obs_count.shape, UNDEFINED_FLOAT, dtype=np.float32)
    alert_count = total_alerts.astype(np.float32)
    sample_weight[valid_train] = np.maximum(
        alert_count[valid_train], obs_count[valid_train].astype(np.float32) - alert_count[valid_train]
    ) / 3.0

    seed_mask = np.zeros(obs_count.shape, dtype=np.uint8)
    positive_conf_sum = (
        (radd_conf * strong_vote_radd)
        + (gladl_conf * strong_vote_gladl)
        + (glads2_conf * strong_vote_glads2)
    ).astype(np.float32)
    positive_conf_mean = np.zeros(obs_count.shape, dtype=np.float32)
    positive_conf_mean[positive] = positive_conf_sum[positive] / strong_vote_sum[positive]

    seed_mask[positive & (positive_conf_mean >= 0.75)] = 1
    seed_mask[negative] = 1

    return {
        "train_mask": train_mask,
        "hard_label": hard_label,
        "seed_mask": seed_mask,
        "soft_target": soft_target,
        "sample_weight": sample_weight,
        "obs_count": obs_count,
    }


def build_labelpack(tile_id: str, s2_ref_path: str | Path, label_paths: dict) -> tuple[np.ndarray, dict]:
    from .reproject import load_reference_grid, reproject_raster_to_grid

    reference_grid = load_reference_grid(s2_ref_path)
    zeros_uint8 = np.zeros(reference_grid.shape, dtype=np.uint8)
    zeros_float = np.zeros(reference_grid.shape, dtype=np.float32)
    absent_days = np.full(reference_grid.shape, ABSENT_DAYS, dtype=np.int32)

    if label_paths["radd"] is None:
        radd_obs = zeros_uint8.copy()
        radd_alert = zeros_uint8.copy()
        radd_conf = zeros_float.copy()
        radd_days = absent_days.copy()
    else:
        radd_raw, radd_cov = reproject_raster_to_grid(label_paths["radd"], reference_grid)
        radd_obs, radd_alert, radd_conf, radd_days = decode_radd(radd_raw)
        radd_obs = np.minimum(radd_obs, radd_cov).astype(np.uint8)
        radd_alert[radd_obs == 0] = 0
        radd_conf[radd_obs == 0] = 0.0
        radd_days[radd_obs == 0] = ABSENT_DAYS

    if not label_paths["gladl"]:
        gladl_obs = zeros_uint8.copy()
        gladl_alert = zeros_uint8.copy()
        gladl_conf = zeros_float.copy()
        gladl_days = absent_days.copy()
    else:
        gladl_alert_arrays: dict[int, np.ndarray] = {}
        gladl_date_arrays: dict[int, np.ndarray] = {}
        gladl_cov_union = np.zeros(reference_grid.shape, dtype=np.uint8)
        for year, paths in sorted(label_paths["gladl"].items()):
            alert_array, alert_cov = reproject_raster_to_grid(paths["alert"], reference_grid)
            date_array, date_cov = reproject_raster_to_grid(paths["date"], reference_grid)
            gladl_alert_arrays[year] = alert_array
            gladl_date_arrays[year] = date_array
            gladl_cov_union = np.maximum(gladl_cov_union, np.maximum(alert_cov, date_cov))

        gladl_obs, gladl_alert, gladl_conf, gladl_days = decode_gladl(
            gladl_alert_arrays, gladl_date_arrays
        )
        gladl_obs = np.minimum(gladl_obs, gladl_cov_union).astype(np.uint8)
        gladl_alert[gladl_obs == 0] = 0
        gladl_conf[gladl_obs == 0] = 0.0
        gladl_days[gladl_obs == 0] = ABSENT_DAYS

    if label_paths["glads2"] is None:
        glads2_obs = zeros_uint8.copy()
        glads2_alert = zeros_uint8.copy()
        glads2_conf = zeros_float.copy()
        glads2_days = absent_days.copy()
    else:
        glads2_alert_array, glads2_alert_cov = reproject_raster_to_grid(
            label_paths["glads2"]["alert"], reference_grid
        )
        glads2_date_array, glads2_date_cov = reproject_raster_to_grid(
            label_paths["glads2"]["date"], reference_grid
        )
        glads2_obs, glads2_alert, glads2_conf, glads2_days = decode_glads2(
            glads2_alert_array, glads2_date_array
        )
        glads2_cov = np.maximum(glads2_alert_cov, glads2_date_cov)
        glads2_obs = np.minimum(glads2_obs, glads2_cov).astype(np.uint8)
        glads2_alert[glads2_obs == 0] = 0
        glads2_conf[glads2_obs == 0] = 0.0
        glads2_days[glads2_obs == 0] = ABSENT_DAYS

    synthesized = synthesize_labels(
        radd_obs=radd_obs,
        radd_alert=radd_alert,
        radd_conf=radd_conf,
        gladl_obs=gladl_obs,
        gladl_alert=gladl_alert,
        gladl_conf=gladl_conf,
        glads2_obs=glads2_obs,
        glads2_alert=glads2_alert,
        glads2_conf=glads2_conf,
    )

    label_raster = np.stack(
        [
            synthesized["train_mask"].astype(np.float32),
            synthesized["hard_label"].astype(np.float32),
            synthesized["seed_mask"].astype(np.float32),
            synthesized["soft_target"].astype(np.float32),
            synthesized["sample_weight"].astype(np.float32),
            synthesized["obs_count"].astype(np.float32),
            radd_obs.astype(np.float32),
            radd_alert.astype(np.float32),
            radd_conf.astype(np.float32),
            radd_days.astype(np.float32),
            gladl_obs.astype(np.float32),
            gladl_alert.astype(np.float32),
            gladl_conf.astype(np.float32),
            gladl_days.astype(np.float32),
            glads2_obs.astype(np.float32),
            glads2_alert.astype(np.float32),
            glads2_conf.astype(np.float32),
            glads2_days.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    train_mask = synthesized["train_mask"] == 1
    hard_label = synthesized["hard_label"]
    seed_mask = synthesized["seed_mask"] == 1

    tile_stats = {
        "tile_id": tile_id,
        "n_train_pixels": int(train_mask.sum()),
        "n_hard_pos": int((hard_label == 1).sum()),
        "n_hard_neg": int((hard_label == 0).sum()),
        "n_seed_pos": int((seed_mask & (hard_label == 1)).sum()),
        "n_seed_neg": int((seed_mask & (hard_label == 0)).sum()),
        "radd_pos_rate": float(radd_alert.sum() / radd_obs.sum()) if radd_obs.sum() else 0.0,
        "gladl_pos_rate": float(gladl_alert.sum() / gladl_obs.sum()) if gladl_obs.sum() else 0.0,
        "glads2_pos_rate": float(glads2_alert.sum() / glads2_obs.sum()) if glads2_obs.sum() else 0.0,
        "source_arrays": {
            "radd": {"obs": radd_obs, "alert": radd_alert},
            "gladl": {"obs": gladl_obs, "alert": gladl_alert},
            "glads2": {"obs": glads2_obs, "alert": glads2_alert},
        },
        "derived_arrays": synthesized,
    }

    return label_raster, tile_stats


def _write_labelpack(label_raster: np.ndarray, s2_ref_path: Path, output_path: Path) -> None:
    import rasterio

    from .reproject import load_reference_grid

    reference_grid = load_reference_grid(s2_ref_path)
    profile = reference_grid.profile.copy()
    profile.update(
        count=label_raster.shape[0],
        dtype="float32",
        nodata=OUTPUT_NODATA,
        compress="deflate",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(label_raster)
        dst.descriptions = tuple(LABELPACK_BAND_NAMES)


def _relative_posix(path: Path, root: Path) -> str:
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    return path.relative_to(root.resolve()).as_posix()


def _accumulate_overlap(
    accumulator: dict[tuple[str, str, str], dict[str, int]],
    *,
    scope_type: str,
    scope_value: str,
    source_arrays: dict,
) -> None:
    for left_name, right_name in combinations(["radd", "gladl", "glads2"], 2):
        left_obs = source_arrays[left_name]["obs"] == 1
        right_obs = source_arrays[right_name]["obs"] == 1
        left_alert = source_arrays[left_name]["alert"] == 1
        right_alert = source_arrays[right_name]["alert"] == 1
        overlap = left_obs & right_obs

        key = (scope_type, scope_value, f"{left_name}_{right_name}")
        bucket = accumulator.setdefault(
            key,
            {
                "n_obs_overlap": 0,
                "n_pos_left": 0,
                "n_pos_right": 0,
                "n_pos_both": 0,
                "n_disagree": 0,
            },
        )
        bucket["n_obs_overlap"] += int(overlap.sum())
        bucket["n_pos_left"] += int((left_alert & overlap).sum())
        bucket["n_pos_right"] += int((right_alert & overlap).sum())
        bucket["n_pos_both"] += int((left_alert & right_alert & overlap).sum())
        bucket["n_disagree"] += int(((left_alert != right_alert) & overlap).sum())


def _finalize_overlap_rows(
    accumulator: dict[tuple[str, str, str], dict[str, int]]
) -> list[dict[str, object]]:
    rows = []
    for (scope_type, scope_value, pair), counts in sorted(accumulator.items()):
        overlap = counts["n_obs_overlap"]
        pos_union = counts["n_pos_left"] + counts["n_pos_right"] - counts["n_pos_both"]
        rows.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "pair": pair,
                "n_obs_overlap": counts["n_obs_overlap"],
                "n_pos_left": counts["n_pos_left"],
                "n_pos_right": counts["n_pos_right"],
                "n_pos_both": counts["n_pos_both"],
                "disagreement_rate": counts["n_disagree"] / overlap if overlap else 0.0,
                "jaccard_pos": counts["n_pos_both"] / pos_union if pos_union else 0.0,
            }
        )
    return rows


def run_build(
    *,
    data_root: str | Path,
    split_dir: str | Path,
    output_root: str | Path,
    tile_id: str | None = None,
    force: bool = False,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(data_root)
    split_dir = Path(split_dir)
    output_root = Path(output_root)

    _ensure_exists(data_root, "Data root")
    _ensure_exists(split_dir, "Split directory")
    _validate_label_inventory(data_root)

    metadata_path = data_root / "metadata" / "train_tiles.geojson"
    assignments = load_split_assignments(split_dir, metadata_path)
    assignment_map = assignments.set_index("tile_id").to_dict("index")

    tile_refs = _discover_train_tiles(data_root)
    if tile_id is not None:
        if tile_id not in tile_refs:
            raise KeyError(f"Unknown tile_id requested: {tile_id}")
        tile_refs = {tile_id: tile_refs[tile_id]}

    missing_assignments = sorted(set(tile_refs) - set(assignment_map))
    if missing_assignments:
        raise ValueError(f"Tiles missing from split assignments: {missing_assignments}")

    labelpack_dir = output_root / "tiles"
    manifest_path = output_root / "manifest.parquet"
    pixel_index_path = output_root / "pixel_index.parquet"
    overlap_path = output_root / "source_overlap.csv"
    outputs = [manifest_path, pixel_index_path, overlap_path]

    if not force:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                f"Output files already exist: {[path.as_posix() for path in existing]}. Use --force to overwrite."
            )

    labelpack_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    pixel_index_frames: list[pd.DataFrame] = []
    overlap_accumulator: dict[tuple[str, str, str], dict[str, int]] = {}

    for current_tile_id, s2_ref_path in sorted(tile_refs.items()):
        labelpack_path = labelpack_dir / f"{current_tile_id}_labelpack.tif"
        if labelpack_path.exists() and not force:
            raise FileExistsError(
                f"Labelpack already exists for tile {current_tile_id}: {labelpack_path}. Use --force to overwrite."
            )

        label_paths = _resolve_label_paths(data_root, current_tile_id)
        label_raster, tile_stats = build_labelpack(current_tile_id, s2_ref_path, label_paths)
        _write_labelpack(label_raster, s2_ref_path, labelpack_path)

        fold_id = int(assignment_map[current_tile_id]["fold_id"])
        region_id = str(assignment_map[current_tile_id]["region_id"])

        manifest_rows.append(
            {
                "tile_id": current_tile_id,
                "labelpack_path": _relative_posix(labelpack_path, repo_root),
                "s2_ref_path": _relative_posix(Path(s2_ref_path), repo_root),
                "region_id": region_id,
                "fold_id": fold_id,
                "n_train_pixels": tile_stats["n_train_pixels"],
                "n_hard_pos": tile_stats["n_hard_pos"],
                "n_hard_neg": tile_stats["n_hard_neg"],
                "n_seed_pos": tile_stats["n_seed_pos"],
                "n_seed_neg": tile_stats["n_seed_neg"],
                "radd_pos_rate": tile_stats["radd_pos_rate"],
                "gladl_pos_rate": tile_stats["gladl_pos_rate"],
                "glads2_pos_rate": tile_stats["glads2_pos_rate"],
            }
        )

        derived = tile_stats["derived_arrays"]
        mask = derived["train_mask"] == 1
        rows, cols = np.nonzero(mask)
        pixel_index_frames.append(
            pd.DataFrame(
                {
                    "tile_id": current_tile_id,
                    "row": rows.astype(np.int32),
                    "col": cols.astype(np.int32),
                    "region_id": region_id,
                    "fold_id": fold_id,
                    "hard_label": derived["hard_label"][mask].astype(np.int32),
                    "seed_mask": derived["seed_mask"][mask].astype(np.int32),
                    "soft_target": derived["soft_target"][mask].astype(np.float32),
                    "sample_weight": derived["sample_weight"][mask].astype(np.float32),
                    "obs_count": derived["obs_count"][mask].astype(np.int32),
                    "radd_alert": tile_stats["source_arrays"]["radd"]["alert"][mask].astype(np.int32),
                    "gladl_alert": tile_stats["source_arrays"]["gladl"]["alert"][mask].astype(np.int32),
                    "glads2_alert": tile_stats["source_arrays"]["glads2"]["alert"][mask].astype(np.int32),
                }
            )
        )

        _accumulate_overlap(
            overlap_accumulator,
            scope_type="overall",
            scope_value="all",
            source_arrays=tile_stats["source_arrays"],
        )
        _accumulate_overlap(
            overlap_accumulator,
            scope_type="region",
            scope_value=region_id,
            source_arrays=tile_stats["source_arrays"],
        )

    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).sort_values("tile_id").to_parquet(manifest_path, index=False)

    if pixel_index_frames:
        pixel_index = pd.concat(pixel_index_frames, ignore_index=True)
    else:
        pixel_index = pd.DataFrame(
            columns=[
                "tile_id",
                "row",
                "col",
                "region_id",
                "fold_id",
                "hard_label",
                "seed_mask",
                "soft_target",
                "sample_weight",
                "obs_count",
                "radd_alert",
                "gladl_alert",
                "glads2_alert",
            ]
        )
    pixel_index.to_parquet(pixel_index_path, index=False)

    overlap_rows = _finalize_overlap_rows(overlap_accumulator)
    pd.DataFrame(overlap_rows).to_csv(overlap_path, index=False)
