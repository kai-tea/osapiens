"""Split loading and validation helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .constants import REGION_FIELD_CANDIDATES


def _tile_region_fallback(tile_id: str) -> str:
    parts = tile_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-2])
    return tile_id


def _metadata_region_mapping(metadata_path: Path) -> dict[str, str]:
    if not metadata_path.exists():
        return {}

    payload = json.loads(metadata_path.read_text())
    features = payload.get("features", [])
    if not features:
        return {}

    mapping: dict[str, str] = {}
    for feature in features:
        properties = feature.get("properties", {})
        tile_id = properties.get("tile_id", properties.get("name"))
        if tile_id is None:
            continue
        region_value = None
        for candidate in REGION_FIELD_CANDIDATES:
            if candidate in properties and properties[candidate] not in (None, ""):
                region_value = str(properties[candidate])
                break
        mapping[str(tile_id)] = region_value or _tile_region_fallback(str(tile_id))
    return mapping


def _load_fold_assignments_csv(split_dir: Path) -> pd.DataFrame | None:
    path = split_dir / "fold_assignments.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    required = {"tile_id", "fold_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df.copy()


def _load_val_tile_assignments(split_dir: Path) -> pd.DataFrame | None:
    val_paths = sorted(split_dir.glob("val_tiles_fold*.csv"))
    if not val_paths:
        return None

    rows = []
    pattern = re.compile(r"val_tiles_fold(\d+)\.csv$")
    for path in val_paths:
        match = pattern.search(path.name)
        if match is None:
            continue
        fold_id = int(match.group(1))
        df = pd.read_csv(path)
        if "tile_id" in df.columns:
            tile_ids = df["tile_id"]
        else:
            tile_ids = df.iloc[:, 0]
        rows.extend({"tile_id": str(tile_id), "fold_id": fold_id} for tile_id in tile_ids)

    if not rows:
        return None
    return pd.DataFrame(rows)


def load_split_assignments(split_dir: str | Path, metadata_path: str | Path) -> pd.DataFrame:
    split_dir = Path(split_dir)
    metadata_path = Path(metadata_path)

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    assignments = _load_fold_assignments_csv(split_dir)
    if assignments is None:
        assignments = _load_val_tile_assignments(split_dir)
    if assignments is None or assignments.empty:
        raise FileNotFoundError(
            f"Could not find split assignments in {split_dir}. "
            "Expected fold_assignments.csv or val_tiles_fold*.csv."
        )

    assignments = assignments.copy()
    if assignments["tile_id"].isna().any():
        raise ValueError("Split assignments contain missing tile_id values.")
    if assignments["fold_id"].isna().any():
        raise ValueError("Split assignments contain missing fold_id values.")
    assignments["tile_id"] = assignments["tile_id"].astype(str)
    assignments["fold_id"] = assignments["fold_id"].astype(int)

    if assignments["tile_id"].duplicated().any():
        duplicated = assignments.loc[assignments["tile_id"].duplicated(), "tile_id"].tolist()
        raise ValueError(f"Duplicate tile assignments found: {duplicated}")

    if "region_id" not in assignments.columns:
        assignments["region_id"] = None

    metadata_mapping = _metadata_region_mapping(metadata_path)
    assignments["region_id"] = assignments.apply(
        lambda row: (
            str(row["region_id"])
            if pd.notna(row["region_id"])
            else metadata_mapping.get(row["tile_id"], _tile_region_fallback(row["tile_id"]))
        ),
        axis=1,
    )

    if assignments["region_id"].isna().any():
        raise ValueError("Failed to derive region_id for all split assignments.")

    return assignments[["tile_id", "fold_id", "region_id"]].sort_values("tile_id").reset_index(
        drop=True
    )
