"""Create a deterministic region-disjoint split_v1 from train tile metadata."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from label_pipeline.constants import REGION_FIELD_CANDIDATES


def _fallback_region(tile_id: str) -> str:
    parts = tile_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-2])
    return tile_id


def _load_tile_regions(metadata_path: Path) -> pd.DataFrame:
    payload = json.loads(metadata_path.read_text())
    rows: list[dict[str, str]] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        tile_id = properties.get("tile_id", properties.get("name"))
        if tile_id is None:
            continue
        region_id = None
        for candidate in REGION_FIELD_CANDIDATES:
            value = properties.get(candidate)
            if value not in (None, ""):
                region_id = str(value)
                break
        rows.append(
            {
                "tile_id": str(tile_id),
                "region_id": region_id or _fallback_region(str(tile_id)),
            }
        )
    if not rows:
        raise ValueError(f"No train tile metadata rows found in {metadata_path}")
    return pd.DataFrame(rows).sort_values("tile_id").reset_index(drop=True)


def build_split_assignments(metadata_path: Path, n_folds: int = 3) -> pd.DataFrame:
    tiles = _load_tile_regions(metadata_path)
    region_groups = tiles.groupby("region_id")["tile_id"].apply(list)
    region_sizes = sorted(
        ((region_id, tile_ids) for region_id, tile_ids in region_groups.items()),
        key=lambda item: (-len(item[1]), item[0]),
    )

    fold_buckets: dict[int, list[str]] = {fold_id: [] for fold_id in range(n_folds)}
    fold_regions: dict[int, list[str]] = {fold_id: [] for fold_id in range(n_folds)}

    for region_id, tile_ids in region_sizes:
        target_fold = min(
            fold_buckets,
            key=lambda fold_id: (len(fold_buckets[fold_id]), len(fold_regions[fold_id]), fold_id),
        )
        fold_buckets[target_fold].extend(tile_ids)
        fold_regions[target_fold].append(region_id)

    tile_to_region = dict(zip(tiles["tile_id"], tiles["region_id"]))
    rows = []
    for fold_id, tile_ids in sorted(fold_buckets.items()):
        for tile_id in sorted(tile_ids):
            rows.append(
                {
                    "tile_id": tile_id,
                    "fold_id": fold_id,
                    "region_id": tile_to_region[tile_id],
                }
            )
    assignments = pd.DataFrame(rows).sort_values("tile_id").reset_index(drop=True)
    return assignments


def write_split_files(assignments: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "fold_assignments.csv", index=False)

    for fold_id in sorted(assignments["fold_id"].unique()):
        val_tiles = assignments.loc[assignments["fold_id"] == fold_id, ["tile_id", "region_id"]]
        train_tiles = assignments.loc[assignments["fold_id"] != fold_id, ["tile_id", "region_id"]]
        val_tiles.to_csv(output_dir / f"val_tiles_fold{fold_id}.csv", index=False)
        train_tiles.to_csv(output_dir / f"train_tiles_fold{fold_id}.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create deterministic split_v1 files.")
    parser.add_argument(
        "--metadata-path",
        default="data/makeathon-challenge/metadata/train_tiles.geojson",
        help="Path to train tile metadata GeoJSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="splits/split_v1",
        help="Directory where split_v1 CSV files will be written.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of folds to create.",
    )
    args = parser.parse_args(argv)

    metadata_path = Path(args.metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata path not found: {metadata_path}")

    assignments = build_split_assignments(metadata_path, n_folds=args.n_folds)
    write_split_files(assignments, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
