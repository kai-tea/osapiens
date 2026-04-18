"""Shared manifest loading and fold helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .contract import required_manifest_paths


@dataclass(frozen=True)
class ManifestBundle:
    tile_manifest: pd.DataFrame
    file_manifest: pd.DataFrame
    label_manifest: pd.DataFrame
    pixel_index: pd.DataFrame
    split_assignments: pd.DataFrame


@dataclass(frozen=True)
class FoldSplit:
    train_tiles: pd.DataFrame
    val_tiles: pd.DataFrame
    test_tiles: pd.DataFrame


def require_manifest_files(root: str | Path | None = None) -> dict[str, Path]:
    paths = required_manifest_paths(root)
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        details = "\n".join(missing)
        raise FileNotFoundError(
            "Required model-input artifacts are missing. Generate/download artifacts first:\n"
            "  make download_data_from_s3\n"
            "  PYTHONPATH=. .venv/bin/python3 -m label_pipeline --data-root data/makeathon-challenge "
            "--split-dir cini/splits/split_v1 --output-root artifacts/labels_v1 --force\n"
            "  make prepare_model_inputs_v1\n\n"
            f"Missing files:\n{details}"
        )
    return paths


def load_manifests(root: str | Path | None = None) -> ManifestBundle:
    paths = require_manifest_files(root)
    return ManifestBundle(
        tile_manifest=pd.read_csv(paths["tile_manifest"]),
        file_manifest=pd.read_csv(paths["file_manifest"]),
        label_manifest=pd.read_parquet(paths["label_manifest"]),
        pixel_index=pd.read_parquet(paths["pixel_index"]),
        split_assignments=pd.read_csv(paths["split_assignments"]),
    )


def split_tiles(tile_manifest: pd.DataFrame, active_fold: int) -> FoldSplit:
    required = {"split", "fold_id", "tile_id"}
    missing = required.difference(tile_manifest.columns)
    if missing:
        raise ValueError(f"tile_manifest missing required columns: {sorted(missing)}")

    train_mask = (tile_manifest["split"] == "train") & (tile_manifest["fold_id"] != active_fold)
    val_mask = (tile_manifest["split"] == "train") & (tile_manifest["fold_id"] == active_fold)
    test_mask = tile_manifest["split"] == "test"

    fold_split = FoldSplit(
        train_tiles=tile_manifest.loc[train_mask].copy(),
        val_tiles=tile_manifest.loc[val_mask].copy(),
        test_tiles=tile_manifest.loc[test_mask].copy(),
    )
    if fold_split.train_tiles.empty:
        raise ValueError(f"No train tiles for active_fold={active_fold}")
    if fold_split.val_tiles.empty:
        raise ValueError(f"No validation tiles for active_fold={active_fold}")
    if fold_split.test_tiles.empty:
        raise ValueError("No test tiles in tile_manifest")
    return fold_split
