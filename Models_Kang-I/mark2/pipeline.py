"""End-to-end data pipeline helpers for the Mark 2 baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from .labels.weak_labels import build_combined_labels, build_valid_training_mask, summarize_label_counts
    from .preprocessing.embeddings import (
        compute_channel_statistics,
        extract_pixel_features,
        normalize_embedding,
        save_normalization_statistics,
    )
    from .utils.io import infer_latest_common_year, load_embedding, list_tiles_for_year, split_training_tiles
except ImportError:
    from labels.weak_labels import build_combined_labels, build_valid_training_mask, summarize_label_counts
    from preprocessing.embeddings import (
        compute_channel_statistics,
        extract_pixel_features,
        normalize_embedding,
        save_normalization_statistics,
    )
    from utils.io import infer_latest_common_year, load_embedding, list_tiles_for_year, split_training_tiles


def prepare_tile_artifact(
    data_root: Path,
    tile_id: str,
    split: str,
    year: int,
    mean: np.ndarray,
    std: np.ndarray,
    include_labels: bool,
) -> dict[str, np.ndarray | str | int]:
    """Prepare normalized features and optional weak labels for one tile."""
    embedding, embedding_meta = load_embedding(data_root=data_root, tile_id=tile_id, split=split, year=year)
    normalized_embedding = normalize_embedding(embedding, mean=mean, std=std)
    features = extract_pixel_features(normalized_embedding)

    # Extension point: image + embedding multimodal fusion can be added here later.
    artifact: dict[str, np.ndarray | str | int] = {
        "tile_id": tile_id,
        "split": split,
        "year": year,
        "features": features,
    }

    if include_labels:
        labels, _ = build_combined_labels(data_root=data_root, tile_id=tile_id, reference_meta=embedding_meta)
        valid_mask = build_valid_training_mask(labels)
    else:
        labels = np.full(features.shape[:2], -1, dtype=np.int8)
        valid_mask = np.zeros(features.shape[:2], dtype=bool)

    artifact["labels"] = labels.astype(np.int8, copy=False)
    artifact["valid_mask"] = valid_mask.astype(bool, copy=False)
    return artifact


def save_tile_artifact(artifact: dict[str, np.ndarray | str | int], output_path: Path) -> None:
    """Persist one tile artifact as a compressed NumPy archive."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        tile_id=np.array([artifact["tile_id"]]),
        split=np.array([artifact["split"]]),
        year=np.array([artifact["year"]], dtype=np.int32),
        features=np.asarray(artifact["features"], dtype=np.float32),
        labels=np.asarray(artifact["labels"], dtype=np.int8),
        valid_mask=np.asarray(artifact["valid_mask"], dtype=bool),
    )


def run_embedding_label_pipeline(
    data_root: Path,
    output_dir: Path,
    year: int | None = None,
    val_fraction: float = 0.25,
    include_test: bool = False,
) -> dict[str, object]:
    """Run the full preprocessing and weak-label pipeline for the embedding baseline."""
    selected_year = year if year is not None else infer_latest_common_year(data_root)
    train_tile_ids = list_tiles_for_year(data_root, split="train", year=selected_year)
    if not train_tile_ids:
        raise RuntimeError(f"No training tiles found for year {selected_year}")

    fit_tiles, validation_tiles = split_training_tiles(train_tile_ids, val_fraction=val_fraction)
    stats = compute_channel_statistics(
        data_root=data_root,
        tile_ids=fit_tiles,
        split="train",
        year=selected_year,
    )
    stats_path = output_dir / "stats" / f"normalization_year_{selected_year}.json"
    save_normalization_statistics(
        stats_path,
        year=selected_year,
        training_tiles=fit_tiles,
        mean=stats["mean"],
        std=stats["std"],
        pixel_count=int(stats["pixel_count"][0]),
    )

    split_plan: list[tuple[str, list[str], str, bool]] = [
        ("train", fit_tiles, "train", True),
        ("validation", validation_tiles, "train", True),
    ]
    if include_test:
        test_tiles = list_tiles_for_year(data_root, split="test", year=selected_year)
        split_plan.append(("test", test_tiles, "test", False))

    summaries: dict[str, dict[str, int]] = {}
    for output_split, tile_ids, data_split, include_labels in split_plan:
        totals = {"positive": 0, "negative": 0, "uncertain": 0}
        for tile_id in tile_ids:
            artifact = prepare_tile_artifact(
                data_root=data_root,
                tile_id=tile_id,
                split=data_split,
                year=selected_year,
                mean=stats["mean"],
                std=stats["std"],
                include_labels=include_labels,
            )
            save_tile_artifact(
                artifact,
                output_dir / output_split / f"{tile_id}_{selected_year}.npz",
            )

            if include_labels:
                tile_counts = summarize_label_counts(np.asarray(artifact["labels"]))
                totals["positive"] += tile_counts["positive"]
                totals["negative"] += tile_counts["negative"]
                totals["uncertain"] += tile_counts["uncertain"]

        summaries[output_split] = totals

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "year": selected_year,
        "train_tiles": fit_tiles,
        "validation_tiles": validation_tiles,
        "stats_path": str(stats_path),
        "summaries": summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    return summary_payload
