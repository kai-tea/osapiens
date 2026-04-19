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
) -> dict[str, np.ndarray | str | int | dict[str, object]]:
    """Prepare normalized features and optional weak labels for one tile."""
    embedding, embedding_meta = load_embedding(data_root=data_root, tile_id=tile_id, split=split, year=year)
    normalized_embedding = normalize_embedding(embedding, mean=mean, std=std)
    features = extract_pixel_features(normalized_embedding)

    # Extension point: image + embedding multimodal fusion can be added here later.
    artifact: dict[str, np.ndarray | str | int | dict[str, object]] = {
        "tile_id": tile_id,
        "split": split,
        "year": year,
        "features": features,
    }

    if include_labels:
        labels, label_diagnostics = build_combined_labels(
            data_root=data_root,
            tile_id=tile_id,
            reference_meta=embedding_meta,
        )
        valid_mask = build_valid_training_mask(labels)
        artifact["label_diagnostics"] = label_diagnostics
    else:
        labels = np.full(features.shape[:2], -1, dtype=np.int8)
        valid_mask = np.zeros(features.shape[:2], dtype=bool)

    artifact["labels"] = labels.astype(np.int8, copy=False)
    artifact["valid_mask"] = valid_mask.astype(bool, copy=False)
    return artifact


def initialize_label_gating_totals() -> dict[str, int | float]:
    """Create an empty accumulator for split-level JRC gating diagnostics."""
    return {
        "tiles_with_jrc": 0,
        "tiles_without_jrc": 0,
        "positive_before": 0,
        "negative_before": 0,
        "uncertain_before": 0,
        "positive_after": 0,
        "negative_after": 0,
        "uncertain_after": 0,
        "positive_suppressed_outside_forest": 0,
        "negative_ignored_outside_forest": 0,
        "forest_fraction_sum": 0.0,
    }


def update_label_gating_totals(
    totals: dict[str, int | float],
    diagnostics: dict[str, object],
) -> None:
    """Accumulate split-level JRC gating diagnostics from one tile."""
    gating_summary = diagnostics["gating_summary"]
    assert isinstance(gating_summary, dict)

    if bool(gating_summary["jrc_available"]):
        totals["tiles_with_jrc"] += 1
        forest_fraction = gating_summary["forest_fraction"]
        if forest_fraction is not None:
            totals["forest_fraction_sum"] += float(forest_fraction)
    else:
        totals["tiles_without_jrc"] += 1

    for key in (
        "positive_before",
        "negative_before",
        "uncertain_before",
        "positive_after",
        "negative_after",
        "uncertain_after",
        "positive_suppressed_outside_forest",
        "negative_ignored_outside_forest",
    ):
        totals[key] += int(gating_summary[key])


def finalize_label_gating_totals(totals: dict[str, int | float]) -> dict[str, int | float | None]:
    """Convert accumulated JRC gating totals into a JSON-friendly split summary."""
    tiles_with_jrc = int(totals["tiles_with_jrc"])
    average_forest_fraction = None
    if tiles_with_jrc > 0:
        average_forest_fraction = float(totals["forest_fraction_sum"]) / tiles_with_jrc

    return {
        "tiles_with_jrc": tiles_with_jrc,
        "tiles_without_jrc": int(totals["tiles_without_jrc"]),
        "positive_before": int(totals["positive_before"]),
        "negative_before": int(totals["negative_before"]),
        "uncertain_before": int(totals["uncertain_before"]),
        "positive_after": int(totals["positive_after"]),
        "negative_after": int(totals["negative_after"]),
        "uncertain_after": int(totals["uncertain_after"]),
        "positive_suppressed_outside_forest": int(totals["positive_suppressed_outside_forest"]),
        "negative_ignored_outside_forest": int(totals["negative_ignored_outside_forest"]),
        "average_forest_fraction": average_forest_fraction,
    }


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
    label_gating_summaries: dict[str, dict[str, int | float | None]] = {}
    for output_split, tile_ids, data_split, include_labels in split_plan:
        totals = {"positive": 0, "negative": 0, "uncertain": 0}
        gating_totals = initialize_label_gating_totals()
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
                label_diagnostics = artifact.get("label_diagnostics")
                if isinstance(label_diagnostics, dict):
                    update_label_gating_totals(gating_totals, label_diagnostics)

        summaries[output_split] = totals
        if include_labels:
            label_gating_summaries[output_split] = finalize_label_gating_totals(gating_totals)

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "year": selected_year,
        "train_tiles": fit_tiles,
        "validation_tiles": validation_tiles,
        "stats_path": str(stats_path),
        "summaries": summaries,
        "label_gating_summaries": label_gating_summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    return summary_payload
