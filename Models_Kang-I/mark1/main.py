"""Hackathon baseline pipeline.

Flow:
load data -> build labels -> train model -> evaluate -> predict
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

try:
    from .data import list_tiles, load_embeddings, sample_training_tile
    from .features import build_feature_names, build_features_for_locations
    from .model import (
        choose_worker_count,
        collect_probabilities_from_cache,
        compute_metrics_from_cache,
        compute_threshold_sweep,
        detect_hardware,
        predict_tile,
        save_model,
        select_best_threshold,
        train_model,
    )
    from .submission import export_thresholded_prediction_set, summarize_submission_outputs
except ModuleNotFoundError as exc:
    if exc.name in {"rasterio", "sklearn", "numpy"}:
        raise SystemExit(
            "Missing Python dependency: "
            f"'{exc.name}'. Install the project dependencies first with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    raise
except ImportError:
    from data import list_tiles, load_embeddings, sample_training_tile
    from features import build_feature_names, build_features_for_locations
    from model import (
        choose_worker_count,
        collect_probabilities_from_cache,
        compute_metrics_from_cache,
        compute_threshold_sweep,
        detect_hardware,
        predict_tile,
        save_model,
        select_best_threshold,
        train_model,
    )
    from submission import export_thresholded_prediction_set, summarize_submission_outputs


DEFAULT_DATA_ROOT = Path("data/makeathon-challenge")
DEFAULT_THRESHOLD_SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_MODEL_TYPE = "hist_gbdt"
DEFAULT_LABEL_POLICY = "legacy_vote"
DEFAULT_TREE_MAX_ITER = 400
DEFAULT_TREE_MAX_LEAF_NODES = 63
DEFAULT_TREE_LEARNING_RATE = 0.05


def split_tiles(tile_ids: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    if len(tile_ids) <= 1:
        return tile_ids, []

    validation_count = max(1, int(round(len(tile_ids) * val_fraction)))
    validation_count = min(validation_count, len(tile_ids) - 1)
    validation_tiles = sorted(tile_ids)[-validation_count:]
    training_tiles = sorted(tile_id for tile_id in tile_ids if tile_id not in validation_tiles)
    return training_tiles, validation_tiles


def build_or_load_tile_cache(
    tile_id: str,
    tile_role: str,
    data_root: Path,
    cache_dir: Path,
    apply_blob_filter: bool,
    label_blob_min_size: int,
    positive_min_votes: int,
    negative_max_votes: int,
    label_policy: str,
    max_samples_per_tile: int | None,
    force_rebuild_cache: bool,
) -> Path:
    cache_path = cache_dir / f"{tile_role}_{tile_id}.npz"
    if cache_path.exists() and not force_rebuild_cache:
        return cache_path

    tile = sample_training_tile(
        data_root=data_root,
        tile_id=tile_id,
        apply_blob_filter=apply_blob_filter,
        min_blob_size=label_blob_min_size,
        max_samples_per_tile=max_samples_per_tile,
        positive_min_votes=positive_min_votes,
        negative_max_votes=negative_max_votes,
        label_policy=label_policy,
    )
    features = build_features_for_locations(tile["embeddings"], tile["rows"], tile["cols"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=features.astype(np.float32, copy=False),
        y=tile["labels"].astype(np.int8, copy=False),
        sample_weight=tile["sample_weights"].astype(np.float32, copy=False),
    )
    return cache_path


def build_or_load_tile_cache_from_kwargs(kwargs: dict) -> Path:
    return build_or_load_tile_cache(**kwargs)


def load_metadata_tile_ids(metadata_path: Path) -> list[str]:
    if not metadata_path.exists():
        return []

    with metadata_path.open() as handle:
        geojson = json.load(handle)

    names = []
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        tile_id = properties.get("name")
        if tile_id:
            names.append(tile_id)
    return sorted(names)


def parse_threshold_sweep(raw_thresholds: str | None) -> list[float]:
    if not raw_thresholds:
        return list(DEFAULT_THRESHOLD_SWEEP)
    thresholds = [float(value.strip()) for value in raw_thresholds.split(",") if value.strip()]
    unique_thresholds = sorted(set(thresholds))
    for threshold in unique_thresholds:
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError(f"Threshold sweep values must be between 0 and 1, got {threshold}")
    return unique_thresholds


def run_pipeline(
    data_root: Path,
    output_dir: Path,
    val_fraction: float = 0.25,
    threshold: float = 0.5,
    max_samples_per_tile: int | None = None,
    apply_blob_filter: bool = False,
    label_blob_min_size: int = 4,
    label_positive_min_votes: int = 2,
    label_negative_max_votes: int = 0,
    label_policy: str = DEFAULT_LABEL_POLICY,
    validation_samples_per_tile: int | None = None,
    train_epochs: int = 2,
    batch_size: int | None = None,
    num_workers: int | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
    tree_max_iter: int = DEFAULT_TREE_MAX_ITER,
    tree_max_leaf_nodes: int = DEFAULT_TREE_MAX_LEAF_NODES,
    tree_learning_rate: float = DEFAULT_TREE_LEARNING_RATE,
    resume: bool = True,
    force_rebuild_cache: bool = False,
    predict_test: bool = False,
    export_submission: bool = False,
    submission_min_area_ha: float = 0.5,
    threshold_sweep: list[float] | None = None,
    threshold_selection_metric: str = "f1",
    apply_prediction_blob_filter: bool = False,
    prediction_blob_min_size: int = 4,
) -> dict:
    selected_threshold_sweep = threshold_sweep or list(DEFAULT_THRESHOLD_SWEEP)
    tile_ids = list_tiles(data_root, split="train")
    train_tiles, validation_tiles = split_tiles(tile_ids, val_fraction)

    if not tile_ids:
        raise RuntimeError("No train tiles found")

    first_tile = sample_training_tile(
        data_root=data_root,
        tile_id=tile_ids[0],
        apply_blob_filter=apply_blob_filter,
        min_blob_size=label_blob_min_size,
        max_samples_per_tile=1,
        positive_min_votes=label_positive_min_votes,
        negative_max_votes=label_negative_max_votes,
        label_policy=label_policy,
    )
    embedding_dim = first_tile["embeddings"][2020].shape[0]
    feature_names = build_feature_names(embedding_dim)
    hardware_info = detect_hardware()
    cache_workers = choose_worker_count(hardware_info, num_workers)

    cache_dir = output_dir / "feature_cache"
    train_cache_jobs = [
        {
            "tile_id": tile_id,
            "tile_role": "train",
            "data_root": data_root,
            "cache_dir": cache_dir,
            "apply_blob_filter": apply_blob_filter,
            "label_blob_min_size": label_blob_min_size,
            "positive_min_votes": label_positive_min_votes,
            "negative_max_votes": label_negative_max_votes,
            "label_policy": label_policy,
            "max_samples_per_tile": max_samples_per_tile,
            "force_rebuild_cache": force_rebuild_cache,
        }
        for tile_id in train_tiles
    ]
    validation_cache_jobs = [
        {
            "tile_id": tile_id,
            "tile_role": "validation",
            "data_root": data_root,
            "cache_dir": cache_dir,
            "apply_blob_filter": apply_blob_filter,
            "label_blob_min_size": label_blob_min_size,
            "positive_min_votes": label_positive_min_votes,
            "negative_max_votes": label_negative_max_votes,
            "label_policy": label_policy,
            "max_samples_per_tile": validation_samples_per_tile,
            "force_rebuild_cache": force_rebuild_cache,
        }
        for tile_id in validation_tiles
    ]

    if cache_workers == 1:
        train_cache_paths = [build_or_load_tile_cache(**job) for job in train_cache_jobs]
        validation_cache_paths = [build_or_load_tile_cache(**job) for job in validation_cache_jobs]
    else:
        with ProcessPoolExecutor(max_workers=cache_workers) as executor:
            train_cache_paths = list(executor.map(build_or_load_tile_cache_from_kwargs, train_cache_jobs))
            validation_cache_paths = list(executor.map(build_or_load_tile_cache_from_kwargs, validation_cache_jobs))

    if not train_cache_paths:
        raise RuntimeError("No training cache files were created")

    model_bundle, training_runtime = train_model(
        train_cache_paths=train_cache_paths,
        feature_names=feature_names,
        output_dir=output_dir,
        model_type=model_type,
        epochs=train_epochs,
        batch_size=batch_size,
        requested_workers=num_workers,
        resume=resume,
        tree_max_iter=tree_max_iter,
        tree_max_leaf_nodes=tree_max_leaf_nodes,
        tree_learning_rate=tree_learning_rate,
    )

    train_metrics = compute_metrics_from_cache(
        model_bundle=model_bundle,
        cache_paths=train_cache_paths,
        threshold=threshold,
        batch_size=batch_size,
        include_auc=False,
    )
    validation_metrics = (
        compute_metrics_from_cache(
            model_bundle=model_bundle,
            cache_paths=validation_cache_paths,
            threshold=threshold,
            batch_size=batch_size,
            include_auc=True,
        )
        if validation_cache_paths
        else {}
    )
    validation_threshold_sweep = []
    recommended_threshold = None
    if validation_cache_paths:
        validation_labels, validation_probabilities = collect_probabilities_from_cache(
            model_bundle=model_bundle,
            cache_paths=validation_cache_paths,
            batch_size=batch_size,
        )
        validation_threshold_sweep = compute_threshold_sweep(
            labels=validation_labels,
            probabilities=validation_probabilities,
            thresholds=selected_threshold_sweep,
        )
        recommended_threshold = select_best_threshold(
            validation_threshold_sweep,
            metric_name=threshold_selection_metric,
        )

    prediction_outputs = []
    for tile_id in validation_tiles:
        embeddings, embedding_meta = load_embeddings(data_root, tile_id, split="train")
        prediction_outputs.append(
            predict_tile(
                model_bundle=model_bundle,
                embeddings=embeddings,
                embedding_meta=embedding_meta,
                output_dir=output_dir / "validation_predictions",
                tile_id=tile_id,
                threshold=threshold,
            )
        )

    test_prediction_outputs = []
    test_tile_ids = list_tiles(data_root, split="test") if predict_test or export_submission else []
    metadata_test_tile_ids = load_metadata_tile_ids(data_root / "metadata" / "test_tiles.geojson")

    if export_submission and not predict_test:
        raise ValueError("--export_submission requires --predict_test so binary test rasters are created in the same run")

    if predict_test:
        for tile_id in test_tile_ids:
            embeddings, embedding_meta = load_embeddings(data_root, tile_id, split="test")
            test_prediction_outputs.append(
                predict_tile(
                    model_bundle=model_bundle,
                    embeddings=embeddings,
                    embedding_meta=embedding_meta,
                    output_dir=output_dir / "test_predictions",
                    tile_id=tile_id,
                    threshold=threshold,
                )
            )

    validation_submission_outputs = export_thresholded_prediction_set(
        prediction_outputs=prediction_outputs,
        output_dir=output_dir / f"validation_submission_threshold_{str(threshold).replace('.', 'p')}",
        threshold=threshold,
        min_area_ha=submission_min_area_ha,
        allow_empty=True,
        apply_blob_filter=apply_prediction_blob_filter,
        min_blob_size=prediction_blob_min_size,
    )
    recommended_validation_submission_outputs = []
    if prediction_outputs and recommended_threshold is not None and recommended_threshold["threshold"] != threshold:
        recommended_validation_submission_outputs = export_thresholded_prediction_set(
            prediction_outputs=prediction_outputs,
            output_dir=output_dir / f"validation_submission_threshold_{str(recommended_threshold['threshold']).replace('.', 'p')}",
            threshold=recommended_threshold["threshold"],
            min_area_ha=submission_min_area_ha,
            allow_empty=True,
            apply_blob_filter=apply_prediction_blob_filter,
            min_blob_size=prediction_blob_min_size,
        )

    threshold_selected_for_test = recommended_threshold["threshold"] if recommended_threshold is not None else threshold
    recommended_test_submission_outputs = []
    if test_prediction_outputs:
        recommended_test_submission_outputs = export_thresholded_prediction_set(
            prediction_outputs=test_prediction_outputs,
            output_dir=output_dir / f"test_submission_threshold_{str(threshold_selected_for_test).replace('.', 'p')}",
            threshold=threshold_selected_for_test,
            min_area_ha=submission_min_area_ha,
            allow_empty=True,
            apply_blob_filter=apply_prediction_blob_filter,
            min_blob_size=prediction_blob_min_size,
        )

    submission_outputs = recommended_test_submission_outputs if export_submission else []

    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(model_bundle, output_dir / "model_bundle.pkl")
    run_config = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "val_fraction": val_fraction,
        "threshold": threshold,
        "threshold_sweep": selected_threshold_sweep,
        "threshold_selection_metric": threshold_selection_metric,
        "max_samples_per_tile": max_samples_per_tile,
        "validation_samples_per_tile": validation_samples_per_tile,
        "apply_blob_filter": apply_blob_filter,
        "label_blob_min_size": label_blob_min_size,
        "label_positive_min_votes": label_positive_min_votes,
        "label_negative_max_votes": label_negative_max_votes,
        "label_policy": label_policy,
        "apply_prediction_blob_filter": apply_prediction_blob_filter,
        "prediction_blob_min_size": prediction_blob_min_size,
        "train_epochs": train_epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "model_type": model_type,
        "tree_max_iter": tree_max_iter,
        "tree_max_leaf_nodes": tree_max_leaf_nodes,
        "tree_learning_rate": tree_learning_rate,
        "resume": resume,
        "force_rebuild_cache": force_rebuild_cache,
        "predict_test": predict_test,
        "export_submission": export_submission,
        "submission_min_area_ha": submission_min_area_ha,
    }
    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(run_config, handle, indent=2)

    summary = {
        "run_config_path": str(output_dir / "run_config.json"),
        "dataset_limits": {
            "train_max_samples_per_tile": max_samples_per_tile,
            "validation_max_samples_per_tile": validation_samples_per_tile,
            "train_tile_count": len(train_tiles),
            "validation_tile_count": len(validation_tiles),
            "test_tile_count": len(test_tile_ids),
        },
        "cache_build": {
            "num_workers": cache_workers,
            "feature_cache_dir": str(cache_dir),
        },
        "data_config": {
            "embedding_years": list(range(2020, 2024)),
            "embedding_years_note": "Mark 1 intentionally uses 2020-2023 AEF embeddings even though the downloaded dataset includes 2024-2025.",
        },
        "label_policy": {
            "label_policy": label_policy,
            "positive_min_votes": label_positive_min_votes,
            "negative_max_votes": label_negative_max_votes,
            "apply_label_blob_filter": apply_blob_filter,
            "label_blob_min_size": label_blob_min_size,
        },
        "prediction_postprocessing": {
            "apply_prediction_blob_filter": apply_prediction_blob_filter,
            "prediction_blob_min_size": prediction_blob_min_size,
            "submission_min_area_ha": submission_min_area_ha,
        },
        "test_tile_metadata_check": {
            "metadata_tile_count": len(metadata_test_tile_ids),
            "metadata_matches_aef_test_tiles": metadata_test_tile_ids == test_tile_ids if metadata_test_tile_ids else None,
        },
        "training_runtime": training_runtime,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "threshold_analysis": {
            "selection_metric": threshold_selection_metric,
            "sweep_thresholds": selected_threshold_sweep,
            "validation_threshold_sweep": validation_threshold_sweep,
            "recommended_submission_threshold": recommended_threshold,
        },
        "prediction_outputs": prediction_outputs,
        "validation_submission_analysis": {
            "current_threshold": threshold,
            "current_threshold_outputs": validation_submission_outputs,
            "current_threshold_summary": summarize_submission_outputs(validation_submission_outputs),
            "recommended_threshold_outputs": recommended_validation_submission_outputs,
            "recommended_threshold_summary": summarize_submission_outputs(recommended_validation_submission_outputs),
        },
        "test_prediction_outputs": test_prediction_outputs,
        "submission_outputs": submission_outputs,
        "submission_summary": summarize_submission_outputs(submission_outputs),
        "recommended_test_submission_outputs": recommended_test_submission_outputs,
        "recommended_test_submission_summary": summarize_submission_outputs(recommended_test_submission_outputs),
    }

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Hackathon-ready per-pixel baseline")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/mark1"))
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_samples_per_tile", type=int, default=None)
    parser.add_argument("--validation_samples_per_tile", type=int, default=None)
    parser.add_argument("--apply_blob_filter", action="store_true")
    parser.add_argument("--label_blob_min_size", type=int, default=4)
    parser.add_argument("--label_positive_min_votes", type=int, default=2)
    parser.add_argument("--label_negative_max_votes", type=int, default=0)
    parser.add_argument("--label_policy", type=str, default=DEFAULT_LABEL_POLICY)
    parser.add_argument("--train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--tree_max_iter", type=int, default=DEFAULT_TREE_MAX_ITER)
    parser.add_argument("--tree_max_leaf_nodes", type=int, default=DEFAULT_TREE_MAX_LEAF_NODES)
    parser.add_argument("--tree_learning_rate", type=float, default=DEFAULT_TREE_LEARNING_RATE)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--force_rebuild_cache", action="store_true")
    parser.add_argument("--predict_test", action="store_true")
    parser.add_argument("--export_submission", action="store_true")
    parser.add_argument("--submission_min_area_ha", type=float, default=0.5)
    parser.add_argument("--threshold_sweep", type=str, default=None)
    parser.add_argument("--threshold_selection_metric", type=str, default="f1")
    parser.add_argument("--apply_prediction_blob_filter", action="store_true")
    parser.add_argument("--prediction_blob_min_size", type=int, default=4)
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Expected dataset at '{args.data_root}'")
    if args.model_type not in {"sgd", "hist_gbdt"}:
        raise ValueError("model_type must be one of: sgd, hist_gbdt")
    if args.label_policy not in {"confidence_date", "legacy_vote"}:
        raise ValueError("label_policy must be one of: confidence_date, legacy_vote")
    if args.threshold_selection_metric not in {"f1", "balanced_accuracy", "precision", "recall"}:
        raise ValueError("threshold_selection_metric must be one of: f1, balanced_accuracy, precision, recall")
    if args.label_positive_min_votes < 1 or args.label_positive_min_votes > 3:
        raise ValueError("label_positive_min_votes must be between 1 and 3")
    if args.label_negative_max_votes < 0 or args.label_negative_max_votes >= args.label_positive_min_votes:
        raise ValueError("label_negative_max_votes must be >= 0 and less than label_positive_min_votes")

    summary = run_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        threshold=args.threshold,
        max_samples_per_tile=args.max_samples_per_tile,
        apply_blob_filter=args.apply_blob_filter,
        label_blob_min_size=args.label_blob_min_size,
        label_positive_min_votes=args.label_positive_min_votes,
        label_negative_max_votes=args.label_negative_max_votes,
        label_policy=args.label_policy,
        validation_samples_per_tile=args.validation_samples_per_tile,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_type=args.model_type,
        tree_max_iter=args.tree_max_iter,
        tree_max_leaf_nodes=args.tree_max_leaf_nodes,
        tree_learning_rate=args.tree_learning_rate,
        resume=not args.no_resume,
        force_rebuild_cache=args.force_rebuild_cache,
        predict_test=args.predict_test,
        export_submission=args.export_submission,
        submission_min_area_ha=args.submission_min_area_ha,
        threshold_sweep=parse_threshold_sweep(args.threshold_sweep),
        threshold_selection_metric=args.threshold_selection_metric,
        apply_prediction_blob_filter=args.apply_prediction_blob_filter,
        prediction_blob_min_size=args.prediction_blob_min_size,
    )

    print(json.dumps(summary["train_metrics"], indent=2))
    print(json.dumps(summary["validation_metrics"], indent=2))


if __name__ == "__main__":
    main()
