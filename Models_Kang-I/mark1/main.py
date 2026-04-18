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
    from .data import list_train_tiles, load_embeddings, sample_training_tile
    from .features import build_feature_names, build_features_for_locations
    from .model import choose_worker_count, compute_metrics_from_cache, detect_hardware, predict_tile, save_model, train_model
except ModuleNotFoundError as exc:
    if exc.name in {"rasterio", "sklearn", "numpy"}:
        raise SystemExit(
            "Missing Python dependency: "
            f"'{exc.name}'. Install the project dependencies first with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    raise
except ImportError:
    from data import list_train_tiles, load_embeddings, sample_training_tile
    from features import build_feature_names, build_features_for_locations
    from model import choose_worker_count, compute_metrics_from_cache, detect_hardware, predict_tile, save_model, train_model


DEFAULT_DATA_ROOT = Path("data/makeathon-challenge")


def split_tiles(tile_ids: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    if len(tile_ids) <= 1:
        return tile_ids, []

    validation_count = max(1, int(round(len(tile_ids) * val_fraction)))
    validation_count = min(validation_count, len(tile_ids) - 1)
    validation_tiles = sorted(tile_ids)[-validation_count:]
    training_tiles = sorted(tile_id for tile_id in tile_ids if tile_id not in validation_tiles)
    return training_tiles, validation_tiles


def concatenate_arrays(parts: list[np.ndarray], empty_shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if parts:
        return np.concatenate(parts, axis=0)
    return np.zeros(empty_shape, dtype=dtype)


def build_or_load_tile_cache(
    tile_id: str,
    tile_role: str,
    data_root: Path,
    cache_dir: Path,
    apply_blob_filter: bool,
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
        max_samples_per_tile=max_samples_per_tile,
    )
    features = build_features_for_locations(tile["embeddings"], tile["rows"], tile["cols"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=features.astype(np.float32, copy=False),
        y=tile["labels"].astype(np.int8, copy=False),
    )
    return cache_path


def build_or_load_tile_cache_from_kwargs(kwargs: dict) -> Path:
    return build_or_load_tile_cache(**kwargs)


def run_pipeline(
    data_root: Path,
    output_dir: Path,
    val_fraction: float = 0.25,
    threshold: float = 0.5,
    max_samples_per_tile: int | None = None,
    apply_blob_filter: bool = False,
    validation_samples_per_tile: int | None = None,
    train_epochs: int = 2,
    batch_size: int | None = None,
    num_workers: int | None = None,
    resume: bool = True,
    force_rebuild_cache: bool = False,
) -> dict:
    tile_ids = list_train_tiles(data_root)
    train_tiles, validation_tiles = split_tiles(tile_ids, val_fraction)

    if not tile_ids:
        raise RuntimeError("No train tiles found")

    first_tile = sample_training_tile(
        data_root=data_root,
        tile_id=tile_ids[0],
        apply_blob_filter=apply_blob_filter,
        max_samples_per_tile=1,
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
        epochs=train_epochs,
        batch_size=batch_size,
        requested_workers=num_workers,
        resume=resume,
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

    prediction_outputs = []
    for tile_id in validation_tiles:
        embeddings, embedding_meta = load_embeddings(data_root, tile_id)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(model_bundle, output_dir / "model_bundle.pkl")

    summary = {
        "dataset_limits": {
            "train_max_samples_per_tile": max_samples_per_tile,
            "validation_max_samples_per_tile": validation_samples_per_tile,
            "train_tile_count": len(train_tiles),
            "validation_tile_count": len(validation_tiles),
        },
        "cache_build": {
            "num_workers": cache_workers,
            "feature_cache_dir": str(cache_dir),
        },
        "training_runtime": training_runtime,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "prediction_outputs": prediction_outputs,
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
    parser.add_argument("--train_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--force_rebuild_cache", action="store_true")
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Expected dataset at '{args.data_root}'")

    summary = run_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        threshold=args.threshold,
        max_samples_per_tile=args.max_samples_per_tile,
        apply_blob_filter=args.apply_blob_filter,
        validation_samples_per_tile=args.validation_samples_per_tile,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resume=not args.no_resume,
        force_rebuild_cache=args.force_rebuild_cache,
    )

    print(json.dumps(summary["train_metrics"], indent=2))
    print(json.dumps(summary["validation_metrics"], indent=2))


if __name__ == "__main__":
    main()
