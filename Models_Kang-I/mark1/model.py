from __future__ import annotations

import json
import os
import pickle
import shutil
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from .data import build_valid_mask, write_raster
    from .features import build_features_for_locations
except ImportError:
    from data import build_valid_mask, write_raster
    from features import build_features_for_locations


@dataclass
class ModelBundle:
    scaler: StandardScaler | None
    classifier: Any
    feature_names: list[str]
    hardware_info: dict
    training_config: dict


def _read_linux_cpu_vendor() -> str | None:
    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        return None

    for line in cpuinfo_path.read_text().splitlines():
        if line.lower().startswith("vendor_id"):
            return line.split(":", 1)[1].strip()
    return None


def detect_hardware() -> dict:
    rocm_path = Path("/opt/rocm")
    rocm_detected = rocm_path.exists() or shutil.which("rocminfo") is not None or Path("/dev/kfd").exists()

    memory_gb = None
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        try:
            memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            memory_gb = round(memory_bytes / (1024**3), 2)
        except (OSError, ValueError):
            memory_gb = None

    return {
        "cpu_count": os.cpu_count(),
        "cpu_vendor": _read_linux_cpu_vendor(),
        "rocm_detected": rocm_detected,
        "rocm_path": str(rocm_path) if rocm_detected else None,
        "mixed_precision_supported": False,
        "accelerator_used": "cpu",
        "system_memory_gb": memory_gb,
    }


def choose_batch_size(hardware_info: dict, requested_batch_size: int | None) -> int:
    if requested_batch_size is not None:
        return requested_batch_size

    memory_gb = hardware_info.get("system_memory_gb")
    if memory_gb is None:
        return 50_000
    if memory_gb >= 64:
        return 200_000
    if memory_gb >= 32:
        return 100_000
    return 50_000


def choose_worker_count(hardware_info: dict, requested_workers: int | None) -> int:
    if requested_workers is not None:
        return max(1, requested_workers)

    cpu_count = hardware_info.get("cpu_count") or 1
    memory_gb = hardware_info.get("system_memory_gb")
    if memory_gb is not None and memory_gb <= 16:
        return 1
    return max(1, min(4, cpu_count // 2 or 1))


def _iter_npz_batches(npz_path: Path, batch_size: int):
    with np.load(npz_path) as data:
        features = data["X"]
        labels = data["y"]
        sample_weights = data["sample_weight"] if "sample_weight" in data.files else None
        for start in range(0, labels.shape[0], batch_size):
            stop = min(start + batch_size, labels.shape[0])
            batch_weights = sample_weights[start:stop] if sample_weights is not None else None
            yield features[start:stop], labels[start:stop], batch_weights


def _count_labels(cache_paths: list[Path]) -> dict:
    positive_count = 0
    negative_count = 0
    sample_count = 0

    for cache_path in cache_paths:
        with np.load(cache_path) as data:
            labels = data["y"]
            sample_count += int(labels.size)
            positive_count += int((labels == 1).sum())
            negative_count += int((labels == 0).sum())

    return {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def _compute_balanced_class_weights(label_summary: dict) -> dict[int, float]:
    sample_count = int(label_summary["sample_count"])
    negative_count = int(label_summary["negative_count"])
    positive_count = int(label_summary["positive_count"])
    if sample_count <= 0 or negative_count <= 0 or positive_count <= 0:
        raise ValueError("Balanced class weights require both positive and negative training samples")

    class_count = 2
    return {
        0: sample_count / (class_count * negative_count),
        1: sample_count / (class_count * positive_count),
    }


def _save_training_log(log_path: Path, payload: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def save_model(model_bundle: ModelBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model_bundle, handle)


def load_model(path: Path) -> ModelBundle:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_full_cache_dataset(cache_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    feature_parts = []
    label_parts = []

    for cache_path in cache_paths:
        with np.load(cache_path) as data:
            feature_parts.append(data["X"].astype(np.float32, copy=False))
            label_parts.append(data["y"].astype(np.int8, copy=False))

    if not feature_parts:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int8)

    return np.concatenate(feature_parts, axis=0), np.concatenate(label_parts, axis=0)


def train_model(
    train_cache_paths: list[Path],
    feature_names: list[str],
    output_dir: Path,
    model_type: str = "sgd",
    epochs: int = 2,
    batch_size: int | None = None,
    random_seed: int = 42,
    requested_workers: int | None = None,
    resume: bool = True,
    tree_max_iter: int = 200,
    tree_max_leaf_nodes: int = 31,
    tree_learning_rate: float = 0.1,
) -> tuple[ModelBundle, dict]:
    hardware_info = detect_hardware()
    effective_batch_size = choose_batch_size(hardware_info, batch_size)
    effective_workers = choose_worker_count(hardware_info, requested_workers)
    label_summary = _count_labels(train_cache_paths)
    class_weights = _compute_balanced_class_weights(label_summary)

    training_config = {
        "model_type": model_type,
        "epochs": epochs,
        "batch_size": effective_batch_size,
        "gradient_accumulation": "not_applicable_cpu_incremental_training",
        "num_workers": effective_workers,
        "resume_enabled": resume,
        "periodic_checkpoints": True,
        "streaming_from_disk": True,
        "class_weight": class_weights,
        "tree_max_iter": tree_max_iter,
        "tree_max_leaf_nodes": tree_max_leaf_nodes,
        "tree_learning_rate": tree_learning_rate,
    }

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = output_dir / "training_log.jsonl"
    latest_checkpoint_path = checkpoints_dir / "latest_model.pkl"

    if model_type == "sgd" and resume and latest_checkpoint_path.exists():
        model_bundle = load_model(latest_checkpoint_path)
        start_epoch = int(model_bundle.training_config.get("completed_epochs", 0))
        if model_bundle.training_config.get("batch_size") != effective_batch_size:
            model_bundle.training_config["batch_size"] = effective_batch_size
    elif model_type == "sgd":
        scaler = StandardScaler()
        classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            learning_rate="optimal",
            class_weight=class_weights,
            random_state=random_seed,
        )
        model_bundle = ModelBundle(
            scaler=scaler,
            classifier=classifier,
            feature_names=feature_names,
            hardware_info=hardware_info,
            training_config={**training_config, "completed_epochs": 0},
        )
        start_epoch = 0

        for cache_path in train_cache_paths:
            for features, _labels, _sample_weights in _iter_npz_batches(cache_path, effective_batch_size):
                model_bundle.scaler.partial_fit(features)

        _save_training_log(
            training_log_path,
            {
                "stage": "scaler_fit_complete",
                "cache_files": len(train_cache_paths),
                "batch_size": effective_batch_size,
                "hardware_info": hardware_info,
            },
        )
        rng = np.random.default_rng(random_seed)
        classes = np.array([0, 1], dtype=np.int8)

        for epoch_index in range(start_epoch, epochs):
            shuffled_paths = list(train_cache_paths)
            rng.shuffle(shuffled_paths)

            for cache_path in shuffled_paths:
                for features, labels, sample_weights in _iter_npz_batches(cache_path, effective_batch_size):
                    scaled_features = model_bundle.scaler.transform(features)
                    model_bundle.classifier.partial_fit(
                        scaled_features,
                        labels,
                        classes=classes,
                        sample_weight=sample_weights,
                    )

            model_bundle.training_config["completed_epochs"] = epoch_index + 1
            epoch_checkpoint_path = checkpoints_dir / f"model_epoch_{epoch_index + 1}.pkl"
            save_model(model_bundle, epoch_checkpoint_path)
            save_model(model_bundle, latest_checkpoint_path)

            _save_training_log(
                training_log_path,
                {
                    "stage": "epoch_complete",
                    "epoch": epoch_index + 1,
                    "epochs_total": epochs,
                    "checkpoint_path": str(epoch_checkpoint_path),
                },
            )
    elif model_type == "hist_gbdt":
        features, labels = _load_full_cache_dataset(train_cache_paths)
        intrinsic_weights = np.where(labels == 1, class_weights[1], class_weights[0]).astype(np.float32, copy=False)
        cached_weights = _load_full_cache_sample_weights(train_cache_paths)
        sample_weight = intrinsic_weights * cached_weights
        classifier = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=tree_learning_rate,
            max_iter=tree_max_iter,
            max_leaf_nodes=tree_max_leaf_nodes,
            random_state=random_seed,
            early_stopping=False,
        )
        classifier.fit(features, labels, sample_weight=sample_weight)
        model_bundle = ModelBundle(
            scaler=None,
            classifier=classifier,
            feature_names=feature_names,
            hardware_info=hardware_info,
            training_config={
                **training_config,
                "completed_epochs": None,
                "resume_enabled": False,
                "streaming_from_disk": False,
            },
        )
        save_model(model_bundle, latest_checkpoint_path)
        _save_training_log(
            training_log_path,
            {
                "stage": "hist_gbdt_fit_complete",
                "sample_count": int(labels.size),
                "feature_count": int(features.shape[1]) if features.ndim == 2 else 0,
                "checkpoint_path": str(latest_checkpoint_path),
            },
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    training_summary = {
        "hardware_info": hardware_info,
        "training_config": model_bundle.training_config,
        "train_cache_files": len(train_cache_paths),
        **label_summary,
    }
    return model_bundle, training_summary


def predict_probabilities(model_bundle: ModelBundle, features: np.ndarray) -> np.ndarray:
    if model_bundle.scaler is not None:
        features = model_bundle.scaler.transform(features)
    probabilities = model_bundle.classifier.predict_proba(features)[:, 1]
    return probabilities.astype(np.float32, copy=False)


def compute_metrics(model_bundle: ModelBundle, X: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    probabilities = predict_probabilities(model_bundle, X)
    metrics = _compute_threshold_metrics(y, probabilities, threshold)
    metrics["pr_auc"] = float(average_precision_score(y, probabilities))
    metrics["roc_auc"] = float(roc_auc_score(y, probabilities)) if len(np.unique(y)) > 1 else None
    return metrics


def compute_metrics_from_cache(
    model_bundle: ModelBundle,
    cache_paths: list[Path],
    threshold: float,
    batch_size: int | None = None,
    include_auc: bool = True,
) -> dict:
    effective_batch_size = batch_size or int(model_bundle.training_config.get("batch_size", 50_000))
    labels, probabilities = collect_probabilities_from_cache(
        model_bundle=model_bundle,
        cache_paths=cache_paths,
        batch_size=effective_batch_size,
    )
    metrics = _compute_threshold_metrics(labels, probabilities, threshold)
    if include_auc and labels.size:
        metrics["pr_auc"] = float(average_precision_score(labels, probabilities))
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else None
    else:
        metrics["pr_auc"] = None
        metrics["roc_auc"] = None
    return metrics


def collect_probabilities_from_cache(
    model_bundle: ModelBundle,
    cache_paths: list[Path],
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    effective_batch_size = batch_size or int(model_bundle.training_config.get("batch_size", 50_000))
    probability_parts = []
    label_parts = []

    for cache_path in cache_paths:
        for features, labels, _sample_weights in _iter_npz_batches(cache_path, effective_batch_size):
            probability_parts.append(predict_probabilities(model_bundle, features))
            label_parts.append(labels.astype(np.int8, copy=False))

    if not label_parts:
        return np.zeros((0,), dtype=np.int8), np.zeros((0,), dtype=np.float32)

    return (
        np.concatenate(label_parts, axis=0),
        np.concatenate(probability_parts, axis=0),
    )


def _load_full_cache_sample_weights(cache_paths: list[Path]) -> np.ndarray:
    weight_parts = []
    for cache_path in cache_paths:
        with np.load(cache_path) as data:
            if "sample_weight" in data.files:
                weight_parts.append(data["sample_weight"].astype(np.float32, copy=False))
            else:
                weight_parts.append(np.ones_like(data["y"], dtype=np.float32))
    if not weight_parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(weight_parts, axis=0)


def compute_threshold_sweep(labels: np.ndarray, probabilities: np.ndarray, thresholds: list[float]) -> list[dict]:
    return [_compute_threshold_metrics(labels, probabilities, threshold) for threshold in thresholds]


def select_best_threshold(sweep_metrics: list[dict], metric_name: str = "f1") -> dict | None:
    if not sweep_metrics:
        return None
    return max(sweep_metrics, key=lambda row: (row.get(metric_name) or 0.0, -(row["threshold"])))


def _compute_threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0

    return {
        "sample_count": int(labels.size),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "predicted_positive_rate": float(predictions.mean()) if predictions.size else 0.0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def predict_tile(
    model_bundle: ModelBundle,
    embeddings: dict[int, np.ndarray],
    embedding_meta: dict,
    output_dir: Path,
    tile_id: str,
    threshold: float,
) -> dict:
    valid_mask = build_valid_mask(embeddings)
    rows, cols = np.where(valid_mask)

    probability_map = np.full((embedding_meta["height"], embedding_meta["width"]), np.nan, dtype=np.float32)
    binary_map = np.full((embedding_meta["height"], embedding_meta["width"]), 255, dtype=np.uint8)

    if rows.size:
        features = build_features_for_locations(embeddings, rows, cols)
        probabilities = predict_probabilities(model_bundle, features)
        probability_map[rows, cols] = probabilities
        binary_map[rows, cols] = (probabilities >= threshold).astype(np.uint8)

    probability_path = output_dir / f"{tile_id}_probability.tif"
    binary_path = output_dir / f"{tile_id}_binary.tif"
    write_raster(probability_path, probability_map, embedding_meta, dtype="float32", nodata=np.nan)
    write_raster(binary_path, binary_map, embedding_meta, dtype="uint8", nodata=255)

    return {
        "tile_id": tile_id,
        "probability_raster": str(probability_path),
        "binary_raster": str(binary_path),
    }
