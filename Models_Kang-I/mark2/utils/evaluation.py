"""Readable evaluation helpers for Mark 2 binary classification outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


DEFAULT_THRESHOLD_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def compute_confusion_counts(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, int]:
    """Compute confusion matrix counts at a given probability threshold."""
    predictions = probabilities >= threshold
    labels_bool = labels == 1

    true_positive = int(np.count_nonzero(predictions & labels_bool))
    true_negative = int(np.count_nonzero((~predictions) & (~labels_bool)))
    false_positive = int(np.count_nonzero(predictions & (~labels_bool)))
    false_negative = int(np.count_nonzero((~predictions) & labels_bool))

    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def summarize_class_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Summarize predicted probabilities separately for positive and negative labels."""
    summary: dict[str, float] = {}
    for class_value, class_name in ((1, "positive"), (0, "negative")):
        class_probabilities = probabilities[labels == class_value]
        if class_probabilities.size == 0:
            summary[f"{class_name}_probability_mean"] = 0.0
            summary[f"{class_name}_probability_p10"] = 0.0
            summary[f"{class_name}_probability_p50"] = 0.0
            summary[f"{class_name}_probability_p90"] = 0.0
            continue

        summary[f"{class_name}_probability_mean"] = float(class_probabilities.mean())
        summary[f"{class_name}_probability_p10"] = float(np.percentile(class_probabilities, 10))
        summary[f"{class_name}_probability_p50"] = float(np.percentile(class_probabilities, 50))
        summary[f"{class_name}_probability_p90"] = float(np.percentile(class_probabilities, 90))
    return summary


def compute_threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Compute explicit binary classification metrics for one threshold."""
    counts = compute_confusion_counts(labels=labels, probabilities=probabilities, threshold=threshold)
    total = int(labels.size)
    positive_count = int(np.count_nonzero(labels == 1))
    negative_count = int(np.count_nonzero(labels == 0))

    precision_denominator = counts["true_positive"] + counts["false_positive"]
    recall_denominator = counts["true_positive"] + counts["false_negative"]
    accuracy_denominator = total

    precision = counts["true_positive"] / precision_denominator if precision_denominator else 0.0
    recall = counts["true_positive"] / recall_denominator if recall_denominator else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (counts["true_positive"] + counts["true_negative"]) / accuracy_denominator if accuracy_denominator else 0.0
    positive_prediction_rate = float(np.mean(probabilities >= threshold)) if total else 0.0

    return {
        "sample_count": total,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_prediction_rate": positive_prediction_rate,
        **counts,
    }


def compute_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute average precision for binary labels and probabilities."""
    if labels.size == 0:
        return 0.0
    return float(average_precision_score(labels, probabilities))


def compute_threshold_sweep(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_SWEEP,
) -> list[dict[str, float | int]]:
    """Evaluate a fixed threshold list in deterministic order."""
    return [compute_threshold_metrics(labels=labels, probabilities=probabilities, threshold=value) for value in thresholds]


def select_best_threshold(sweep_metrics: list[dict[str, float | int]]) -> dict[str, float | int] | None:
    """Select the threshold with the best F1 score, breaking ties toward lower thresholds."""
    if not sweep_metrics:
        return None
    return max(sweep_metrics, key=lambda row: (float(row["f1"]), -float(row["threshold"])))


def build_validation_report(
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    validation_loss: float,
    default_threshold: float,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_SWEEP,
) -> dict[str, object]:
    """Build a full validation report with metrics, summaries, and threshold sweep."""
    base_metrics = compute_threshold_metrics(labels=labels, probabilities=probabilities, threshold=default_threshold)
    probability_summary = summarize_class_probabilities(labels=labels, probabilities=probabilities)
    threshold_sweep = compute_threshold_sweep(labels=labels, probabilities=probabilities, thresholds=thresholds)
    selected_threshold = select_best_threshold(threshold_sweep)

    return {
        "validation_loss": float(validation_loss),
        "default_threshold_metrics": {
            **base_metrics,
            "average_precision": compute_average_precision(labels=labels, probabilities=probabilities),
            **probability_summary,
        },
        "threshold_sweep": threshold_sweep,
        "selected_threshold": selected_threshold,
    }


def save_json_report(report: dict[str, object], output_path: Path) -> None:
    """Write a JSON report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
