"""Shared prediction schema validation and simple binary metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import TEST_PREDICTION_COLUMNS, VALIDATION_PREDICTION_COLUMNS


def validate_prediction_schema(df: pd.DataFrame, *, split: str) -> None:
    expected = VALIDATION_PREDICTION_COLUMNS if split == "validation" else TEST_PREDICTION_COLUMNS
    if tuple(df.columns) != expected:
        raise ValueError(
            f"{split} prediction schema mismatch. Expected columns={list(expected)} "
            f"actual columns={list(df.columns)}"
        )
    missing = [column for column in expected if column not in df.columns]
    extra = [column for column in df.columns if column not in expected]
    if missing or extra:
        raise ValueError(
            f"{split} prediction schema mismatch. Missing={missing or 'none'} extra={extra or 'none'}"
        )


def binary_metrics(y_true: pd.Series, score: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    pred = (score.astype(float).to_numpy() >= threshold).astype(int)
    if y.size == 0:
        raise ValueError("Cannot compute metrics for empty predictions")

    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / y.size
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
