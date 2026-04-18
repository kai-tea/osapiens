"""Pixel-level evaluation metrics + geographic CV aggregation.

Since no clean ground truth exists, we define a proxy target as
``fuse_post2020(..., min_sources=2)["confident_pos"]`` — pixels flagged by at
least two independent sources. This is what the leaderboard approximates; the
jury cares about generalisation across the folds, not absolute numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cv_split import fold_iter
from .label_fusion import fuse_post2020


@dataclass
class PixelMetrics:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else float("nan")

    @property
    def iou(self) -> float:
        d = self.tp + self.fp + self.fn
        return self.tp / d if d else float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "iou": self.iou,
            "support_pos": self.tp + self.fn,
            "support_neg": self.tn + self.fp,
        }

    def __add__(self, other: "PixelMetrics") -> "PixelMetrics":
        return PixelMetrics(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn)


def pixel_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> PixelMetrics:
    """Binary confusion metrics. ``mask`` (bool) selects pixels that count."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if mask is not None:
        m = mask.astype(bool)
        pred = pred[m]
        target = target[m]
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    tn = int((~pred & ~target).sum())
    return PixelMetrics(tp, fp, fn, tn)


def target_for_tile(tile_id: str, split: str = "train", min_sources: int = 2) -> np.ndarray:
    """Proxy ground truth: post-2020 confident_pos, gated by 2020 forest mask."""
    fused = fuse_post2020(tile_id, split=split, gate_by_forest=True, min_sources=min_sources)
    return fused["confident_pos"].astype(bool)


def evaluate_fold(predict_fn, tile_ids: list[str], split: str = "train") -> tuple[PixelMetrics, dict[str, dict]]:
    """Aggregate pixel metrics across ``tile_ids``.

    ``predict_fn(tile_id) -> np.ndarray[bool]`` must produce a prediction on the
    tile's canonical grid.
    """
    total = PixelMetrics(0, 0, 0, 0)
    per_tile: dict[str, dict] = {}
    for tile in tile_ids:
        pred = predict_fn(tile).astype(bool)
        target = target_for_tile(tile, split=split)
        if pred.shape != target.shape:
            raise ValueError(f"{tile}: pred shape {pred.shape} != target {target.shape}")
        m = pixel_metrics(pred, target)
        per_tile[tile] = m.as_dict()
        total = total + m
    return total, per_tile


def cv_evaluate(
    predict_fn_factory,
    tiles: list[str],
    n_folds: int = 5,
    split: str = "train",
) -> dict:
    """Run geographic CV. ``predict_fn_factory(train_tiles) -> predict_fn``."""
    fold_results = []
    for fold_idx, train, val in fold_iter(tiles, n_folds):
        predict_fn = predict_fn_factory(train)
        total, per_tile = evaluate_fold(predict_fn, val, split=split)
        fold_results.append(
            {
                "fold": fold_idx,
                "n_train": len(train),
                "n_val": len(val),
                "aggregate": total.as_dict(),
                "per_tile": per_tile,
            }
        )
    # Macro-average across folds
    macro = {
        k: float(np.mean([fr["aggregate"][k] for fr in fold_results if np.isfinite(fr["aggregate"][k])]))
        for k in ("precision", "recall", "f1", "iou")
    }
    return {"folds": fold_results, "macro": macro}
