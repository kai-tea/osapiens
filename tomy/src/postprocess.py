"""Post-processing for per-pixel deforestation probability rasters.

Turns a noisy float probability raster into a clean binary raster that
(a) respects the challenge rules (post-2020 event AND forest in 2020) and
(b) matches the real spatial structure of deforestation (contiguous patches,
not isolated pixels).

Pipeline per tile
-----------------
    prob (H,W) float32
      │  1. threshold  → bool
      │  2. morphological opening (removes stray single/few-pixel positives)
      │  3. morphological closing (fills pinholes inside real patches)
      │  4. connected-component filter (drops components below min_component_px)
      │  5. AND with 2020 forest mask
      ▼
    binary (H,W) uint8   → fed to submission_utils.raster_to_geojson
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

DEFAULT_THRESHOLD = 0.5
DEFAULT_OPENING_KERNEL = 3
DEFAULT_CLOSING_KERNEL = 3
# 10m pixels -> 5 px ≈ 500 m² (well below the 0.5 ha = 5000 m² polygon filter,
# but useful to kill salt noise before it survives morphology).
DEFAULT_MIN_COMPONENT_PX = 5


@dataclass
class PostprocessConfig:
    threshold: float = DEFAULT_THRESHOLD
    opening_kernel: int = DEFAULT_OPENING_KERNEL  # 0 to disable
    closing_kernel: int = DEFAULT_CLOSING_KERNEL  # 0 to disable
    min_component_px: int = DEFAULT_MIN_COMPONENT_PX  # 0 to disable

    @classmethod
    def from_dict(cls, d: dict) -> "PostprocessConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _square_kernel(size: int) -> np.ndarray:
    return np.ones((size, size), dtype=bool)


def postprocess(
    prob: np.ndarray,
    forest_mask: np.ndarray | None = None,
    config: PostprocessConfig | None = None,
) -> np.ndarray:
    """Apply the full post-processing chain.

    Args:
        prob: (H, W) float32 model output in [0, 1].
        forest_mask: (H, W) bool. If given, final AND restricts predictions
            to pixels that were forest in 2020 (challenge requirement).
        config: thresholds and kernel sizes.

    Returns:
        (H, W) uint8 binary raster, ready to write as a GeoTIFF.
    """
    cfg = config or PostprocessConfig()

    binary = prob > cfg.threshold

    if cfg.opening_kernel and cfg.opening_kernel > 1:
        binary = ndimage.binary_opening(binary, structure=_square_kernel(cfg.opening_kernel))
    if cfg.closing_kernel and cfg.closing_kernel > 1:
        binary = ndimage.binary_closing(binary, structure=_square_kernel(cfg.closing_kernel))

    if cfg.min_component_px and cfg.min_component_px > 1:
        binary = filter_components_by_size(binary, cfg.min_component_px)

    if forest_mask is not None:
        binary &= forest_mask.astype(bool)

    return binary.astype(np.uint8)


def filter_components_by_size(binary: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected components smaller than ``min_px`` (8-connectivity)."""
    if min_px <= 1:
        return binary
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    labels, n = ndimage.label(binary, structure=structure)
    if n == 0:
        return binary
    sizes = ndimage.sum_labels(binary, labels, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_px
    return keep[labels]


# ---------- Threshold tuning ----------


def best_threshold(
    prob: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    candidates: np.ndarray | None = None,
    metric: str = "iou",
) -> tuple[float, float]:
    """Return the (threshold, score) that maximises the chosen metric on one tile.

    The ``metric`` is ``"iou"`` or ``"f1"``. ``mask`` selects which pixels count
    (e.g. the forest-gated region).
    """
    if candidates is None:
        candidates = np.linspace(0.1, 0.9, 33)

    p = prob.ravel()
    t = target.astype(bool).ravel()
    if mask is not None:
        m = mask.astype(bool).ravel()
        p = p[m]
        t = t[m]

    best: tuple[float, float] = (float("nan"), -1.0)
    for thr in candidates:
        pred = p > thr
        tp = int((pred & t).sum())
        fp = int((pred & ~t).sum())
        fn = int((~pred & t).sum())
        if metric == "iou":
            d = tp + fp + fn
            score = tp / d if d else 0.0
        elif metric == "f1":
            denom = 2 * tp + fp + fn
            score = 2 * tp / denom if denom else 0.0
        else:
            raise ValueError(f"unknown metric: {metric}")
        if score > best[1]:
            best = (float(thr), float(score))
    return best


def best_threshold_aggregate(
    probs_and_targets: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    candidates: np.ndarray | None = None,
    metric: str = "iou",
) -> tuple[float, float]:
    """Same as ``best_threshold`` but pools TP/FP/FN across tiles (micro-averaged)."""
    if candidates is None:
        candidates = np.linspace(0.1, 0.9, 33)

    flats: list[tuple[np.ndarray, np.ndarray]] = []
    for prob, target, mask in probs_and_targets:
        p = prob.ravel()
        t = target.astype(bool).ravel()
        if mask is not None:
            m = mask.astype(bool).ravel()
            p = p[m]
            t = t[m]
        flats.append((p, t))

    best: tuple[float, float] = (float("nan"), -1.0)
    for thr in candidates:
        tp = fp = fn = 0
        for p, t in flats:
            pred = p > thr
            tp += int((pred & t).sum())
            fp += int((pred & ~t).sum())
            fn += int((~pred & t).sum())
        if metric == "iou":
            d = tp + fp + fn
            score = tp / d if d else 0.0
        elif metric == "f1":
            denom = 2 * tp + fp + fn
            score = 2 * tp / denom if denom else 0.0
        else:
            raise ValueError(f"unknown metric: {metric}")
        if score > best[1]:
            best = (float(thr), float(score))
    return best
