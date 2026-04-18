"""Training, ungated 3-fold CV, and self-training for the autoloop MLP.

Mirrors the contract of src.eval.run_fold_ungated: pixels outside
Cini's train_mask are labelled 0 at eval time, and the Hansen GFC 2020
forest gate is applied to the probability raster before scoring.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.optim as optim
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    jaccard_score,
    precision_recall_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from src.data import build_tile_features, resolve_tile_paths
from src.masks import DEFAULT_TREECOVER_THRESHOLD, forest_mask_2020

from .io import (
    REPO_ROOT,
    aef_column_mask,
    feature_names,
    label_targets,
    load_fold_assignments,
    load_tile_manifest,
    load_train_frame,
    per_region_weights,
)
from .model import ModelConfig, PixelMLP, combined_loss

logger = logging.getLogger(__name__)

REPORTS_ROOT = REPO_ROOT / "reports"
MODELS_ROOT = REPO_ROOT / "artifacts" / "models_autoloop"
POSITIVE_THRESHOLD = 0.5
DEFAULT_EPOCHS = 20
DEFAULT_BATCH = 8192
DEFAULT_LR = 1e-3


@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = 0.3
    aef_drop_prob: float = 0.5
    iou_weight: float = 0.5
    pos_weight: float = 3.0  # bumps recall vs BCE default
    lr: float = DEFAULT_LR
    weight_decay: float = 1e-4
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH
    seed: int = 42


@dataclass
class FoldResult:
    fold_id: int
    val_tiles: list[str]
    threshold: float
    precision: float
    recall: float
    f1: float
    iou: float
    pr_auc: float
    positive_fraction: float
    per_region: dict[str, dict[str, float]] = field(default_factory=dict)


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _build_model(cfg: TrainConfig, feature_columns: list[str]) -> PixelMLP:
    mcfg = ModelConfig(
        in_dim=len(feature_columns),
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        aef_mask=tuple(bool(b) for b in aef_column_mask(feature_columns)),
        aef_drop_prob=cfg.aef_drop_prob,
    )
    return PixelMLP(mcfg)


def _train_one(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    cfg: TrainConfig,
    feature_columns: list[str],
) -> PixelMLP:
    device = _device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
        torch.from_numpy(w).float(),
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda"
    )
    model = _build_model(cfg, feature_columns).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    for ep in range(cfg.epochs):
        model.train()
        total = 0.0
        n_batches = 0
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            logits = model(xb)
            loss = combined_loss(logits, yb, wb, cfg.pos_weight, cfg.iou_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n_batches += 1
        sched.step()
        logger.info("epoch %d/%d  loss=%.4f", ep + 1, cfg.epochs, total / max(1, n_batches))
    return model


@torch.no_grad()
def _predict_df(model: PixelMLP, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    device = _device()
    model.eval()
    out = np.empty(X.shape[0], dtype=np.float32)
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i : i + batch]).float().to(device)
        p = torch.sigmoid(model(xb)).detach().cpu().numpy()
        out[i : i + batch] = p
    return out


_TILE_FEATURE_CACHE: dict[str, tuple[np.ndarray, dict, dict]] = {}


def _get_tile_features(
    tile_id: str, manifest: pd.DataFrame
) -> tuple[np.ndarray, dict, dict]:
    """Cache the expensive full-raster feature extraction (NaN-filled X)."""
    if tile_id not in _TILE_FEATURE_CACHE:
        paths = resolve_tile_paths(tile_id, manifest=manifest)
        features, names, labels, ref_profile = build_tile_features(paths)
        assert names == feature_names()
        F, H, W = features.shape
        X = np.nan_to_num(
            features.reshape(F, -1).T.astype(np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        _TILE_FEATURE_CACHE[tile_id] = (X, ref_profile, labels)
    return _TILE_FEATURE_CACHE[tile_id]


@torch.no_grad()
def _predict_tile_full_raster(
    model: PixelMLP,
    tile_id: str,
    manifest: pd.DataFrame,
    batch: int = 262144,
) -> tuple[np.ndarray, dict, dict]:
    """Return (proba (H, W), ref_profile, labels_dict) for a full raster."""
    X, ref_profile, labels = _get_tile_features(tile_id, manifest)
    H = ref_profile["height"]
    W = ref_profile["width"]
    proba = _predict_df(model, X, batch=batch)
    return proba.reshape(H, W), ref_profile, labels


def _f1_from_pr(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * p * r / (p + r)
    f1[~np.isfinite(f1)] = 0.0
    return f1


def _fold_eval(
    model: PixelMLP,
    val_tiles: list[str],
    manifest: pd.DataFrame,
    fold_id: int,
    treecover_threshold: int,
) -> FoldResult:
    proba_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    region_chunks: list[np.ndarray] = []

    for tid in val_tiles:
        logger.info("fold %d ungated: scoring tile %s", fold_id, tid)
        proba_hw, ref_profile, labels = _predict_tile_full_raster(model, tid, manifest)
        forest = forest_mask_2020(ref_profile, treecover_threshold=treecover_threshold)
        proba = (proba_hw * forest).reshape(-1).astype(np.float32)

        train_msk = labels["train_mask"].reshape(-1) == 1
        soft = labels["soft_target"].reshape(-1)
        y = np.zeros_like(proba, dtype=np.int8)
        y[train_msk] = (soft[train_msk] >= POSITIVE_THRESHOLD).astype(np.int8)

        region = resolve_tile_paths(tid, manifest=manifest).region_id
        proba_chunks.append(proba)
        y_chunks.append(y)
        region_chunks.append(np.full(proba.shape, region, dtype=object))

    proba_full = np.concatenate(proba_chunks)
    y_full = np.concatenate(y_chunks)
    region_full = np.concatenate(region_chunks)

    prec_curve, rec_curve, thr_curve = precision_recall_curve(y_full, proba_full)
    f1_curve = _f1_from_pr(prec_curve, rec_curve)
    best_idx = int(np.argmax(f1_curve[:-1])) if len(thr_curve) else 0
    threshold = float(thr_curve[best_idx]) if len(thr_curve) else 0.5
    pred = (proba_full >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(y_full, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = float(jaccard_score(y_full, pred, zero_division=0))
    pr_auc = float(average_precision_score(y_full, proba_full))

    per_region: dict[str, dict[str, float]] = {}
    for region in np.unique(region_full):
        mask = region_full == region
        yr = y_full[mask]
        pr = (proba_full[mask] >= threshold).astype(np.int8)
        tp_r = int(((pr == 1) & (yr == 1)).sum())
        fp_r = int(((pr == 1) & (yr == 0)).sum())
        fn_r = int(((pr == 0) & (yr == 1)).sum())
        prec_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0.0
        rec_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
        f1_r = 2 * prec_r * rec_r / (prec_r + rec_r) if (prec_r + rec_r) > 0 else 0.0
        per_region[str(region)] = {
            "precision": prec_r,
            "recall": rec_r,
            "f1": f1_r,
            "n_rows": int(mask.sum()),
        }

    return FoldResult(
        fold_id=fold_id,
        val_tiles=val_tiles,
        threshold=threshold,
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        iou=iou,
        pr_auc=pr_auc,
        positive_fraction=float(pred.mean()),
        per_region=per_region,
    )


def run_ungated_cv(
    cfg: TrainConfig,
    tile_ids: Iterable[str] | None = None,
    treecover_threshold: int = DEFAULT_TREECOVER_THRESHOLD,
) -> tuple[list[FoldResult], list[PixelMLP]]:
    manifest = load_tile_manifest()
    fold_assignments = load_fold_assignments()
    if tile_ids is not None:
        fold_assignments = fold_assignments[fold_assignments["tile_id"].isin(list(tile_ids))]

    feats = feature_names()
    results: list[FoldResult] = []
    fold_models: list[PixelMLP] = []

    for fold_id in sorted(fold_assignments["fold_id"].unique()):
        train_tiles = fold_assignments.loc[
            fold_assignments["fold_id"] != fold_id, "tile_id"
        ].tolist()
        val_tiles = fold_assignments.loc[
            fold_assignments["fold_id"] == fold_id, "tile_id"
        ].tolist()
        logger.info(
            "fold %d: training on %d tiles, validating on %s",
            fold_id,
            len(train_tiles),
            val_tiles,
        )
        df = load_train_frame(train_tiles)
        X = df[feats].to_numpy().astype(np.float32)
        y, w = label_targets(df, POSITIVE_THRESHOLD)
        model = _train_one(X, y, w, cfg, feats)
        result = _fold_eval(model, val_tiles, manifest, fold_id, treecover_threshold)
        logger.info(
            "fold %d: F1=%.3f  IoU=%.3f  PR-AUC=%.3f  thr=%.3f",
            fold_id,
            result.f1,
            result.iou,
            result.pr_auc,
            result.threshold,
        )
        results.append(result)
        fold_models.append(model)
    return results, fold_models


def train_full(cfg: TrainConfig, tile_ids: list[str]) -> tuple[PixelMLP, dict]:
    """Train on all labeled tiles — no held-out fold. For test-tile inference."""
    df = load_train_frame(tile_ids)
    feats = feature_names()
    X = df[feats].to_numpy().astype(np.float32)
    y, w = label_targets(df, POSITIVE_THRESHOLD)
    model = _train_one(X, y, w, cfg, feats)
    info = {
        "n_pixels": int(len(df)),
        "n_positives": int(y.sum()),
        "tile_ids": list(tile_ids),
        "config": asdict(cfg),
    }
    return model, info


def save_checkpoint(model: PixelMLP, info: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_cfg": asdict(model.cfg),
            "info": info,
        },
        path,
    )
    (path.with_suffix(".json")).write_text(json.dumps(info, indent=2, default=str) + "\n")


def load_checkpoint(path: Path) -> tuple[PixelMLP, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**blob["model_cfg"])
    model = PixelMLP(mcfg)
    model.load_state_dict(blob["state_dict"])
    model.to(_device())
    return model, blob["info"]


def generate_pseudo_labels(
    base_model: PixelMLP,
    tile_ids: list[str],
    pos_threshold: float = 0.7,
    neg_threshold: float = 0.1,
) -> pd.DataFrame:
    """Re-label confident pixels where model output agrees with weak evidence.

    For each training tile, read the cached parquet plus the labelpack, run
    the base model, and promote/demote rows where the model is very confident
    AND at least one weak source agrees (for positives) or the pixel is
    definitively not-forest / low-signal (for negatives). Conflicts keep
    the original soft_target.
    """
    from src.naive import read_labelpack

    feats = feature_names()
    frames: list[pd.DataFrame] = []
    for tid in tile_ids:
        from src.data import load_tile_features as _load

        df = _load(tid).dropna(subset=feats).reset_index(drop=True)
        X = df[feats].to_numpy().astype(np.float32)
        proba = _predict_df(base_model, X)

        pack = read_labelpack(tid)
        # Evidence channels at the sampled (row, col) positions.
        rows = df["row"].to_numpy()
        cols = df["col"].to_numpy()

        def pick(band: str) -> np.ndarray:
            return pack[band][rows, cols]

        radd_alert = pick("radd_alert") == 1
        gladl_alert = pick("gladl_alert") == 1
        glads2_alert = pick("glads2_alert") == 1
        any_alert = radd_alert | gladl_alert | glads2_alert
        majority_alert = (
            radd_alert.astype(int) + gladl_alert.astype(int) + glads2_alert.astype(int)
        ) >= 2

        promote_pos = (proba > pos_threshold) & any_alert
        demote_neg = (proba < neg_threshold) & ~any_alert

        soft = df["soft_target"].to_numpy().astype(np.float32).copy()
        soft[promote_pos] = np.maximum(soft[promote_pos], 0.9)
        soft[demote_neg] = np.minimum(soft[demote_neg], 0.05)
        # Upweight confident agreements so they drive the retrain.
        w = df["sample_weight"].to_numpy().astype(np.float32).copy()
        w[promote_pos | majority_alert] *= 1.5

        df = df.copy()
        df["soft_target"] = soft
        df["sample_weight"] = w
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def self_train(
    base_model: PixelMLP,
    cfg: TrainConfig,
    tile_ids: list[str],
    pos_threshold: float = 0.7,
    neg_threshold: float = 0.1,
) -> tuple[PixelMLP, dict]:
    """Run one round of pseudo-labeling + retrain on the augmented frame."""
    df = generate_pseudo_labels(base_model, tile_ids, pos_threshold, neg_threshold)
    df = df.dropna(subset=feature_names()).reset_index(drop=True)
    df["mgrs"] = df["region_id"].astype(str)
    feats = feature_names()
    X = df[feats].to_numpy().astype(np.float32)
    y = (df["soft_target"].to_numpy() >= POSITIVE_THRESHOLD).astype(np.float32)
    w = df["sample_weight"].to_numpy().astype(np.float32)
    w_region = per_region_weights(df)
    w = (w * w_region).astype(np.float32)

    model = _train_one(X, y, w, cfg, feats)
    info = {
        "n_pixels": int(len(df)),
        "n_positives": int(y.sum()),
        "tile_ids": list(tile_ids),
        "config": asdict(cfg),
        "mode": "self_trained",
        "pos_threshold": pos_threshold,
        "neg_threshold": neg_threshold,
    }
    return model, info
