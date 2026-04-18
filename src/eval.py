"""Cross-validation harness for the Kaite v1 baseline.

Runs grouped 3-fold cross-validation over Cini's frozen
``cini/splits/split_v1/fold_assignments.csv``. Each fold trains a fresh
``LGBMClassifier`` on the two held-in folds and scores the held-out
fold; metrics are the standard binary scores (precision, recall, F1,
IoU, PR-AUC) at each fold's F1-optimal threshold, plus a per-MGRS
breakdown to surface regional generalisation gaps.

Also writes :file:`reports/baseline_results.md` with the per-fold table,
per-region breakdown, confusion matrix at the chosen threshold, top-20
LightGBM feature importances, and the train-time config + git SHA.

Fallback: if Cini's split file is missing, grouped CV is reconstructed
by MGRS prefix across all tiles found in ``data/makeathon-challenge/labels/train/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    jaccard_score,
    precision_recall_curve,
)

from .data import (
    FEATURES_ROOT,
    build_tile_features,
    feature_names,
    load_tile_manifest,
    resolve_tile_paths,
)
from .train import (
    POSITIVE_THRESHOLD,
    TrainConfig,
    load_training_dataframe,
    prepare_xy,
    train_classifier,
    _git_sha,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = REPO_ROOT / "reports"
SPLIT_ASSIGNMENTS_PATH = REPO_ROOT / "cini" / "splits" / "split_v1" / "fold_assignments.csv"
LEGACY_SPLIT_PATH = REPO_ROOT / "splits" / "mgrs_5fold.json"


@dataclass
class FoldMetrics:
    """Per-fold evaluation summary at the F1-optimal threshold."""

    fold_id: int
    train_tiles: list[str]
    val_tiles: list[str]
    threshold: float
    precision: float
    recall: float
    f1: float
    iou: float
    pr_auc: float
    n_val_rows: int
    n_val_positives: int
    per_region: dict[str, dict[str, float]]


def load_fold_assignments() -> pd.DataFrame:
    """Load Cini's frozen fold assignments.

    Falls back to a grouped-by-MGRS-prefix reconstruction over the
    labelpack tiles if Cini's CSV isn't present. Warns loudly in the
    fallback branch so reviewers notice.
    """
    if SPLIT_ASSIGNMENTS_PATH.exists():
        df = pd.read_csv(SPLIT_ASSIGNMENTS_PATH)
        return df

    logger.warning(
        "%s missing — falling back to MGRS-prefix grouped CV over label tiles.",
        SPLIT_ASSIGNMENTS_PATH,
    )
    label_dir = REPO_ROOT / "data" / "makeathon-challenge" / "labels" / "train" / "radd"
    tiles = sorted({p.stem.replace("radd_", "").replace("_labels", "") for p in label_dir.glob("*.tif")})
    prefixes = sorted({t.split("_")[0] for t in tiles})
    n_folds = min(3, len(prefixes))
    fold_of_prefix = {prefix: i % n_folds for i, prefix in enumerate(prefixes)}
    rows = [
        {
            "tile_id": t,
            "fold_id": fold_of_prefix[t.split("_")[0]],
            "region_id": t.split("_")[0],
        }
        for t in tiles
    ]
    return pd.DataFrame(rows)


def _f1_from_pr(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * p * r / (p + r)
    f1[~np.isfinite(f1)] = 0.0
    return f1


def _region_metrics(
    df_val: pd.DataFrame,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for region, block in df_val.groupby("region_id"):
        idx = block.index.to_numpy()
        y = y_true[idx]
        p = (y_pred_proba[idx] >= threshold).astype(np.int8)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out[str(region)] = {
            "n_rows": int(len(idx)),
            "n_positives": int(y.sum()),
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    return out


def run_fold(
    df: pd.DataFrame,
    fold_id: int,
    fold_assignments: pd.DataFrame,
    config: TrainConfig,
    positive_threshold: float,
):
    """Train on held-in folds, score on held-out fold."""
    val_tiles = fold_assignments.loc[fold_assignments["fold_id"] == fold_id, "tile_id"].tolist()
    train_tiles = fold_assignments.loc[fold_assignments["fold_id"] != fold_id, "tile_id"].tolist()

    train_mask = df["tile_id"].isin(train_tiles)
    val_mask = df["tile_id"].isin(val_tiles)
    df_train = df.loc[train_mask].reset_index(drop=True)
    df_val = df.loc[val_mask].reset_index(drop=True)
    if df_val.empty:
        raise RuntimeError(f"No validation rows for fold {fold_id} — missing features?")

    X_tr, y_tr, w_tr = prepare_xy(df_train, threshold=positive_threshold)
    X_val, y_val, w_val = prepare_xy(df_val, threshold=positive_threshold)

    groups = df_train.loc[X_tr.index, "tile_id"]
    model, info = train_classifier(X_tr, y_tr, w_tr, config, groups=groups)

    proba = model.predict_proba(X_val)[:, 1]
    prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_val, proba, sample_weight=w_val)
    f1_curve = _f1_from_pr(prec_curve, rec_curve)
    best_idx = int(np.argmax(f1_curve[:-1])) if len(thresh_curve) else 0
    best_threshold = float(thresh_curve[best_idx]) if len(thresh_curve) else 0.5
    pred = (proba >= best_threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = jaccard_score(y_val, pred, zero_division=0)
    pr_auc = average_precision_score(y_val, proba, sample_weight=w_val)

    df_val_indexed = df_val.copy()
    df_val_indexed.reset_index(drop=True, inplace=True)
    region_breakdown = _region_metrics(df_val_indexed, y_val, proba, best_threshold)

    metrics = FoldMetrics(
        fold_id=fold_id,
        train_tiles=list(train_tiles),
        val_tiles=list(val_tiles),
        threshold=best_threshold,
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        iou=float(iou),
        pr_auc=float(pr_auc),
        n_val_rows=int(len(y_val)),
        n_val_positives=int(y_val.sum()),
        per_region=region_breakdown,
    )
    importances = dict(zip(X_tr.columns, model.booster_.feature_importance(importance_type="gain")))
    confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics, importances, confusion, info


def run_fold_ungated(
    df_features_cache: pd.DataFrame,
    fold_id: int,
    fold_assignments: pd.DataFrame,
    config: TrainConfig,
    positive_threshold: float,
    manifest: pd.DataFrame,
):
    """Ungated-eval variant: train on the gated sampled cache, score over
    every pixel of each held-out tile's full raster.

    Pixels outside ``train_mask`` are labelled ``0`` (assumed-negative,
    per the leaderboard framing). Threshold selection, per-region
    aggregation, and metric computation otherwise match ``run_fold``.
    """
    val_tiles = fold_assignments.loc[fold_assignments["fold_id"] == fold_id, "tile_id"].tolist()
    train_tiles = fold_assignments.loc[fold_assignments["fold_id"] != fold_id, "tile_id"].tolist()

    df_train = df_features_cache.loc[df_features_cache["tile_id"].isin(train_tiles)].reset_index(
        drop=True
    )
    X_tr, y_tr, w_tr = prepare_xy(df_train, threshold=positive_threshold)
    groups = df_train.loc[X_tr.index, "tile_id"]
    model, info = train_classifier(X_tr, y_tr, w_tr, config, groups=groups)

    region_of_tile = (
        fold_assignments.set_index("tile_id")["region_id"].to_dict()
    )
    per_tile: dict[str, dict[str, object]] = {}
    proba_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    region_chunks: list[np.ndarray] = []

    for tile_id in val_tiles:
        logger.info("fold %d ungated: scoring full raster of tile %s", fold_id, tile_id)
        paths = resolve_tile_paths(tile_id, manifest=manifest)
        features, _names, labels, _ref = build_tile_features(paths)
        F, H, W = features.shape
        X = features.reshape(F, -1).T.astype(np.float32)
        proba = model.predict_proba(X)[:, 1].astype(np.float32)

        train_msk = labels["train_mask"].reshape(-1) == 1
        soft = labels["soft_target"].reshape(-1)
        y = np.zeros(H * W, dtype=np.int8)
        y[train_msk] = (soft[train_msk] >= positive_threshold).astype(np.int8)

        proba_chunks.append(proba)
        y_chunks.append(y)
        region_chunks.append(
            np.full(H * W, region_of_tile.get(tile_id, paths.region_id), dtype=object)
        )
        per_tile[tile_id] = {
            "region_id": region_of_tile.get(tile_id, paths.region_id),
            "n_pixels": int(H * W),
            "n_train_mask": int(train_msk.sum()),
            "n_positive": int(y.sum()),
            "positive_fraction_at_05": float((proba >= 0.5).mean()),
        }

    proba_full = np.concatenate(proba_chunks)
    y_full = np.concatenate(y_chunks)
    region_full = np.concatenate(region_chunks)

    prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_full, proba_full)
    f1_curve = _f1_from_pr(prec_curve, rec_curve)
    best_idx = int(np.argmax(f1_curve[:-1])) if len(thresh_curve) else 0
    best_threshold = float(thresh_curve[best_idx]) if len(thresh_curve) else 0.5
    pred = (proba_full >= best_threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(y_full, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = jaccard_score(y_full, pred, zero_division=0)
    pr_auc = average_precision_score(y_full, proba_full)

    region_breakdown: dict[str, dict[str, float]] = {}
    for region in np.unique(region_full):
        mask = region_full == region
        yr = y_full[mask]
        pr = (proba_full[mask] >= best_threshold).astype(np.int8)
        tp_r = int(((pr == 1) & (yr == 1)).sum())
        fp_r = int(((pr == 1) & (yr == 0)).sum())
        fn_r = int(((pr == 0) & (yr == 1)).sum())
        prec_r = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0.0
        rec_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
        f1_r = 2 * prec_r * rec_r / (prec_r + rec_r) if (prec_r + rec_r) > 0 else 0.0
        region_breakdown[str(region)] = {
            "n_rows": int(mask.sum()),
            "n_positives": int(yr.sum()),
            "precision": prec_r,
            "recall": rec_r,
            "f1": f1_r,
        }

    metrics = FoldMetrics(
        fold_id=fold_id,
        train_tiles=list(train_tiles),
        val_tiles=list(val_tiles),
        threshold=best_threshold,
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        iou=float(iou),
        pr_auc=float(pr_auc),
        n_val_rows=int(len(y_full)),
        n_val_positives=int(y_full.sum()),
        per_region=region_breakdown,
    )
    importances = dict(zip(X_tr.columns, model.booster_.feature_importance(importance_type="gain")))
    confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics, importances, confusion, info, per_tile


def _aggregate_region(folds: list[FoldMetrics]) -> dict[str, dict[str, float]]:
    regions: dict[str, dict[str, float]] = {}
    for fold in folds:
        for region, scores in fold.per_region.items():
            regions.setdefault(region, {"f1_by_fold": [], "n_rows_by_fold": []})
            regions[region]["f1_by_fold"].append(scores["f1"])
            regions[region]["n_rows_by_fold"].append(scores["n_rows"])
    for region, payload in regions.items():
        payload["f1_mean"] = float(np.mean(payload["f1_by_fold"]))
        payload["n_rows_total"] = int(sum(payload["n_rows_by_fold"]))
    return regions


def _top_features(importance_lists: list[dict[str, float]], k: int) -> list[tuple[str, float]]:
    agg: dict[str, float] = {}
    for imp in importance_lists:
        for name, val in imp.items():
            agg[name] = agg.get(name, 0.0) + float(val)
    return sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:k]


def _write_report(
    path: Path,
    folds: list[FoldMetrics],
    confusions: list[dict[str, int]],
    top_features: list[tuple[str, float]],
    config: TrainConfig,
    positive_threshold: float,
    git_sha: str,
    split_source: Path | None,
    ungated: bool = False,
    per_tile_ungated: dict[str, dict[str, object]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [f.threshold for f in folds]
    thresh_summary = {
        "mean": float(np.mean(thresholds)),
        "median": float(np.median(thresholds)),
        "min": float(np.min(thresholds)),
        "max": float(np.max(thresholds)),
    }
    regions = _aggregate_region(folds)
    best_region = max(regions.items(), key=lambda kv: kv[1]["f1_mean"])
    worst_region = min(regions.items(), key=lambda kv: kv[1]["f1_mean"])

    lines: list[str] = []
    heading = (
        "# Baseline v1 — cross-validation results (**ungated**, full-raster)"
        if ungated
        else "# Baseline v1 — cross-validation results"
    )
    lines.append(heading)
    lines.append("")
    if ungated:
        lines.append(
            "> Eval scored every pixel of each held-out tile. Pixels outside "
            "Cini's `train_mask` were labelled `0`. Numbers here track the "
            "leaderboard setup; gated results live in the sibling report."
        )
        lines.append("")
    lines.append(f"- Git SHA: `{git_sha}`")
    lines.append(
        f"- Split source: `{split_source.relative_to(REPO_ROOT) if split_source else 'MGRS-prefix fallback'}`"
    )
    lines.append(f"- Positive threshold (training target): soft_target >= {positive_threshold}")
    lines.append(f"- Classifier: LightGBM binary, {len(feature_names())} features")
    lines.append("")
    lines.append("## Per-fold metrics at F1-optimal threshold")
    lines.append("")
    lines.append("| fold | threshold | precision | recall | F1 | IoU | PR-AUC | rows | positives |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for f in folds:
        lines.append(
            f"| {f.fold_id} | {f.threshold:.3f} | {f.precision:.3f} | {f.recall:.3f} | "
            f"{f.f1:.3f} | {f.iou:.3f} | {f.pr_auc:.3f} | {f.n_val_rows} | {f.n_val_positives} |"
        )
    agg = {
        "precision": np.mean([f.precision for f in folds]),
        "recall": np.mean([f.recall for f in folds]),
        "f1": np.mean([f.f1 for f in folds]),
        "iou": np.mean([f.iou for f in folds]),
        "pr_auc": np.mean([f.pr_auc for f in folds]),
    }
    lines.append(
        f"| **mean** |  | {agg['precision']:.3f} | {agg['recall']:.3f} | {agg['f1']:.3f} | "
        f"{agg['iou']:.3f} | {agg['pr_auc']:.3f} |  |  |"
    )
    lines.append("")
    lines.append("### Submission threshold summary")
    lines.append("")
    lines.append(f"- mean: {thresh_summary['mean']:.3f}")
    lines.append(f"- median: {thresh_summary['median']:.3f}")
    lines.append(f"- min: {thresh_summary['min']:.3f}")
    lines.append(f"- max: {thresh_summary['max']:.3f}")
    lines.append("")
    lines.append("## Per-MGRS region breakdown")
    lines.append("")
    lines.append("| region | mean F1 across folds | total rows |")
    lines.append("| --- | ---: | ---: |")
    for region, scores in sorted(regions.items(), key=lambda kv: kv[1]["f1_mean"], reverse=True):
        lines.append(f"| {region} | {scores['f1_mean']:.3f} | {scores['n_rows_total']} |")
    gap = best_region[1]["f1_mean"] - worst_region[1]["f1_mean"]
    lines.append("")
    lines.append(
        f"**Regional generalisation gap (best − worst F1)**: "
        f"{best_region[0]} ({best_region[1]['f1_mean']:.3f}) − "
        f"{worst_region[0]} ({worst_region[1]['f1_mean']:.3f}) = **{gap:.3f}**"
    )
    lines.append("")
    lines.append("## Confusion matrices per fold (at F1-optimal threshold)")
    lines.append("")
    lines.append("| fold | TN | FP | FN | TP |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for f, c in zip(folds, confusions):
        lines.append(f"| {f.fold_id} | {c['tn']} | {c['fp']} | {c['fn']} | {c['tp']} |")
    lines.append("")
    lines.append("## Top-20 feature importances (total gain across folds)")
    lines.append("")
    lines.append("| rank | feature | gain |")
    lines.append("| --- | --- | ---: |")
    for i, (name, gain) in enumerate(top_features, start=1):
        lines.append(f"| {i} | `{name}` | {gain:,.1f} |")
    lines.append("")
    if ungated and per_tile_ungated:
        lines.append("## Per-tile sanity (ungated, on held-out fold)")
        lines.append("")
        lines.append(
            "| tile | region | n_pixels | train_mask pixels | positives | fraction @ proba≥0.5 |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for tile_id, stats in sorted(per_tile_ungated.items()):
            lines.append(
                f"| {tile_id} | {stats['region_id']} | {stats['n_pixels']} | "
                f"{stats['n_train_mask']} | {stats['n_positive']} | "
                f"{stats['positive_fraction_at_05']:.4f} |"
            )
        lines.append("")
    lines.append("## Training configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(asdict(config), indent=2, sort_keys=True))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n")


def _write_threshold_sidecar(folds: list[FoldMetrics], path: Path) -> None:
    thresholds = [f.threshold for f in folds]
    payload = {
        "per_fold": {str(f.fold_id): f.threshold for f in folds},
        "mean": float(np.mean(thresholds)),
        "median": float(np.median(thresholds)),
        "min": float(np.min(thresholds)),
        "max": float(np.max(thresholds)),
        "per_fold_f1": {str(f.fold_id): f.f1 for f in folds},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_cv(
    tile_ids: list[str],
    fold_assignments: pd.DataFrame,
    config: TrainConfig,
    positive_threshold: float,
    report_path: Path,
    split_source: Path | None,
    ungated: bool = False,
    manifest: pd.DataFrame | None = None,
) -> tuple[list[FoldMetrics], list[dict[str, int]]]:
    df = load_training_dataframe(tile_ids)
    df = df[df["tile_id"].isin(fold_assignments["tile_id"])].reset_index(drop=True)
    fold_ids = sorted(fold_assignments["fold_id"].unique())
    folds: list[FoldMetrics] = []
    confusions: list[dict[str, int]] = []
    importance_lists: list[dict[str, float]] = []
    per_tile_ungated: dict[str, dict[str, object]] = {}

    if ungated and manifest is None:
        raise ValueError("ungated mode requires the tile manifest")

    for fold_id in fold_ids:
        logger.info("running fold %d (%s)", fold_id, "ungated" if ungated else "gated")
        if ungated:
            metrics, importance, confusion, info, per_tile = run_fold_ungated(
                df, fold_id, fold_assignments, config, positive_threshold, manifest
            )
            per_tile_ungated.update(per_tile)
        else:
            metrics, importance, confusion, info = run_fold(
                df, fold_id, fold_assignments, config, positive_threshold
            )
        logger.info(
            "fold %d: F1=%.3f  IoU=%.3f  PR-AUC=%.3f  thr=%.3f  (val rows=%d, positives=%d, best_iter=%s)",
            fold_id,
            metrics.f1,
            metrics.iou,
            metrics.pr_auc,
            metrics.threshold,
            metrics.n_val_rows,
            metrics.n_val_positives,
            info["best_iteration"],
        )
        folds.append(metrics)
        confusions.append(confusion)
        importance_lists.append(importance)

    top = _top_features(importance_lists, k=20)
    _write_report(
        report_path,
        folds,
        confusions,
        top,
        config,
        positive_threshold,
        _git_sha(),
        split_source,
        ungated=ungated,
        per_tile_ungated=per_tile_ungated if ungated else None,
    )
    _write_threshold_sidecar(folds, report_path.with_name(report_path.stem + "_thresholds.json"))
    logger.info("wrote report -> %s", report_path)
    return folds, confusions


def _resolve_tile_ids(manifest: pd.DataFrame, args_tiles: list[str] | None) -> list[str]:
    if args_tiles:
        return list(args_tiles)
    return manifest.loc[
        (manifest["split"] == "train") & manifest["fold_id"].notna(), "tile_id"
    ].tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", nargs="*", help="Subset of train tiles to evaluate")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORTS_ROOT / "baseline_results.md",
    )
    parser.add_argument(
        "--positive-threshold", type=float, default=POSITIVE_THRESHOLD
    )
    parser.add_argument("--n-estimators", type=int, default=TrainConfig.n_estimators)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--num-leaves", type=int, default=TrainConfig.num_leaves)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--ungated",
        action="store_true",
        help=(
            "Score over the full held-out raster (pixels outside train_mask "
            "labelled 0) — matches the leaderboard framing."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    manifest = load_tile_manifest()
    tile_ids = _resolve_tile_ids(manifest, args.tiles)
    fold_assignments = load_fold_assignments()
    fold_assignments = fold_assignments[fold_assignments["tile_id"].isin(tile_ids)].reset_index(
        drop=True
    )
    if fold_assignments.empty:
        raise RuntimeError("No tiles with fold assignments found")

    config = TrainConfig(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        seed=args.seed,
    )
    split_source = SPLIT_ASSIGNMENTS_PATH if SPLIT_ASSIGNMENTS_PATH.exists() else None
    if args.ungated and args.report_path == REPORTS_ROOT / "baseline_results.md":
        args.report_path = REPORTS_ROOT / "baseline_results_ungated.md"
    run_cv(
        tile_ids=tile_ids,
        fold_assignments=fold_assignments,
        config=config,
        positive_threshold=args.positive_threshold,
        report_path=args.report_path,
        split_source=split_source,
        ungated=args.ungated,
        manifest=manifest if args.ungated else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
