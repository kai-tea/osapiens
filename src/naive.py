"""Naive weak-label baselines scored on the ungated eval harness.

Provides an apples-to-apples comparison against the LightGBM baseline's
ungated CV. Each strategy derives its prediction directly from Cini's
labelpack, without any training:

- ``copy_radd``   — predict positive where RADD fired (``radd_alert == 1``).
- ``copy_gladl``  — predict positive where GLAD-L fired.
- ``copy_glads2`` — predict positive where GLAD-S2 fired.
- ``majority_2of3`` — predict positive where at least 2 of the 3 sources
  fired a confident alert (``alert == 1 ∧ conf ≥ 0.5``), matching Cini's
  ``hard_label == 1`` rule.

For PR-AUC we use each source's confidence band as the score
(``{source}_conf``); for the majority strategy the score is the number
of confident votes divided by three.

Truth: ``y = 1`` iff ``train_mask == 1 ∧ soft_target ≥ 0.5``, else ``0``
(pixels outside ``train_mask`` are assumed-negative per the leaderboard
framing). Same definition used by :mod:`src.eval` in ``--ungated`` mode.

Writes :file:`reports/baseline_naive.md` with per-fold + per-MGRS tables
so reviewers can see whether any naive baseline beats the ML model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import average_precision_score, confusion_matrix, jaccard_score

from .data import LABELPACK_NODATA, REPO_ROOT
from .eval import REPORTS_ROOT, SPLIT_ASSIGNMENTS_PATH, load_fold_assignments
from .train import POSITIVE_THRESHOLD, _git_sha

logger = logging.getLogger(__name__)

LABEL_ROOT = REPO_ROOT / "artifacts" / "labels_v1"
LABELPACK_BAND_ORDER = (
    "train_mask",
    "hard_label",
    "seed_mask",
    "soft_target",
    "sample_weight",
    "obs_count",
    "radd_obs",
    "radd_alert",
    "radd_conf",
    "radd_days",
    "gladl_obs",
    "gladl_alert",
    "gladl_conf",
    "gladl_days",
    "glads2_obs",
    "glads2_alert",
    "glads2_conf",
    "glads2_days",
)
STRATEGIES = ("copy_radd", "copy_gladl", "copy_glads2", "majority_2of3")


def read_labelpack(tile_id: str) -> dict[str, np.ndarray]:
    """Load Cini's 18-band labelpack keyed by band name."""
    path = LABEL_ROOT / "tiles" / f"{tile_id}_labelpack.tif"
    if not path.exists():
        raise FileNotFoundError(f"labelpack missing for tile {tile_id}: {path}")
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return {name: arr[i] for i, name in enumerate(LABELPACK_BAND_ORDER)}


def build_tile_predictions(
    tile_id: str, positive_threshold: float
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For one tile return ``{strategy: (y_true, pred, score)}`` arrays."""
    pack = read_labelpack(tile_id)
    train_mask = pack["train_mask"].reshape(-1) == 1
    soft = pack["soft_target"].reshape(-1)

    y_true = np.zeros(soft.shape, dtype=np.int8)
    y_true[train_mask] = (soft[train_mask] >= positive_threshold).astype(np.int8)

    strategies: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for src_name in ("radd", "gladl", "glads2"):
        pred = (pack[f"{src_name}_alert"].reshape(-1) == 1).astype(np.int8)
        score = pack[f"{src_name}_conf"].reshape(-1).astype(np.float32)
        score = np.where(score == LABELPACK_NODATA, 0.0, score)
        strategies[f"copy_{src_name}"] = (y_true, pred, score)

    vote_stack = np.stack(
        [
            (pack[f"{s}_alert"] == 1) & (pack[f"{s}_conf"] >= 0.5)
            for s in ("radd", "gladl", "glads2")
        ],
        axis=0,
    )
    vote_sum = vote_stack.sum(axis=0).reshape(-1)
    strategies["majority_2of3"] = (
        y_true,
        (vote_sum >= 2).astype(np.int8),
        (vote_sum.astype(np.float32) / 3.0),
    )
    return strategies


def _binary_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = float(jaccard_score(y, pred, zero_division=0))
    try:
        pr_auc = float(average_precision_score(y, score))
    except ValueError:
        pr_auc = float("nan")
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "iou": iou,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "positive_fraction_pred": float(pred.mean()),
        "positive_fraction_true": float(y.mean()),
    }


def run_cv(
    fold_assignments: pd.DataFrame,
    positive_threshold: float,
) -> dict:
    """Compute per-fold + per-region metrics for every naive strategy."""
    fold_ids = sorted(fold_assignments["fold_id"].unique())
    region_of_tile = fold_assignments.set_index("tile_id")["region_id"].to_dict()

    per_fold: dict[str, dict[int, dict]] = {s: {} for s in STRATEGIES}
    per_region_fold: dict[str, dict[str, dict[str, list]]] = {s: {} for s in STRATEGIES}

    for fold_id in fold_ids:
        val_tiles = fold_assignments.loc[
            fold_assignments["fold_id"] == fold_id, "tile_id"
        ].tolist()
        logger.info("fold %d: %d val tiles", fold_id, len(val_tiles))

        per_strategy_chunks = {s: {"y": [], "pred": [], "score": [], "region": []} for s in STRATEGIES}
        for tile_id in val_tiles:
            strategies = build_tile_predictions(tile_id, positive_threshold)
            region = region_of_tile[tile_id]
            for strat, (y, pred, score) in strategies.items():
                per_strategy_chunks[strat]["y"].append(y)
                per_strategy_chunks[strat]["pred"].append(pred)
                per_strategy_chunks[strat]["score"].append(score)
                per_strategy_chunks[strat]["region"].append(
                    np.full(y.shape, region, dtype=object)
                )

        for strat, chunks in per_strategy_chunks.items():
            y_full = np.concatenate(chunks["y"])
            pred_full = np.concatenate(chunks["pred"])
            score_full = np.concatenate(chunks["score"])
            region_full = np.concatenate(chunks["region"])

            per_fold[strat][fold_id] = _binary_metrics(y_full, pred_full, score_full)

            for region in np.unique(region_full):
                mask = region_full == region
                m = _binary_metrics(y_full[mask], pred_full[mask], score_full[mask])
                per_region_fold[strat].setdefault(str(region), {}).setdefault(
                    "f1_by_fold", []
                ).append(m["f1"])
                per_region_fold[strat][str(region)].setdefault("n_rows_by_fold", []).append(
                    int(mask.sum())
                )

    for strat, regions in per_region_fold.items():
        for region, payload in regions.items():
            payload["f1_mean"] = float(np.mean(payload["f1_by_fold"]))
            payload["n_rows_total"] = int(sum(payload["n_rows_by_fold"]))

    return {"per_fold": per_fold, "per_region": per_region_fold}


def _aggregate(strategy_folds: dict[int, dict]) -> dict[str, float]:
    keys = ("precision", "recall", "f1", "iou", "pr_auc")
    return {k: float(np.mean([m[k] for m in strategy_folds.values()])) for k in keys}


def _write_report(results: dict, report_path: Path, git_sha: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Naive weak-label baselines — ungated CV")
    lines.append("")
    lines.append(f"- Git SHA: `{git_sha}`")
    lines.append(
        "- Truth: `soft_target >= 0.5 ∧ train_mask == 1`; otherwise `0`. "
        "Same harness as `reports/baseline_results_ungated.md`."
    )
    lines.append("")

    lines.append("## Per-strategy aggregate (mean across folds)")
    lines.append("")
    lines.append("| strategy | precision | recall | F1 | IoU | PR-AUC |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for strat in STRATEGIES:
        agg = _aggregate(results["per_fold"][strat])
        lines.append(
            f"| `{strat}` | {agg['precision']:.3f} | {agg['recall']:.3f} | "
            f"{agg['f1']:.3f} | {agg['iou']:.3f} | {agg['pr_auc']:.3f} |"
        )
    lines.append("")

    for strat in STRATEGIES:
        lines.append(f"## Per-fold — `{strat}`")
        lines.append("")
        lines.append(
            "| fold | precision | recall | F1 | IoU | PR-AUC | pred pos frac | true pos frac |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for fold_id, m in sorted(results["per_fold"][strat].items()):
            lines.append(
                f"| {fold_id} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                f"{m['iou']:.3f} | {m['pr_auc']:.3f} | "
                f"{m['positive_fraction_pred']:.4f} | {m['positive_fraction_true']:.4f} |"
            )
        lines.append("")

    lines.append("## Per-MGRS F1 (mean across folds)")
    lines.append("")
    regions = sorted(
        {r for regions in results["per_region"].values() for r in regions}
    )
    header = "| region | " + " | ".join(f"`{s}`" for s in STRATEGIES) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(STRATEGIES)) + " |"
    lines.append(header)
    lines.append(sep)
    for region in regions:
        row = [region]
        for strat in STRATEGIES:
            val = results["per_region"][strat].get(region, {}).get("f1_mean")
            row.append(f"{val:.3f}" if val is not None else "–")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-threshold", type=float, default=POSITIVE_THRESHOLD
    )
    parser.add_argument(
        "--report-path", type=Path, default=REPORTS_ROOT / "baseline_naive.md"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    fold_assignments = load_fold_assignments()
    results = run_cv(fold_assignments, args.positive_threshold)
    _write_report(results, args.report_path, _git_sha())
    logger.info("wrote report -> %s", args.report_path)

    # also emit a machine-readable JSON sidecar
    json_path = args.report_path.with_name(args.report_path.stem + ".json")
    json_path.write_text(
        json.dumps(
            {
                strat: {str(k): v for k, v in folds.items()}
                for strat, folds in results["per_fold"].items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
