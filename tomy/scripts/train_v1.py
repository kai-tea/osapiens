"""Train V1 baseline on a fold, validate on held-out tiles, sweep thresholds.

Modes
-----
--fold N       : train on CV-fold N's train set, validate on its val tiles,
                 report per-threshold IoU/F1, save model + metrics.
--fold all     : train on every available train tile (final submission model).

Outputs
-------
models/v1_fold{N}.joblib        # trained model
reports/v1_fold{N}_metrics.json # per-threshold val metrics (fold mode only)
reports/v1_fold{N}_summary.txt  # human-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tomy.src.baseline_v1 import (
    BaselineConfig,
    build_training_set,
    predict_proba_max,
    save_model,
    train_model,
)
from tomy.src.cv_split import fold_iter
from tomy.src.data_loader import list_tiles
from tomy.src.evaluate import PixelMetrics, pixel_metrics
from tomy.src.label_fusion import forest_mask_2020
from tomy.src.evaluate import target_for_tile
from tomy.src.postprocess import best_threshold_aggregate


def filter_tiles_with_data(tiles: list[str], split: str = "train") -> list[str]:
    """Skip tiles that don't yet have any S2 scene on disk (dataset still downloading)."""
    from tomy.src.data_loader import S2_DIR

    keep = []
    for t in tiles:
        if list((S2_DIR / split / f"{t}__s2_l2a").glob("*.tif")):
            keep.append(t)
    return keep


def _evaluate_on_val(model, val_tiles: list[str], cfg: BaselineConfig, eval_gate: str = "jrc") -> dict:
    """Run inference on val tiles, sweep thresholds, return metrics.

    ``eval_gate`` picks the forest definition used to build the evaluation
    target and the validity mask, independent of ``cfg.forest_source`` (which
    controls training labels + postprocess gating). Keeping them decoupled
    lets us A/B gates on a fixed reference target.
    """
    probs_targets = []
    for tile in val_tiles:
        t0 = time.time()
        prob = predict_proba_max(model, tile, "train", cfg)
        forest = forest_mask_2020(tile, "train", source=eval_gate).astype(bool)
        target = target_for_tile(tile, split="train", forest_source=eval_gate)
        probs_targets.append((prob, target, forest))
        print(f"[VAL] {tile}: prob={prob.mean():.3f} target_frac={target.mean():.3%} ({time.time() - t0:.1f}s)")

    sweep = np.linspace(0.1, 0.9, 33)
    per_threshold: list[dict] = []
    for thr in sweep:
        m_total = PixelMetrics(0, 0, 0, 0)
        for prob, target, forest in probs_targets:
            pred = prob > thr
            m_total = m_total + pixel_metrics(pred, target, mask=forest)
        d = m_total.as_dict()
        d["threshold"] = float(thr)
        per_threshold.append(d)

    best_thr_iou, best_iou = best_threshold_aggregate(probs_targets, candidates=sweep, metric="iou")
    best_thr_f1, best_f1 = best_threshold_aggregate(probs_targets, candidates=sweep, metric="f1")

    return {
        "per_threshold": per_threshold,
        "best_threshold_iou": best_thr_iou,
        "best_iou": best_iou,
        "best_threshold_f1": best_thr_f1,
        "best_f1": best_f1,
        "val_tiles": val_tiles,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True, help="Fold index (0..n-1) or 'all'")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--pos-per-tile", type=int, default=20_000)
    parser.add_argument("--neg-per-tile", type=int, default=20_000)
    parser.add_argument("--years", type=str, default="2021,2022,2023,2024,2025")
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--forest-source",
        choices=("jrc", "ndvi"),
        default="jrc",
        help="2020-forest mask used for training labels + predict-time gating",
    )
    parser.add_argument(
        "--eval-gate",
        choices=("jrc", "ndvi"),
        default="jrc",
        help="2020-forest mask used to build the evaluation target (kept decoupled from "
        "--forest-source so A/B runs are scored on the same reference).",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional suffix appended to output filenames (e.g. 'jrc', 'ndvi') so "
        "runs don't overwrite each other's models/reports.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    years = tuple(int(y) for y in args.years.split(","))
    cfg = BaselineConfig(
        analysis_years=years,
        pos_per_tile=args.pos_per_tile,
        neg_per_tile=args.neg_per_tile,
        seed=args.seed,
        forest_source=args.forest_source,
    )
    tag = f"_{args.run_tag}" if args.run_tag else ""
    print(f"[CFG ] forest_source={args.forest_source} eval_gate={args.eval_gate} tag={tag or '(none)'}")

    all_tiles = filter_tiles_with_data(list_tiles("train"))
    if not all_tiles:
        print("[FAIL] no train tiles with S2 data yet — download still in progress.", file=sys.stderr)
        return 1
    print(f"[INFO] {len(all_tiles)} train tiles with S2 data available")

    if args.fold == "all":
        train_tiles, val_tiles, fold_label = all_tiles, [], "all"
    else:
        fold_idx = int(args.fold)
        for i, train, val in fold_iter(all_tiles, args.n_folds):
            if i == fold_idx:
                train_tiles = [t for t in train if t in all_tiles]
                val_tiles = [t for t in val if t in all_tiles]
                fold_label = str(fold_idx)
                break
        else:
            print(f"[FAIL] fold {fold_idx} out of range (n_folds={args.n_folds})", file=sys.stderr)
            return 1

    print(f"[INFO] fold={fold_label} | train={len(train_tiles)} val={len(val_tiles)}")
    print(f"[INFO] train_tiles: {train_tiles}")
    if val_tiles:
        print(f"[INFO] val_tiles:   {val_tiles}")

    t0 = time.time()
    print("[STEP] building training set ...")
    X, y, names = build_training_set(train_tiles, cfg, split="train")
    print(f"[STEP] X={X.shape} y={y.shape} pos_frac={y.mean():.3%} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("[STEP] training model ...")
    model = train_model(X, y, cfg)
    print(f"[STEP] trained in {time.time() - t0:.1f}s")

    model_path = out_dir / "models" / f"v1_fold{fold_label}{tag}.joblib"
    save_model(model, model_path)
    print(f"[STEP] saved {model_path}")

    summary: dict = {
        "fold": fold_label,
        "train_tiles": train_tiles,
        "val_tiles": val_tiles,
        "n_train_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "pos_frac_train": float(y.mean()),
        "channels": names,
        "cfg": {
            "analysis_years": list(cfg.analysis_years),
            "pos_per_tile": cfg.pos_per_tile,
            "neg_per_tile": cfg.neg_per_tile,
            "seed": cfg.seed,
            "forest_source": cfg.forest_source,
            "eval_gate": args.eval_gate,
        },
    }

    if val_tiles:
        print("[STEP] evaluating on val ...")
        summary["val_metrics"] = _evaluate_on_val(model, val_tiles, cfg, eval_gate=args.eval_gate)
        print(
            f"[VAL] best IoU={summary['val_metrics']['best_iou']:.4f} "
            f"@ thr={summary['val_metrics']['best_threshold_iou']:.2f}"
        )
        print(
            f"[VAL] best F1 ={summary['val_metrics']['best_f1']:.4f} "
            f"@ thr={summary['val_metrics']['best_threshold_f1']:.2f}"
        )

    metrics_path = out_dir / "reports" / f"v1_fold{fold_label}{tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
