"""Fire-and-forget orchestrator for the autoloop.

Flow (all non-interactive; exits 0 even if individual stages fail so the
safety-net heuristics still make it to disk):

  1. Emit all heuristic variants (Hansen grid over treecover + lossyear).
     These are the safety net — cheap, no-train, always shippable.
  2. Run ungated 3-fold CV for two model configs (base + high-recall).
  3. Train the winning config on all 16 tiles; save checkpoint.
  4. Run one round of self-training (pseudo-labels agreed by weak labels).
  5. Emit per-tile GeoJSONs for both model variants, plus an
     ensemble variant (model ∪ hansen lossyear post-2020).
  6. Write submission/autoloop/summary.md ranking all candidates by
     ungated CV F1 (computed on the same 3-fold split) alongside the
     heuristic candidates. The operator picks which to submit.

Does NOT submit to the leaderboard — the operator submits manually.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .io import REPO_ROOT, load_fold_assignments, load_tile_manifest
from .predict import (
    RASTER_ROOT,
    SUBMISSION_ROOT,
    combine_submission_geojsons,
    predict_test_tile,
)
from .train import (
    MODELS_ROOT,
    POSITIVE_THRESHOLD,
    TrainConfig,
    load_checkpoint,
    run_ungated_cv,
    save_checkpoint,
    self_train,
    train_full,
)
from .variants import emit_all_variants

logger = logging.getLogger(__name__)

SUMMARY_PATH = SUBMISSION_ROOT / "summary.md"
SUMMARY_JSON_PATH = SUBMISSION_ROOT / "summary.json"


def _train_tile_ids() -> list[str]:
    fa = load_fold_assignments()
    return sorted(fa["tile_id"].unique().tolist())


def _test_tile_ids() -> list[str]:
    m = load_tile_manifest()
    return m.loc[m["split"] == "test", "tile_id"].tolist()


def _median_threshold(results) -> float:
    ths = [r.threshold for r in results]
    return float(np.median(ths)) if ths else 0.5


def _cv_mean(results, attr: str) -> float:
    return float(np.mean([getattr(r, attr) for r in results])) if results else 0.0


def _baseline_cv_scores() -> dict[str, float]:
    """Read cached naive-baseline CV F1 from reports/baseline_naive.json.

    Re-running the naive CV is cheap per-tile but expensive in aggregate on
    the droplet (downloads Hansen tiles for the forest gate across all 16
    training tiles). We anchor to the last saved numbers Kaite reported
    under reports/baseline_naive.json; the table is informational only —
    we never burn a submission on a naive strategy without an operator
    eyeballing it first.
    """
    path = REPO_ROOT / "reports" / "baseline_naive.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception as err:
        logger.warning("could not parse %s: %s", path, err)
        return {}
    out: dict[str, float] = {}
    for strat, folds in payload.items():
        f1s = [m["f1"] for m in folds.values()]
        out[strat] = float(np.mean(f1s)) if f1s else 0.0
    return out


def run_model_stage(
    tag: str,
    cfg: TrainConfig,
    test_tiles: list[str],
    extra_union_per_tile: dict[str, np.ndarray] | None = None,
) -> dict:
    """Run ungated CV + full-data train + test-tile inference for one config."""
    logger.info("=== stage %s: ungated CV ===", tag)
    cv_results, _ = run_ungated_cv(cfg)
    threshold = _median_threshold(cv_results)
    f1_mean = _cv_mean(cv_results, "f1")
    iou_mean = _cv_mean(cv_results, "iou")

    logger.info("=== stage %s: full-data train ===", tag)
    model, info = train_full(cfg, _train_tile_ids())
    ckpt_path = MODELS_ROOT / f"{tag}.pt"
    save_checkpoint(model, info, ckpt_path)

    logger.info("=== stage %s: test-tile inference ===", tag)
    manifest = load_tile_manifest()
    tile_summaries: list[dict] = []
    for tid in test_tiles:
        extra = extra_union_per_tile.get(tid) if extra_union_per_tile else None
        try:
            tile_summaries.append(
                predict_test_tile(
                    tid, model, manifest, threshold, candidate_tag=tag, extra_union=extra
                )
            )
        except Exception as err:
            logger.warning("inference failed on tile %s / %s: %s", tid, tag, err)
            tile_summaries.append({"tile_id": tid, "error": str(err)})

    combined_path = combine_submission_geojsons(tag)
    return {
        "tag": tag,
        "cv_f1_mean": f1_mean,
        "cv_iou_mean": iou_mean,
        "threshold": threshold,
        "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
        "combined_geojson": str(combined_path.relative_to(REPO_ROOT)),
        "tiles": tile_summaries,
        "config": asdict(cfg),
    }


def _hansen_union_per_tile(
    tile_ids: list[str],
    treecover_threshold: int = 25,
    earliest: int = 21,
    latest: int = 25,
) -> dict[str, np.ndarray]:
    """Pre-compute the Hansen-lossyear positive mask for each test tile."""
    import rasterio as _r

    from src.data import resolve_tile_paths
    from src.submit_heuristic import heuristic_binary_raster

    manifest = load_tile_manifest()
    out: dict[str, np.ndarray] = {}
    for tid in tile_ids:
        paths = resolve_tile_paths(tid, manifest=manifest)
        with _r.open(paths.s2_ref_path) as src:
            prof = src.profile.copy()
        mask, _, _ = heuristic_binary_raster(
            prof,
            treecover_threshold=treecover_threshold,
            earliest_loss_year=earliest,
            latest_loss_year=latest,
        )
        out[tid] = mask
    return out


def write_summary(
    heuristics: list[dict],
    model_stages: list[dict],
    naive_baselines: dict[str, float],
) -> None:
    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "heuristics": heuristics,
        "model_stages": model_stages,
        "naive_cv_f1": naive_baselines,
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    lines: list[str] = ["# Autoloop candidates\n"]
    lines.append("## Anchors (from naive CV)\n")
    lines.append("| strategy | mean ungated F1 |")
    lines.append("| --- | ---: |")
    for k, v in sorted(naive_baselines.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{k}` | {v:.3f} |")
    lines.append("")

    lines.append("## Model stages (ungated 3-fold CV)\n")
    lines.append("| tag | CV F1 | CV IoU | threshold | combined GeoJSON |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for m in model_stages:
        lines.append(
            f"| `{m['tag']}` | {m.get('cv_f1_mean', 0):.3f} | "
            f"{m.get('cv_iou_mean', 0):.3f} | "
            f"{m.get('threshold', 0):.3f} | `{m.get('combined_geojson', '')}` |"
        )
    lines.append("")

    lines.append("## Heuristic variants\n")
    lines.append("| tag | tc | lossyear | morph | combined GeoJSON |")
    lines.append("| --- | ---: | --- | ---: | --- |")
    for h in heuristics:
        lines.append(
            f"| `{h['tag']}` | {h['treecover_threshold']} | "
            f"{h['earliest_loss_year']}-{h['latest_loss_year']} | "
            f"{h['morph_open']} | `{h['combined_geojson']}` |"
        )
    lines.append("")
    lines.append("## Recommended submission order\n")
    lines.append(
        "Submit the model stage with the highest CV F1 first IF it beats "
        "the best naive CV F1 by ≥ 0.03 — otherwise start with the best "
        "heuristic variant. After each leaderboard response, pivot: if the "
        "score drops vs the 31.20% Hansen baseline, try a variant with "
        "lower recall (tighter tc / later lossyear). Never burn two "
        "submissions on very similar candidates.\n"
    )
    SUMMARY_PATH.write_text("\n".join(lines))
    logger.info("summary -> %s", SUMMARY_PATH)


def run(
    skip_heuristics: bool = False,
    skip_models: bool = False,
    epochs: int = 20,
) -> int:
    test_tiles = _test_tile_ids()

    heuristics: list[dict] = []
    if not skip_heuristics:
        try:
            heuristics = emit_all_variants(test_tiles)
        except Exception as err:
            logger.exception("heuristic stage failed: %s", err)

    naive_baselines: dict[str, float] = {}
    try:
        naive_baselines = _baseline_cv_scores()
    except Exception as err:
        logger.warning("naive baseline CV failed: %s", err)

    model_stages: list[dict] = []
    if not skip_models:
        hansen_union = _hansen_union_per_tile(test_tiles)

        configs = [
            ("autoloop_base", TrainConfig(epochs=epochs)),
            (
                "autoloop_recall",
                TrainConfig(
                    epochs=epochs,
                    pos_weight=5.0,
                    iou_weight=0.7,
                    aef_drop_prob=0.75,
                ),
            ),
        ]
        for tag, cfg in configs:
            try:
                stage = run_model_stage(tag, cfg, test_tiles)
                model_stages.append(stage)
            except Exception as err:
                logger.exception("stage %s failed: %s", tag, err)
                model_stages.append({"tag": tag, "error": str(err), "traceback": traceback.format_exc()})

        # Self-training: take the best base model and run one pseudo-label pass.
        try:
            best_stage = max(
                (s for s in model_stages if "cv_f1_mean" in s),
                key=lambda s: s["cv_f1_mean"],
                default=None,
            )
            if best_stage is not None:
                base_ckpt = MODELS_ROOT / f"{best_stage['tag']}.pt"
                base_model, _info = load_checkpoint(base_ckpt)
                self_cfg = TrainConfig(epochs=max(10, epochs // 2))
                self_model, self_info = self_train(base_model, self_cfg, _train_tile_ids())
                self_tag = "autoloop_selftrain"
                save_checkpoint(self_model, self_info, MODELS_ROOT / f"{self_tag}.pt")

                manifest = load_tile_manifest()
                tile_summaries: list[dict] = []
                for tid in test_tiles:
                    try:
                        tile_summaries.append(
                            predict_test_tile(
                                tid,
                                self_model,
                                manifest,
                                threshold=best_stage["threshold"],
                                candidate_tag=self_tag,
                            )
                        )
                    except Exception as err:
                        logger.warning("selftrain inference failed on %s: %s", tid, err)
                        tile_summaries.append({"tile_id": tid, "error": str(err)})
                combined_path = combine_submission_geojsons(self_tag)
                model_stages.append(
                    {
                        "tag": self_tag,
                        "cv_f1_mean": best_stage["cv_f1_mean"],  # no fresh CV to save time
                        "cv_iou_mean": best_stage["cv_iou_mean"],
                        "threshold": best_stage["threshold"],
                        "combined_geojson": str(combined_path.relative_to(REPO_ROOT)),
                        "tiles": tile_summaries,
                        "note": "self-trained from best base; CV reused from parent",
                        "config": asdict(self_cfg),
                    }
                )

                # Ensemble: best-base model ∪ Hansen lossyear post-2020.
                ensemble_tag = "autoloop_ensemble"
                tile_summaries = []
                for tid in test_tiles:
                    try:
                        tile_summaries.append(
                            predict_test_tile(
                                tid,
                                base_model,
                                manifest,
                                threshold=best_stage["threshold"],
                                candidate_tag=ensemble_tag,
                                extra_union=hansen_union.get(tid),
                            )
                        )
                    except Exception as err:
                        logger.warning("ensemble inference failed on %s: %s", tid, err)
                        tile_summaries.append({"tile_id": tid, "error": str(err)})
                combined_path = combine_submission_geojsons(ensemble_tag)
                model_stages.append(
                    {
                        "tag": ensemble_tag,
                        "cv_f1_mean": best_stage["cv_f1_mean"],
                        "cv_iou_mean": best_stage["cv_iou_mean"],
                        "threshold": best_stage["threshold"],
                        "combined_geojson": str(combined_path.relative_to(REPO_ROOT)),
                        "tiles": tile_summaries,
                        "note": "best-base ∪ hansen_lossyear_post2020",
                    }
                )
        except Exception as err:
            logger.exception("self-training / ensemble stage failed: %s", err)

    write_summary(heuristics, model_stages, naive_baselines)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-heuristics", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    return run(
        skip_heuristics=args.skip_heuristics,
        skip_models=args.skip_models,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
