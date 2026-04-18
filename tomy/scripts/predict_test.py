"""Run a trained V1 model on the test tiles -> per-tile binary prediction rasters.

Example:
  .venv/bin/python -m scripts.predict_test \
      --model models/v1_foldall.joblib \
      --threshold 0.45 \
      --out-dir predictions/v1_test

After this, run:
  .venv/bin/python -m scripts.make_submission --pred-dir predictions/v1_test --out-dir submission/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tomy.src.baseline_v1 import (
    BaselineConfig,
    load_model,
    predict_proba_max,
    save_prediction_raster,
)
from tomy.src.data_loader import list_tiles
from tomy.src.label_fusion import forest_mask_2020
from tomy.src.postprocess import PostprocessConfig, postprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .joblib model")
    parser.add_argument("--out-dir", default="predictions/v1_test")
    parser.add_argument("--threshold", type=float, default=None, help="Binarisation threshold (default: PostprocessConfig default)")
    parser.add_argument("--threshold-file", default=None, help="JSON file with {'threshold': float} (e.g. reports/v1_fold*_metrics.json)")
    parser.add_argument("--opening-kernel", type=int, default=3)
    parser.add_argument("--closing-kernel", type=int, default=3)
    parser.add_argument("--min-component-px", type=int, default=5)
    parser.add_argument("--years", default="2021,2022,2023,2024,2025")
    parser.add_argument("--split", default="test")
    parser.add_argument("--tiles", default=None, help="comma-separated subset of tile IDs; default: all")
    args = parser.parse_args()

    threshold = args.threshold
    if threshold is None and args.threshold_file:
        with open(args.threshold_file) as f:
            data = json.load(f)
        threshold = data.get("best_threshold_iou") or data.get("threshold") or (
            data.get("val_metrics", {}).get("best_threshold_iou")
        )
        if threshold is None:
            print(f"[FAIL] couldn't find threshold key in {args.threshold_file}", file=sys.stderr)
            return 1
        print(f"[INFO] loaded threshold={threshold:.3f} from {args.threshold_file}")
    if threshold is None:
        threshold = 0.5

    pp_cfg = PostprocessConfig(
        threshold=float(threshold),
        opening_kernel=args.opening_kernel,
        closing_kernel=args.closing_kernel,
        min_component_px=args.min_component_px,
    )
    cfg = BaselineConfig(
        analysis_years=tuple(int(y) for y in args.years.split(",")),
        postprocess=pp_cfg,
    )
    print(f"[INFO] {pp_cfg}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tiles = list_tiles(args.split)
    if not all_tiles:
        print(f"[FAIL] no tiles under {args.split}/ yet", file=sys.stderr)
        return 1
    tiles = [t for t in all_tiles if (args.tiles is None or t in args.tiles.split(","))]
    print(f"[INFO] predicting on {len(tiles)} tiles from split='{args.split}'")

    model = load_model(Path(args.model))
    meta: dict[str, dict] = {}
    for tile in tiles:
        t0 = time.time()
        prob = predict_proba_max(model, tile, args.split, cfg)
        if not prob.any():
            print(f"[SKIP] {tile}: no probability mass")
            meta[tile] = {"positive_fraction": 0.0}
            # still write an all-zero raster so downstream tooling has a consistent set
            save_prediction_raster(prob.astype("uint8"), tile, args.split, out_dir)
            continue
        forest = forest_mask_2020(tile, args.split)
        binary = postprocess(prob, forest_mask=forest, config=pp_cfg)
        path = save_prediction_raster(binary, tile, args.split, out_dir)
        frac = float(binary.mean())
        meta[tile] = {"raster": str(path), "positive_fraction": frac}
        print(f"[PRED] {tile}: pos_frac={frac:.3%} ({time.time() - t0:.1f}s) -> {path}")

    with open(out_dir / "meta.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "threshold": float(threshold),
                "postprocess": vars(pp_cfg),
                "split": args.split,
                "tiles": meta,
            },
            f,
            indent=2,
        )
    print(f"[DONE] wrote {len(tiles)} rasters -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
