"""Convert a directory of per-tile prediction rasters into a submission.

Run:
  .venv/bin/python -m scripts.make_submission \
      --pred-dir predictions/v1 \
      --out-dir submission/v1

Expects files named ``pred_{tile_id}.tif`` (what ``baseline_v1.save_prediction_raster``
writes) and emits ``{tile_id}.geojson`` next to the input, plus a combined
``submission.geojson`` FeatureCollection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submission_utils import raster_to_geojson


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True, help="Directory of pred_{tile}.tif rasters")
    parser.add_argument("--out-dir", default="submission", help="Where to write per-tile + combined GeoJSON")
    parser.add_argument("--min-area-ha", type=float, default=0.5)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted(pred_dir.glob("*_binary.tif"))
    if not tifs:
        tifs = sorted(pred_dir.glob("pred_*.tif"))
    if not tifs:
        print(f"[FAIL] no *_binary.tif or pred_*.tif in {pred_dir}", file=sys.stderr)
        return 1

    combined_features: list[dict] = []
    summary: dict[str, dict] = {}
    for tif in tifs:
        tile_id = tif.stem.removeprefix("pred_").removesuffix("_binary")
        per_tile_out = out_dir / f"{tile_id}.geojson"
        try:
            gj = raster_to_geojson(tif, output_path=per_tile_out, min_area_ha=args.min_area_ha)
        except ValueError as e:
            # empty rasters / filtered out: note but continue
            summary[tile_id] = {"features": 0, "error": str(e)}
            print(f"[SKIP] {tile_id}: {e}")
            continue
        feats = gj.get("features", [])
        for f in feats:
            props = f.setdefault("properties", {})
            props["tile_id"] = tile_id
        combined_features.extend(feats)
        summary[tile_id] = {"features": len(feats)}
        print(f"[OK  ] {tile_id}: {len(feats)} polygons -> {per_tile_out}")

    combined = {"type": "FeatureCollection", "features": combined_features}
    combined_path = out_dir / "submission.geojson"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    with open(out_dir / "submission_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DONE] combined -> {combined_path} ({len(combined_features)} polygons across {len(tifs)} tiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
