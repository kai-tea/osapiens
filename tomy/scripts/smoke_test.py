"""End-to-end smoke on one training tile.

Run: `.venv/bin/python -m scripts.smoke_test [--tile TILE_ID] [--year YEAR]`

Verifies that the data loader + label fusion produce aligned, sane outputs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tomy.src import data_loader as dl
from tomy.src import label_fusion as lf


def human(n: float) -> str:
    for unit in ("", "K", "M", "G"):
        if n < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}T"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default=None, help="Tile ID (default: first available)")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to use (default: latest year with both S2 and S1 data for the tile)",
    )
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    tiles = dl.list_tiles(args.split)
    if not tiles:
        print(f"[FAIL] No tiles found under {dl.S2_DIR / args.split} — has the dataset finished downloading?")
        return 1
    tile = args.tile or tiles[0]
    print(f"[INFO] {len(tiles)} tiles available; using '{tile}'")

    if args.year is None:
        for cand in range(2025, 2019, -1):
            if dl.s2_paths(tile, cand, args.split) and dl.s1_paths(tile, cand, args.split):
                args.year = cand
                break
        if args.year is None:
            print(f"[FAIL] No year has both S2 and S1 data for {tile} — download still too partial.")
            return 1
        print(f"[INFO] auto-picked year={args.year} (latest with S2+S1 coverage)")

    t0 = time.time()
    grid = dl.tile_grid(tile, args.split)
    print(f"[GRID] crs={grid.crs} shape={grid.shape} transform={grid.transform!r}")

    # --- S2 ---
    s2_paths = dl.s2_paths(tile, args.year, args.split)
    print(f"[S2 ] {args.year}: {len(s2_paths)} months -> {sorted(s2_paths)}")
    s2_stats = dl.s2_year_stats(tile, args.year, args.split)
    print(f"[S2 ] year-stats shape={s2_stats.shape} nonzero={np.count_nonzero(s2_stats) / s2_stats.size:.2%}")

    # --- S1 ---
    s1_list = dl.s1_paths(tile, args.year, args.split)
    print(f"[S1 ] {args.year}: {len(s1_list)} scenes")
    s1_stats = dl.s1_year_stats(tile, args.year, args.split)
    print(f"[S1 ] year-stats shape={s1_stats.shape} median_db_range=[{s1_stats[0].min():.1f}, {s1_stats[0].max():.1f}]")

    # --- AEF ---
    aef = dl.load_aef_on_tile_grid(tile, args.year, args.split)
    print(f"[AEF] reprojected shape={aef.shape} mean_abs={np.abs(aef).mean():.3f}")

    # --- Combined features ---
    feats, names = dl.year_over_year_features(tile, args.year, args.split, baseline_year=2020)
    print(f"[FEAT] year_over_year_features -> {feats.shape} ({len(names)} channels)")

    # --- Forest mask: compare JRC vs NDVI ---
    # Bypass the cache by calling each source explicitly so we don't depend on
    # call order across a shared lru_cache key.
    from tomy.src import jrc_forest as jrc

    ndvi_mask = lf._ndvi_forest_mask(tile, args.split, ndvi_threshold=0.6)
    jrc_mask = jrc.load_jrc_on_tile_grid(tile, args.split)
    if jrc_mask is None:
        print(
            f"[MASK] JRC unavailable for {tile} — run `make download_jrc_forest` first. "
            f"NDVI-only forest fraction: {ndvi_mask.mean():.2%}"
        )
        forest = ndvi_mask
    else:
        inter = (jrc_mask & ndvi_mask).sum()
        union = (jrc_mask | ndvi_mask).sum()
        iou = inter / union if union else float("nan")
        print(
            f"[MASK] JRC forest={jrc_mask.mean():.2%}  NDVI forest={ndvi_mask.mean():.2%}  "
            f"IoU(JRC,NDVI)={iou:.3f}"
        )
        if iou < 0.5:
            print(
                f"[WARN] low JRC-vs-NDVI agreement (IoU={iou:.3f}) — probable reprojection "
                "or tile-naming bug. Visualise before retraining."
            )
        forest = jrc_mask  # downstream fusion uses the default (jrc) via lru_cache

    # Exercise the public forest_mask_2020 path too so any integration bug surfaces here.
    default_mask = lf.forest_mask_2020(tile, args.split)
    print(f"[MASK] default gate forest fraction: {default_mask.mean():.2%}")

    # --- Label fusion ---
    fused = lf.fuse_post2020(tile, args.split)
    total_px = fused["any_alert"].size
    print(
        f"[LBL ] any_alert={fused['any_alert'].sum() / total_px:.3%} "
        f"confident_pos={fused['confident_pos'].sum() / total_px:.3%} "
        f"confident_neg={fused['confident_neg'].sum() / total_px:.3%} "
        f"mean_soft={fused['soft'][fused['any_alert'] > 0].mean() if fused['any_alert'].sum() else 0.0:.3f}"
    )
    events = fused["event_days"][fused["any_alert"] > 0]
    if events.size:
        print(f"[LBL ] event_days range: [{events.min()}, {events.max()}]  (days since 1970-01-01)")

    # --- Alignment sanity ---
    assert feats.shape[1:] == grid.shape, f"feature HW mismatch {feats.shape[1:]} vs {grid.shape}"
    assert forest.shape == grid.shape, f"forest mask HW mismatch {forest.shape} vs {grid.shape}"
    assert fused["any_alert"].shape == grid.shape, "label HW mismatch"
    print(f"[OK  ] all modalities aligned on ({grid.shape}) | elapsed {time.time() - t0:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
