"""Label EDA — per-source positive rates and cross-source agreement.

Runs without Sentinel-2 data: reads labels in their native EPSG:4326 grids,
reprojects GLAD-L (coarser) onto the RADD / GLAD-S2 grid for agreement analysis.

Key questions this answers
--------------------------
1. What fraction of pixels does each source flag post-2020?
2. How often do sources agree? (low agreement -> `min_sources=2` starves the
   classifier; high agreement -> trustworthy fused target)
3. Alert-year distribution across sources.
4. Per-tile coverage: which tiles have all three sources, which are thin.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tomy.src.label_fusion import (
    POST2020,
    _days_since_epoch,
    _merge_decoded,
    decode_gladl,
    decode_glads2,
    decode_radd,
)

LABELS_ROOT = Path("data/makeathon-challenge/labels/train")
METADATA_PATH = Path("data/makeathon-challenge/metadata/train_tiles.geojson")

RADD_ROOT = LABELS_ROOT / "radd"
GLADS2_ROOT = LABELS_ROOT / "glads2"
GLADL_ROOT = LABELS_ROOT / "gladl"


def load_tile_ids_from_metadata() -> list[str]:
    if not METADATA_PATH.exists():
        return []
    with open(METADATA_PATH) as f:
        gj = json.load(f)
    ids = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        for k in ("tile_id", "id", "name"):
            if k in props:
                ids.append(str(props[k]))
                break
    return sorted(set(ids))


def list_radd_tiles() -> list[str]:
    return sorted(p.stem.removeprefix("radd_").removesuffix("_labels") for p in RADD_ROOT.glob("radd_*_labels.tif"))


def _read(path: Path, band: int = 1) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS, tuple[int, int]]:
    with rasterio.open(path) as src:
        return src.read(band), src.transform, src.crs, src.shape


def _reproject_onto(
    src_data: np.ndarray,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
) -> np.ndarray:
    dst = np.zeros(dst_shape, dtype=src_data.dtype)
    reproject(
        source=src_data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return dst


def load_gladl_merged_on_grid(
    tile_id: str,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
):
    """Decode every GLAD-L year and merge into a single DecodedLabels on the target grid."""
    per_year = []
    for alert_path in sorted(GLADL_ROOT.glob(f"gladl_{tile_id}_alert[0-9][0-9].tif")):
        yy = int(alert_path.stem[-2:])
        date_path = GLADL_ROOT / f"gladl_{tile_id}_alertDate{yy:02d}.tif"
        if not date_path.exists():
            continue
        with rasterio.open(alert_path) as src:
            alert_nat = src.read(1)
            alert = _reproject_onto(alert_nat, src.transform, src.crs, dst_transform, dst_crs, dst_shape)
        with rasterio.open(date_path) as src:
            date_nat = src.read(1)
            adate = _reproject_onto(date_nat, src.transform, src.crs, dst_transform, dst_crs, dst_shape)
        per_year.append(decode_gladl(alert.astype(np.int32), adate.astype(np.int32), yy))
    return _merge_decoded(per_year, "gladl")


def post2020_mask(decoded) -> np.ndarray:
    if decoded is None:
        return np.zeros((1, 1), dtype=bool)
    cutoff = _days_since_epoch(POST2020)
    return decoded.alert.astype(bool) & (decoded.date_days >= cutoff)


def decode_year_from_days(days_since_epoch: np.ndarray) -> np.ndarray:
    """int32 days-since-1970 -> int16 year (0 stays 0)."""
    out = np.zeros_like(days_since_epoch, dtype=np.int16)
    valid = days_since_epoch > 0
    dt = np.array(days_since_epoch[valid], dtype="datetime64[D]")
    out[valid] = dt.astype("datetime64[Y]").astype(int) + 1970
    return out


def analyse_tile(tile_id: str) -> dict:
    radd_path = RADD_ROOT / f"radd_{tile_id}_labels.tif"
    glads2_alert_path = GLADS2_ROOT / f"glads2_{tile_id}_alert.tif"
    glads2_date_path = GLADS2_ROOT / f"glads2_{tile_id}_alertDate.tif"

    if not radd_path.exists():
        return {"tile": tile_id, "error": "no RADD"}

    radd_raw, rt, rc, rshape = _read(radd_path)
    radd = decode_radd(radd_raw)
    n_pix = radd.alert.size

    # GLAD-S2 natively shares RADD's grid for this dataset; read + decode directly.
    glads2 = None
    if glads2_alert_path.exists() and glads2_date_path.exists():
        ga_raw, ga_t, ga_c, ga_s = _read(glads2_alert_path)
        gd_raw, _, _, _ = _read(glads2_date_path)
        if ga_s != rshape or ga_t != rt or ga_c != rc:
            ga_raw = _reproject_onto(ga_raw, ga_t, ga_c, rt, rc, rshape)
            gd_raw = _reproject_onto(gd_raw, ga_t, ga_c, rt, rc, rshape)
        glads2 = decode_glads2(ga_raw.astype(np.int32), gd_raw.astype(np.int32))

    gladl = load_gladl_merged_on_grid(tile_id, rt, rc, rshape)

    radd_p = post2020_mask(radd)
    glads2_p = post2020_mask(glads2) if glads2 is not None else np.zeros_like(radd_p)
    gladl_p = post2020_mask(gladl) if gladl is not None else np.zeros_like(radd_p)

    source_count = radd_p.astype(np.uint8) + glads2_p.astype(np.uint8) + gladl_p.astype(np.uint8)
    any_alert = source_count > 0
    pair_rs = radd_p & glads2_p
    pair_rl = radd_p & gladl_p
    pair_sl = glads2_p & gladl_p
    triple = radd_p & glads2_p & gladl_p

    # Year histogram: use earliest post-2020 date among sources that fired.
    earliest_days = np.full_like(radd.date_days, np.iinfo(np.int32).max, dtype=np.int32)
    for dl_mask, dl in ((radd_p, radd), (glads2_p, glads2), (gladl_p, gladl)):
        if dl is None:
            continue
        earliest_days = np.where(dl_mask, np.minimum(earliest_days, dl.date_days), earliest_days)
    earliest_days = np.where(any_alert, earliest_days, 0)
    years = decode_year_from_days(earliest_days)
    year_counts = dict(Counter(years[any_alert].tolist()))

    return {
        "tile": tile_id,
        "shape": list(rshape),
        "n_pixels": int(n_pix),
        "radd_post2020_frac": float(radd_p.mean()),
        "glads2_post2020_frac": float(glads2_p.mean()) if glads2 is not None else None,
        "gladl_post2020_frac": float(gladl_p.mean()) if gladl is not None else None,
        "any_alert_frac": float(any_alert.mean()),
        "confident_pos_frac_min2": float((source_count >= 2).mean()),
        "confident_pos_frac_min3": float((source_count >= 3).mean()),
        "pair_radd_glads2_frac": float(pair_rs.mean()),
        "pair_radd_gladl_frac": float(pair_rl.mean()),
        "pair_glads2_gladl_frac": float(pair_sl.mean()),
        "triple_frac": float(triple.mean()),
        "year_counts": {int(k): int(v) for k, v in sorted(year_counts.items())},
        "sources_present": {
            "radd": True,
            "glads2": glads2 is not None,
            "gladl": gladl is not None,
        },
    }


def aggregate(per_tile: list[dict]) -> dict:
    valid = [t for t in per_tile if "error" not in t]
    def avg(key):
        vals = [t[key] for t in valid if t.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    all_years = Counter()
    for t in valid:
        for y, n in t["year_counts"].items():
            all_years[y] += n

    return {
        "n_tiles": len(per_tile),
        "n_valid": len(valid),
        "mean_radd_post2020_frac": avg("radd_post2020_frac"),
        "mean_glads2_post2020_frac": avg("glads2_post2020_frac"),
        "mean_gladl_post2020_frac": avg("gladl_post2020_frac"),
        "mean_any_alert_frac": avg("any_alert_frac"),
        "mean_confident_pos_frac_min2": avg("confident_pos_frac_min2"),
        "mean_confident_pos_frac_min3": avg("confident_pos_frac_min3"),
        "mean_pair_radd_glads2_frac": avg("pair_radd_glads2_frac"),
        "mean_pair_radd_gladl_frac": avg("pair_radd_gladl_frac"),
        "mean_pair_glads2_gladl_frac": avg("pair_glads2_gladl_frac"),
        "mean_triple_frac": avg("triple_frac"),
        "year_totals": {int(k): int(v) for k, v in sorted(all_years.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/label_eda.json")
    args = parser.parse_args()

    tiles_meta = load_tile_ids_from_metadata()
    tiles_on_disk = list_radd_tiles()
    tiles = tiles_on_disk or tiles_meta
    if not tiles:
        print("[FAIL] no tiles found on disk or in metadata", file=sys.stderr)
        return 1

    print(f"[INFO] {len(tiles)} tiles with RADD labels on disk")
    print(f"[INFO] {len(tiles_meta)} tiles listed in metadata")

    per_tile = []
    for i, t in enumerate(tiles, 1):
        try:
            res = analyse_tile(t)
        except Exception as exc:  # pragma: no cover
            res = {"tile": t, "error": str(exc)}
        per_tile.append(res)
        if "error" in res:
            print(f"[{i:3d}/{len(tiles)}] {t}: ERROR {res['error']}")
        else:
            def _fmt(x):
                return f"{x:.3%}" if x is not None else "  n/a"
            print(
                f"[{i:3d}/{len(tiles)}] {t} "
                f"radd={_fmt(res['radd_post2020_frac'])} "
                f"glads2={_fmt(res['glads2_post2020_frac'])} "
                f"gladl={_fmt(res['gladl_post2020_frac'])} "
                f"any={_fmt(res['any_alert_frac'])} "
                f"pos2={_fmt(res['confident_pos_frac_min2'])} "
                f"pos3={_fmt(res['confident_pos_frac_min3'])}"
            )

    agg = aggregate(per_tile)
    print("\n=== AGGREGATE ===")
    for k, v in agg.items():
        if k in ("year_totals",):
            continue
        if isinstance(v, float):
            print(f"  {k:<40} {v:.4%}")
        else:
            print(f"  {k:<40} {v}")
    print("  year_totals (year -> pixels flagged by ≥1 source post-2020):")
    for y, n in agg["year_totals"].items():
        print(f"    {y}: {n:,}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"aggregate": agg, "per_tile": per_tile}, f, indent=2)
    print(f"\n[DONE] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
