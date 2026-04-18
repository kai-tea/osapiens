"""Download JRC Global Forest Cover 2020 V3 tiles covering our project AOIs.

Reads tile footprints from ``data/makeathon-challenge/metadata/{train,test}_tiles.geojson``,
enumerates the 10°×10° JRC tiles needed to cover them, and downloads each to
``data/external/jrc_gfc2020/``.

Idempotent: tiles already on disk with a matching size are skipped. Tiles that
come back 404 (ocean-only regions that JRC does not publish) are silently
skipped — the loader handles missing coverage with the NDVI fallback.

Naming convention (confirmed against the live FTP on 2026-04-18):
    JRC_GFC2020_V3_<NS><lat>_<EW><lon>.tif
        lat : 2-digit minimum, natural width, upper-left corner of the tile
        lon : 2-digit minimum, natural width, upper-left corner of the tile

Usage:
    .venv/bin/python -m tomy.scripts.download_jrc_forest
"""

from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/GFC2020/LATEST/tiles"
METADATA_DIR = Path("data/makeathon-challenge/metadata")
OUT_DIR = Path("data/external/jrc_gfc2020")
METADATA_FILES = ("train_tiles.geojson", "test_tiles.geojson")

# Polygon boundaries are in EPSG:4326 — confirmed by inspecting values.


def _iter_coord_rings(geom: dict):
    t = geom["type"]
    if t == "Polygon":
        yield from geom["coordinates"]
    elif t == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield from poly
    else:
        raise ValueError(f"Unsupported geometry type: {t}")


def _bbox_of_geom(geom: dict) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) of the geometry in EPSG:4326."""
    xs: list[float] = []
    ys: list[float] = []
    for ring in _iter_coord_rings(geom):
        for lon, lat, *_ in ring:
            xs.append(lon)
            ys.append(lat)
    return min(xs), min(ys), max(xs), max(ys)


def _jrc_tiles_covering(bbox: tuple[float, float, float, float]) -> set[tuple[int, int]]:
    """Enumerate (lat_top, lon_left) pairs covering the bbox.

    JRC names tiles by their upper-left (north-west) corner: tile ``(lat, lon)``
    covers ``[lat - 10, lat]`` in latitude and ``[lon, lon + 10]`` in longitude.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    # Upper-left lat of the tile containing a point at lat_p: ceil(lat_p/10)*10
    lat_top_min = int(math.ceil(lat_min / 10.0)) * 10
    lat_top_max = int(math.ceil(lat_max / 10.0)) * 10
    lon_left_min = int(math.floor(lon_min / 10.0)) * 10
    lon_left_max = int(math.floor(lon_max / 10.0)) * 10
    out: set[tuple[int, int]] = set()
    for lat_top in range(lat_top_min, lat_top_max + 1, 10):
        for lon_left in range(lon_left_min, lon_left_max + 1, 10):
            out.add((lat_top, lon_left))
    return out


def jrc_tile_name(lat_top: int, lon_left: int) -> str:
    """JRC V3 filename for a tile identified by its upper-left corner."""
    ns = "N" if lat_top >= 0 else "S"
    ew = "E" if lon_left >= 0 else "W"
    return f"JRC_GFC2020_V3_{ns}{abs(lat_top):02d}_{ew}{abs(lon_left):02d}.tif"


def jrc_tile_url(lat_top: int, lon_left: int) -> str:
    return f"{BASE_URL}/{jrc_tile_name(lat_top, lon_left)}"


def _gather_required_tiles() -> tuple[set[tuple[int, int]], list[tuple[str, set[tuple[int, int]]]]]:
    """Read the metadata geojsons and compute the union of needed JRC tiles.

    Returns (all_tiles, per_project_tile) where per_project_tile is a list of
    ``(project_tile_id, required_jrc_tiles)`` so we can print coverage details.
    """
    all_tiles: set[tuple[int, int]] = set()
    per_tile: list[tuple[str, set[tuple[int, int]]]] = []
    for fname in METADATA_FILES:
        path = METADATA_DIR / fname
        if not path.exists():
            print(f"[WARN] {path} missing; skipping", file=sys.stderr)
            continue
        with open(path) as f:
            data = json.load(f)
        for feat in data.get("features", []):
            tile_id = feat["properties"].get("name", "?")
            bbox = _bbox_of_geom(feat["geometry"])
            needed = _jrc_tiles_covering(bbox)
            per_tile.append((tile_id, needed))
            all_tiles |= needed
    return all_tiles, per_tile


def _download_one(url: str, dest: Path, expected_size: int | None = None) -> str:
    """Download ``url`` to ``dest``. Returns one of 'ok', 'skipped', 'missing', 'fail'."""
    if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
        return "skipped"
    if dest.exists() and expected_size is None and dest.stat().st_size > 0:
        return "skipped"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=120) as r:
            total = int(r.headers.get("content-length") or 0)
            with open(tmp, "wb") as f:
                downloaded = 0
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
        tmp.rename(dest)
        return "ok"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "missing"
        raise
    finally:
        if tmp.exists():
            tmp.unlink()


def _head_size(url: str) -> int | None:
    """Return Content-Length via HEAD, or None (404 / no length)."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("content-length")
            return int(cl) if cl else None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _verify_first_tile(path: Path, lat_top: int, lon_left: int) -> None:
    """Open one tile and confirm its georeferencing matches the assumed top-left convention.

    Failing here means the tile naming convention in ``_jrc_tiles_covering`` is
    wrong and *every* tile we've downloaded needs re-interpretation — catching
    it once is worth the cost. Skipped with a warning if rasterio isn't
    installed in the current interpreter (the downloader itself is stdlib-only).
    """
    try:
        import rasterio  # type: ignore
    except ImportError:
        print(
            f"[WARN] rasterio not available in this interpreter — skipping geotransform "
            f"verification. Will be re-checked on first use via smoke_test.py."
        )
        return

    with rasterio.open(path) as src:
        ul_x, ul_y = src.transform * (0, 0)
        lr_x, lr_y = src.transform * (src.width, src.height)
        expected_ul = (float(lon_left), float(lat_top))
        expected_lr = (float(lon_left + 10), float(lat_top - 10))
        ok_ul = abs(ul_x - expected_ul[0]) < 0.01 and abs(ul_y - expected_ul[1]) < 0.01
        ok_lr = abs(lr_x - expected_lr[0]) < 0.01 and abs(lr_y - expected_lr[1]) < 0.01
        if not (ok_ul and ok_lr):
            raise RuntimeError(
                f"JRC tile geotransform mismatch for {path.name}: "
                f"upper-left=({ul_x:.4f},{ul_y:.4f}) expected={expected_ul}; "
                f"lower-right=({lr_x:.4f},{lr_y:.4f}) expected={expected_lr}. "
                "Tile-naming convention assumption is wrong."
            )
        print(
            f"[OK] {path.name}: geotransform matches top-left convention "
            f"({ul_x:.1f},{ul_y:.1f}) → ({lr_x:.1f},{lr_y:.1f})"
        )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_tiles, per_tile = _gather_required_tiles()
    print(f"[INFO] {len(per_tile)} project tiles → {len(all_tiles)} unique JRC tiles needed")
    for proj_id, needed in per_tile:
        names = sorted(jrc_tile_name(lat, lon) for lat, lon in needed)
        print(f"  {proj_id}: {len(needed)} tile(s) — {names}")

    verified = False
    counts = {"ok": 0, "skipped": 0, "missing": 0}
    for lat_top, lon_left in sorted(all_tiles):
        name = jrc_tile_name(lat_top, lon_left)
        url = jrc_tile_url(lat_top, lon_left)
        dest = OUT_DIR / name
        size = _head_size(url)
        if size is None:
            print(f"[MISSING] {name} not published by JRC (likely ocean) — loader will fall back")
            counts["missing"] += 1
            continue
        status = _download_one(url, dest, expected_size=size)
        print(f"[{status.upper():7s}] {name} ({size/1e6:.1f} MB) → {dest}")
        counts[status] = counts.get(status, 0) + 1
        if status == "ok" and not verified:
            _verify_first_tile(dest, lat_top, lon_left)
            verified = True
    print(f"[DONE] downloaded={counts['ok']} skipped={counts['skipped']} missing={counts['missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
