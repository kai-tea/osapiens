"""Geographic cross-validation splits.

Tiles in the same MGRS 5-char prefix (e.g. ``18NWG``) are spatially adjacent —
keeping them together in the same fold prevents leakage and makes validation
a realistic stand-in for OOD generalisation across regions.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .data_loader import METADATA_DIR, list_tiles


def mgrs_prefix(tile_id: str) -> str:
    """Return the MGRS grid zone prefix of a tile ID (e.g. '18NWG' from '18NWG_6_6')."""
    return tile_id.split("_", 1)[0]


def _stable_bucket(key: str, n_folds: int) -> int:
    """Deterministic hash-based bucketing."""
    h = hashlib.md5(key.encode()).digest()
    return int.from_bytes(h[:8], "big") % n_folds


def split_by_mgrs(
    tiles: list[str],
    n_folds: int = 5,
) -> list[list[str]]:
    """Return a list of folds (each a list of tile IDs).

    All tiles sharing an MGRS prefix end up in the same fold, so validation
    folds correspond to geographically coherent regions.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for t in tiles:
        groups[mgrs_prefix(t)].append(t)

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    # Assign whole groups to the currently smallest fold (balanced by tile count),
    # tie-broken by deterministic hash so runs are reproducible.
    for prefix in sorted(groups, key=lambda p: (-len(groups[p]), p)):
        bucket_sizes = [(len(folds[i]), _stable_bucket(prefix + str(i), 1_000_000), i) for i in range(n_folds)]
        _, _, target = min(bucket_sizes)
        folds[target].extend(sorted(groups[prefix]))
    return folds


def fold_iter(tiles: list[str], n_folds: int = 5):
    """Yield (fold_idx, train_tiles, val_tiles) for each fold."""
    folds = split_by_mgrs(tiles, n_folds)
    for i, val in enumerate(folds):
        train = [t for j, f in enumerate(folds) if j != i for t in f]
        yield i, train, val


def load_tile_metadata(split: str = "train") -> list[dict]:
    """Load metadata/{split}_tiles.geojson as list of feature dicts.

    Returns [] if the file is not yet downloaded.
    """
    path = METADATA_DIR / f"{split}_tiles.geojson"
    if not path.exists():
        return []
    with open(path) as f:
        gj = json.load(f)
    return gj.get("features", [])


def write_splits(
    out_path: str | Path = "splits/mgrs_5fold.json",
    n_folds: int = 5,
    split: str = "train",
) -> Path:
    """Persist the CV split to disk so baseline + evaluator share the same folds."""
    tiles = list_tiles(split)
    folds = split_by_mgrs(tiles, n_folds)
    payload = {
        "n_folds": n_folds,
        "split": split,
        "n_tiles": len(tiles),
        "folds": folds,
        "mgrs_prefixes_per_fold": [sorted({mgrs_prefix(t) for t in f}) for f in folds],
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


if __name__ == "__main__":
    p = write_splits()
    print(f"wrote {p}")
