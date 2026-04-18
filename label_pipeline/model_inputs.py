"""Model-input inventory, download, and handoff manifest helpers."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import REGION_FIELD_CANDIDATES
from .splits import load_split_assignments


DEFAULT_BUCKET_NAME = "osapiens-terra-challenge"
REMOTE_PREFIXES = (
    "makeathon-challenge/sentinel-1/train/",
    "makeathon-challenge/sentinel-1/test/",
    "makeathon-challenge/sentinel-2/train/",
    "makeathon-challenge/sentinel-2/test/",
    "makeathon-challenge/aef-embeddings/train/",
    "makeathon-challenge/aef-embeddings/test/",
)
ORBIT_ORDER = {"ascending": 0, "descending": 1}
MODALITY_DIRS = {"s1": "sentinel-1", "s2": "sentinel-2", "aef": "aef-embeddings"}


@dataclass(frozen=True)
class ParsedRemoteFile:
    remote_key: str
    split: str
    tile_id: str
    modality: str
    year: int
    month: int | None
    orbit: str | None
    remote_size: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _relative_posix(path: str | Path, root: str | Path | None = None) -> str:
    root_path = Path(root) if root is not None else _repo_root()
    path = Path(path)
    if not path.is_absolute():
        path = (root_path / path).resolve()
    return path.resolve().relative_to(root_path.resolve()).as_posix()


def _tile_region_fallback(tile_id: str) -> str:
    parts = tile_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-2])
    return tile_id


def _metadata_tile_rows(metadata_path: str | Path, split: str) -> pd.DataFrame:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    payload = json.loads(metadata_path.read_text())
    rows: list[dict[str, object]] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        tile_id = properties.get("tile_id", properties.get("name"))
        if tile_id is None:
            continue

        region_id = None
        for candidate in REGION_FIELD_CANDIDATES:
            value = properties.get(candidate)
            if value not in (None, ""):
                region_id = str(value)
                break

        rows.append(
            {
                "split": split,
                "tile_id": str(tile_id),
                "fold_id": pd.NA,
                "region_id": region_id or _tile_region_fallback(str(tile_id)),
            }
        )

    if not rows:
        raise ValueError(f"No tile metadata rows found in {metadata_path}")
    return pd.DataFrame(rows).sort_values("tile_id").reset_index(drop=True)


def load_tile_table(data_root: str | Path, split_dir: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    train_metadata_path = data_root / "metadata" / "train_tiles.geojson"
    test_metadata_path = data_root / "metadata" / "test_tiles.geojson"

    train = load_split_assignments(split_dir, train_metadata_path)
    train.insert(0, "split", "train")
    test = _metadata_tile_rows(test_metadata_path, split="test")
    return pd.concat([train, test], ignore_index=True).sort_values(["split", "tile_id"]).reset_index(
        drop=True
    )


def _is_inventory_object(remote_key: str) -> bool:
    name = remote_key.rsplit("/", 1)[-1]
    return bool(name) and not remote_key.endswith("/") and name != ".DS_Store"


def parse_s1_filename(filename: str) -> tuple[str, int, int, str]:
    match = re.fullmatch(
        r"(?P<tile_id>.+)__s1_rtc_(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<orbit>ascending|descending)\.tif",
        filename,
    )
    if match is None:
        raise ValueError(f"Invalid Sentinel-1 filename: {filename}")
    return (
        match.group("tile_id"),
        int(match.group("year")),
        int(match.group("month")),
        match.group("orbit"),
    )


def parse_s2_filename(filename: str) -> tuple[str, int, int]:
    match = re.fullmatch(
        r"(?P<tile_id>.+)__s2_l2a_(?P<year>\d{4})_(?P<month>\d{1,2})\.tif",
        filename,
    )
    if match is None:
        raise ValueError(f"Invalid Sentinel-2 filename: {filename}")
    return match.group("tile_id"), int(match.group("year")), int(match.group("month"))


def parse_aef_filename(filename: str) -> tuple[str, int]:
    match = re.fullmatch(r"(?P<tile_id>.+)_(?P<year>\d{4})\.tiff", filename)
    if match is None:
        raise ValueError(f"Invalid AEF filename: {filename}")
    return match.group("tile_id"), int(match.group("year"))


def parse_remote_key(remote_key: str, remote_size: int = 0) -> ParsedRemoteFile:
    if not _is_inventory_object(remote_key):
        raise ValueError(f"Remote key is not a data object: {remote_key}")

    parts = remote_key.split("/")
    if len(parts) < 4 or parts[0] != "makeathon-challenge":
        raise ValueError(f"Unexpected remote key layout: {remote_key}")

    source_dir = parts[1]
    split = parts[2]
    filename = parts[-1]

    if source_dir == "sentinel-1":
        try:
            tile_id, year, month, orbit = parse_s1_filename(filename)
        except ValueError:
            if len(parts) < 5:
                raise
            # Some S1 objects are nested under the timestamp/orbit stem with UUID TIFF names.
            tile_id, year, month, orbit = parse_s1_filename(parts[-2] + ".tif")
        return ParsedRemoteFile(remote_key, split, tile_id, "s1", year, month, orbit, remote_size)
    if source_dir == "sentinel-2":
        tile_id, year, month = parse_s2_filename(filename)
        return ParsedRemoteFile(remote_key, split, tile_id, "s2", year, month, None, remote_size)
    if source_dir == "aef-embeddings":
        tile_id, year = parse_aef_filename(filename)
        return ParsedRemoteFile(remote_key, split, tile_id, "aef", year, None, None, remote_size)

    raise ValueError(f"Unsupported source directory in remote key: {remote_key}")


def sequence_sort_key(record: ParsedRemoteFile) -> tuple[int, int, int]:
    month = record.month if record.month is not None else 0
    orbit = ORBIT_ORDER.get(record.orbit or "", 0)
    return record.year, month, orbit


def expected_local_path(remote_key: str, data_root: str | Path) -> Path:
    data_root = Path(data_root)
    return data_root.parent.joinpath(*remote_key.split("/"))


def is_local_complete(record: ParsedRemoteFile, data_root: str | Path) -> bool:
    local_path = expected_local_path(record.remote_key, data_root)
    if not local_path.exists() or not local_path.is_file():
        return False
    if record.remote_size <= 0:
        return True
    return local_path.stat().st_size == record.remote_size


def list_remote_records(
    *,
    bucket_name: str = DEFAULT_BUCKET_NAME,
    prefixes: Iterable[str] = REMOTE_PREFIXES,
) -> list[ParsedRemoteFile]:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    records: list[ParsedRemoteFile] = []
    for prefix in prefixes:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not _is_inventory_object(key):
                    continue
                records.append(parse_remote_key(key, int(obj.get("Size", 0))))
    return sorted(records, key=lambda record: (record.split, record.tile_id, record.modality, sequence_sort_key(record)))


def filter_records_to_tiles(
    records: Iterable[ParsedRemoteFile], tile_table: pd.DataFrame
) -> list[ParsedRemoteFile]:
    expected = {
        (row["split"], row["tile_id"])
        for row in tile_table[["split", "tile_id"]].to_dict("records")
    }
    filtered = [record for record in records if (record.split, record.tile_id) in expected]
    return sorted(filtered, key=lambda record: (record.split, record.tile_id, record.modality, sequence_sort_key(record)))


def download_missing_records(
    records: Iterable[ParsedRemoteFile],
    *,
    data_root: str | Path,
    bucket_name: str = DEFAULT_BUCKET_NAME,
    max_workers: int = 8,
) -> int:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    missing = [record for record in records if not is_local_complete(record, data_root)]
    if not missing:
        return 0

    def _download(record: ParsedRemoteFile) -> str:
        client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        target = expected_local_path(record.remote_key, data_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket_name, record.remote_key, str(target))
        return record.remote_key

    completed = 0
    workers = max(1, min(max_workers, len(missing)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download, record) for record in missing]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == len(missing):
                print(f"downloaded_missing={completed}/{len(missing)}")
    return completed


def _load_label_manifest(label_root: str | Path) -> pd.DataFrame:
    manifest_path = Path(label_root) / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Label manifest not found: {manifest_path}")
    return pd.read_parquet(manifest_path)


def _tile_lookup(tile_table: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    return {
        (row["split"], row["tile_id"]): row
        for row in tile_table.to_dict("records")
    }


def _label_lookup(label_manifest: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {row["tile_id"]: row for row in label_manifest.to_dict("records")}


def build_file_manifest(
    records: Iterable[ParsedRemoteFile],
    *,
    tile_table: pd.DataFrame,
    data_root: str | Path,
    label_manifest: pd.DataFrame,
) -> pd.DataFrame:
    repo_root = _repo_root()
    tile_map = _tile_lookup(tile_table)
    label_map = _label_lookup(label_manifest)

    reference_s2_by_tile: dict[tuple[str, str], str] = {}
    grouped: dict[tuple[str, str, str], list[ParsedRemoteFile]] = {}
    for record in records:
        grouped.setdefault((record.split, record.tile_id, record.modality), []).append(record)

    for key, group in grouped.items():
        split, tile_id, modality = key
        if modality != "s2":
            continue
        first = sorted(group, key=sequence_sort_key)[0]
        reference_s2_by_tile[(split, tile_id)] = _relative_posix(
            expected_local_path(first.remote_key, data_root), repo_root
        )

    for tile_id, label_row in label_map.items():
        key = ("train", tile_id)
        computed_ref = reference_s2_by_tile.get(key)
        label_ref = str(label_row["s2_ref_path"])
        if computed_ref is not None and computed_ref != label_ref:
            raise ValueError(
                f"S2 reference mismatch for tile {tile_id}: label={label_ref} computed={computed_ref}"
            )

    rows: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        split, tile_id, modality = key
        tile_info = tile_map.get((split, tile_id), {})
        for sequence_order, record in enumerate(sorted(group, key=sequence_sort_key)):
            local_path = expected_local_path(record.remote_key, data_root)
            local_rel = _relative_posix(local_path, repo_root)
            rows.append(
                {
                    "split": split,
                    "tile_id": tile_id,
                    "fold_id": tile_info.get("fold_id", pd.NA),
                    "region_id": tile_info.get("region_id", _tile_region_fallback(tile_id)),
                    "modality": modality,
                    "year": record.year,
                    "month": record.month,
                    "orbit": record.orbit,
                    "sequence_order": sequence_order,
                    "local_path": local_rel,
                    "remote_key": record.remote_key,
                    "exists_local": int(is_local_complete(record, data_root)),
                    "is_reference_s2": int(modality == "s2" and reference_s2_by_tile.get((split, tile_id)) == local_rel),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["split", "tile_id", "modality", "sequence_order"]
    ).reset_index(drop=True)


def _modality_dir_path(data_root: Path, split: str, tile_id: str, modality: str) -> Path:
    source_dir = MODALITY_DIRS[modality]
    if modality == "s1":
        return data_root / source_dir / split / f"{tile_id}__s1_rtc"
    if modality == "s2":
        return data_root / source_dir / split / f"{tile_id}__s2_l2a"
    raise ValueError(f"Flat AEF modality has no tile directory: {modality}")


def _modality_summary(file_manifest: pd.DataFrame, split: str, tile_id: str, modality: str) -> pd.DataFrame:
    return file_manifest[
        (file_manifest["split"] == split)
        & (file_manifest["tile_id"] == tile_id)
        & (file_manifest["modality"] == modality)
    ].sort_values("sequence_order")


def build_tile_manifest(
    *,
    tile_table: pd.DataFrame,
    file_manifest: pd.DataFrame,
    data_root: str | Path,
    label_manifest: pd.DataFrame,
) -> pd.DataFrame:
    repo_root = _repo_root()
    data_root = Path(data_root)
    label_map = _label_lookup(label_manifest)
    rows: list[dict[str, object]] = []

    for tile in tile_table.sort_values(["split", "tile_id"]).to_dict("records"):
        split = tile["split"]
        tile_id = tile["tile_id"]
        fold_id = tile.get("fold_id", pd.NA)
        region_id = tile.get("region_id", _tile_region_fallback(tile_id))

        s2 = _modality_summary(file_manifest, split, tile_id, "s2")
        s1 = _modality_summary(file_manifest, split, tile_id, "s1")
        aef = _modality_summary(file_manifest, split, tile_id, "aef")

        label_row = label_map.get(tile_id, {}) if split == "train" else {}
        labelpack_path = str(label_row.get("labelpack_path", ""))
        label_available = int(bool(labelpack_path) and (_repo_root() / labelpack_path).exists())

        s2_ref_path = ""
        if split == "train" and label_row:
            s2_ref_path = str(label_row["s2_ref_path"])
        elif not s2.empty:
            ref_rows = s2.loc[s2["is_reference_s2"] == 1, "local_path"]
            s2_ref_path = ref_rows.iloc[0] if not ref_rows.empty else s2.iloc[0]["local_path"]

        row = {
            "split": split,
            "tile_id": tile_id,
            "fold_id": fold_id,
            "region_id": region_id,
            "labelpack_path": labelpack_path,
            "label_available": label_available,
            "s2_dir_path": _relative_posix(_modality_dir_path(data_root, split, tile_id, "s2"), repo_root),
            "s2_ref_path": s2_ref_path,
            "s2_first_year": int(s2["year"].min()) if not s2.empty else pd.NA,
            "s2_first_month": int(s2.loc[s2["sequence_order"].idxmin(), "month"]) if not s2.empty else pd.NA,
            "s2_last_year": int(s2["year"].max()) if not s2.empty else pd.NA,
            "s2_last_month": int(s2.loc[s2["sequence_order"].idxmax(), "month"]) if not s2.empty else pd.NA,
            "s2_file_count_local": int(s2["exists_local"].sum()) if not s2.empty else 0,
            "s2_file_count_remote": int(len(s2)),
            "s2_complete": int((not s2.empty) and int(s2["exists_local"].sum()) == len(s2)),
            "s1_dir_path": _relative_posix(_modality_dir_path(data_root, split, tile_id, "s1"), repo_root),
            "s1_first_path": s1.iloc[0]["local_path"] if not s1.empty else "",
            "s1_file_count_local": int(s1["exists_local"].sum()) if not s1.empty else 0,
            "s1_file_count_remote": int(len(s1)),
            "s1_complete": int((not s1.empty) and int(s1["exists_local"].sum()) == len(s1)),
            "aef_first_path": aef.iloc[0]["local_path"] if not aef.empty else "",
            "aef_file_count_local": int(aef["exists_local"].sum()) if not aef.empty else 0,
            "aef_file_count_remote": int(len(aef)),
            "aef_complete": int((not aef.empty) and int(aef["exists_local"].sum()) == len(aef)),
        }
        row["all_modalities_complete"] = int(
            row["s1_complete"] == 1 and row["s2_complete"] == 1 and row["aef_complete"] == 1
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _format_counts(file_manifest: pd.DataFrame) -> str:
    summary = (
        file_manifest.groupby(["split", "modality"])
        .agg(remote_files=("remote_key", "size"), local_files=("exists_local", "sum"))
        .reset_index()
        .sort_values(["split", "modality"])
    )
    lines = ["| split | modality | local files | remote files |", "| --- | --- | ---: | ---: |"]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['split']} | {row['modality']} | {int(row['local_files']):,} | {int(row['remote_files']):,} |"
        )
    return "\n".join(lines)


def write_handoff_doc(
    *,
    output_root: str | Path,
    tile_manifest: pd.DataFrame,
    file_manifest: pd.DataFrame,
    label_root: str | Path,
) -> Path:
    output_root = Path(output_root)
    doc_path = output_root / "DATA_HANDOFF.md"
    complete_tiles = int(tile_manifest["all_modalities_complete"].sum())
    total_tiles = int(len(tile_manifest))
    incomplete_tiles = tile_manifest.loc[
        tile_manifest["all_modalities_complete"] != 1, ["split", "tile_id"]
    ]

    incomplete_text = "none"
    if not incomplete_tiles.empty:
        incomplete_text = ", ".join(
            f"{row.split}/{row.tile_id}" for row in incomplete_tiles.itertuples(index=False)
        )

    content = f"""# Model Input Handoff v1

## Included Artifacts
- `tile_manifest.csv`: one row per train/test tile with modality completeness and reference paths.
- `file_manifest.csv`: one row per raw S1/S2/AEF file with sequence order and local/remote paths.
- `{Path(label_root).as_posix()}`: weak-label package for train tiles.

## Current Completeness
- Complete tiles: {complete_tiles}/{total_tiles}
- Incomplete tiles: {incomplete_text}

{_format_counts(file_manifest)}

## Loader Contract
- Canonical grid: Sentinel-2.
- Train labels: join `tile_manifest.csv` to `{Path(label_root).as_posix()}/manifest.parquet` by `tile_id`.
- Reproject Sentinel-1 and AEF onto each tile's `s2_ref_path` grid inside B/C loaders.
- S2 sequence order is `(year, month)`.
- S1 sequence order is `(year, month, orbit)` with `ascending` before `descending`.
- AEF sequence order is `year`.

## Label Caveats
- Some train tiles have no GLAD-S2 weak-label coverage; this is encoded in `labels_v1`, not imputed.
- `labels_v1` currently has essentially no hard negatives, so prefer `soft_target` and `sample_weight`.
- `region_id` is coarse and mostly tile-prefix based because metadata has no richer region field.
"""
    doc_path.write_text(content)
    return doc_path


def prepare_model_inputs(
    *,
    data_root: str | Path,
    split_dir: str | Path,
    label_root: str | Path,
    output_root: str | Path,
    bucket_name: str = DEFAULT_BUCKET_NAME,
    verify_only: bool = False,
    force: bool = False,
    max_workers: int = 8,
) -> dict[str, object]:
    data_root = Path(data_root)
    output_root = Path(output_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    tile_manifest_path = output_root / "tile_manifest.csv"
    file_manifest_path = output_root / "file_manifest.csv"
    handoff_path = output_root / "DATA_HANDOFF.md"
    existing_outputs = [path for path in [tile_manifest_path, file_manifest_path, handoff_path] if path.exists()]
    if existing_outputs and not force:
        raise FileExistsError(
            f"Output files already exist: {[path.as_posix() for path in existing_outputs]}. Use --force."
        )

    tile_table = load_tile_table(data_root, split_dir)
    label_manifest = _load_label_manifest(label_root)
    remote_records = filter_records_to_tiles(
        list_remote_records(bucket_name=bucket_name),
        tile_table,
    )
    downloaded = 0
    if not verify_only:
        downloaded = download_missing_records(
            remote_records,
            data_root=data_root,
            bucket_name=bucket_name,
            max_workers=max_workers,
        )

    file_manifest = build_file_manifest(
        remote_records,
        tile_table=tile_table,
        data_root=data_root,
        label_manifest=label_manifest,
    )
    tile_manifest = build_tile_manifest(
        tile_table=tile_table,
        file_manifest=file_manifest,
        data_root=data_root,
        label_manifest=label_manifest,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    tile_manifest.to_csv(tile_manifest_path, index=False)
    file_manifest.to_csv(file_manifest_path, index=False)
    handoff_doc_path = write_handoff_doc(
        output_root=output_root,
        tile_manifest=tile_manifest,
        file_manifest=file_manifest,
        label_root=label_root,
    )

    if not verify_only and not (tile_manifest["all_modalities_complete"] == 1).all():
        incomplete = tile_manifest.loc[
            tile_manifest["all_modalities_complete"] != 1,
            ["split", "tile_id", "s1_complete", "s2_complete", "aef_complete"],
        ]
        raise RuntimeError(f"Some tile modalities are incomplete after download:\n{incomplete}")

    return {
        "downloaded": downloaded,
        "tile_manifest_path": tile_manifest_path,
        "file_manifest_path": file_manifest_path,
        "handoff_path": handoff_doc_path,
        "tile_count": int(len(tile_manifest)),
        "file_count": int(len(file_manifest)),
        "complete_tile_count": int(tile_manifest["all_modalities_complete"].sum()),
    }
