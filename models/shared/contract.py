"""Shared paths and output schemas for all model sandboxes."""

from __future__ import annotations

from pathlib import Path


TEAM_MEMBERS = ("cini", "kaite", "kangi", "tomy")

TILE_MANIFEST_PATH = Path("artifacts/model_inputs_v1/tile_manifest.csv")
FILE_MANIFEST_PATH = Path("artifacts/model_inputs_v1/file_manifest.csv")
LABEL_MANIFEST_PATH = Path("artifacts/labels_v1/manifest.parquet")
PIXEL_INDEX_PATH = Path("artifacts/labels_v1/pixel_index.parquet")
SPLIT_ASSIGNMENTS_PATH = Path("cini/splits/split_v1/fold_assignments.csv")

VALIDATION_PREDICTION_COLUMNS = (
    "tile_id",
    "row",
    "col",
    "y_true",
    "score",
    "fold_id",
    "model_name",
)
TEST_PREDICTION_COLUMNS = ("tile_id", "row", "col", "score", "model_name")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def output_dir_for_owner(owner: str, root: str | Path | None = None) -> Path:
    if owner not in TEAM_MEMBERS:
        raise ValueError(f"Unknown owner '{owner}'. Expected one of {TEAM_MEMBERS}.")
    base = Path(root) if root is not None else repo_root()
    return base / "artifacts" / "models" / owner


def required_manifest_paths(root: str | Path | None = None) -> dict[str, Path]:
    base = Path(root) if root is not None else repo_root()
    return {
        "tile_manifest": base / TILE_MANIFEST_PATH,
        "file_manifest": base / FILE_MANIFEST_PATH,
        "label_manifest": base / LABEL_MANIFEST_PATH,
        "pixel_index": base / PIXEL_INDEX_PATH,
        "split_assignments": base / SPLIT_ASSIGNMENTS_PATH,
    }
