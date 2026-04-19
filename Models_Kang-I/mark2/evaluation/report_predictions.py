"""Build a validation report from saved prediction `.npz` outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from ..utils.evaluation import DEFAULT_THRESHOLD_SWEEP, build_validation_report, save_json_report
    from ..utils.npz_data import list_npz_files, load_tile_npz
except ImportError:
    from utils.evaluation import DEFAULT_THRESHOLD_SWEEP, build_validation_report, save_json_report
    from utils.npz_data import list_npz_files, load_tile_npz


DEFAULT_INPUT_DIR = Path("Models_Kang-I/mark2/outputs/predictions/mlp_validation")
DEFAULT_OUTPUT_PATH = Path("Models_Kang-I/mark2/outputs/models/mlp_validation_report.json")


def parse_thresholds(raw_thresholds: str | None) -> tuple[float, ...]:
    """Parse an optional comma-separated threshold list."""
    if not raw_thresholds:
        return DEFAULT_THRESHOLD_SWEEP
    thresholds = sorted(set(float(value.strip()) for value in raw_thresholds.split(",") if value.strip()))
    return tuple(thresholds)


def load_labels_and_probabilities(prediction_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Collect valid labels and probabilities from saved prediction tiles."""
    labels_list: list[np.ndarray] = []
    probabilities_list: list[np.ndarray] = []

    for path in list_npz_files(prediction_dir):
        tile = load_tile_npz(path)
        if "labels" not in tile or "valid_mask" not in tile:
            raise ValueError(f"Prediction file {path} is missing labels or valid_mask needed for validation reporting")

        labels = tile["labels"].reshape(-1).astype(np.int64, copy=False)
        valid_mask = tile["valid_mask"].reshape(-1).astype(bool, copy=False)
        probabilities = tile["probabilities"].reshape(-1).astype(np.float32, copy=False)
        keep_mask = valid_mask & ((labels == 0) | (labels == 1))

        labels_list.append(labels[keep_mask])
        probabilities_list.append(probabilities[keep_mask])

    if not labels_list:
        raise FileNotFoundError(f"No usable prediction files found in {prediction_dir}")

    return np.concatenate(labels_list), np.concatenate(probabilities_list)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for validation report generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with validation prediction `.npz` files.")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path for the JSON validation report.")
    parser.add_argument("--validation_loss", type=float, default=0.0, help="Optional validation loss to record in the report.")
    parser.add_argument("--default_threshold", type=float, default=0.5, help="Threshold used for the default metric block.")
    parser.add_argument("--thresholds", type=str, default=None, help="Optional comma-separated threshold sweep.")
    return parser


def main() -> None:
    """Generate a validation report from saved probability maps."""
    args = build_argument_parser().parse_args()
    labels, probabilities = load_labels_and_probabilities(args.input_dir)
    report = build_validation_report(
        labels=labels,
        probabilities=probabilities,
        validation_loss=args.validation_loss,
        default_threshold=args.default_threshold,
        thresholds=parse_thresholds(args.thresholds),
    )
    save_json_report(report, args.output_path)
    selected_threshold = report["selected_threshold"]
    print(f"saved {args.output_path}")
    if isinstance(selected_threshold, dict):
        print(f"selected_threshold={selected_threshold['threshold']:.2f} selected_f1={selected_threshold['f1']:.4f}")


if __name__ == "__main__":
    main()
