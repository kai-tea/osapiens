"""Run a saved small MLP on Mark 2 `.npz` tiles and export probability maps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from ..models.mlp import load_model_checkpoint
    from ..utils.prediction import load_selected_threshold, save_prediction_set
except ImportError:
    from models.mlp import load_model_checkpoint
    from utils.prediction import load_selected_threshold, save_prediction_set


DEFAULT_INPUT_DIR = Path("Models_Kang-I/mark2/outputs/baseline_v1/validation")
DEFAULT_MODEL_PATH = Path("Models_Kang-I/mark2/outputs/models/mlp_best.pt")
DEFAULT_OUTPUT_DIR = Path("Models_Kang-I/mark2/outputs/predictions/mlp_validation")
DEFAULT_BATCH_SIZE = 4096
DEFAULT_THRESHOLD = 0.5


def get_device() -> torch.device:
    """Pick the inference device, preferring CUDA when available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for MLP prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with input `.npz` tiles.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the saved MLP checkpoint.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for prediction `.npz` outputs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size used during inference.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold for optional binary maps.")
    parser.add_argument(
        "--threshold_report",
        type=Path,
        default=None,
        help="Optional validation report JSON from which to read the selected threshold.",
    )
    parser.add_argument(
        "--save_binary",
        action="store_true",
        help="Also save a thresholded binary prediction map alongside probabilities.",
    )
    return parser


def main() -> None:
    """Load a trained MLP, predict tile probability maps, and save them as `.npz`."""
    args = build_argument_parser().parse_args()
    device = get_device()
    model = load_model_checkpoint(args.model_path, map_location=device).to(device)
    chosen_threshold = load_selected_threshold(args.threshold_report) if args.threshold_report is not None else args.threshold

    # Extension point: forest-map prior features can be added to the model inputs later.
    output_paths = save_prediction_set(
        model=model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device,
        threshold=chosen_threshold if args.save_binary else None,
    )
    for output_path in output_paths:
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
