"""Run a saved small MLP on Mark 2 `.npz` tiles and export probability maps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from ..models.mlp import load_model_checkpoint
    from ..utils.npz_data import iter_prediction_inputs
except ImportError:
    from models.mlp import load_model_checkpoint
    from utils.npz_data import iter_prediction_inputs


DEFAULT_INPUT_DIR = Path("Models_Kang-I/mark2/outputs/baseline_v1/validation")
DEFAULT_MODEL_PATH = Path("Models_Kang-I/mark2/outputs/models/mlp_best.pt")
DEFAULT_OUTPUT_DIR = Path("Models_Kang-I/mark2/outputs/predictions/mlp_validation")
DEFAULT_BATCH_SIZE = 4096
DEFAULT_THRESHOLD = 0.5


def get_device() -> torch.device:
    """Pick the inference device, preferring CUDA when available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_probability_map(
    model: torch.nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predict a dense probability map from per-pixel features."""
    height, width, channels = features.shape
    flat_features = features.reshape(-1, channels).astype(np.float32, copy=False)
    probabilities = np.empty(flat_features.shape[0], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, flat_features.shape[0], batch_size):
            end = start + batch_size
            batch = torch.from_numpy(flat_features[start:end]).to(device=device, dtype=torch.float32)
            logits = model(batch)
            probabilities[start:end] = torch.sigmoid(logits).cpu().numpy()

    return probabilities.reshape(height, width)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for MLP prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with input `.npz` tiles.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the saved MLP checkpoint.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for prediction `.npz` outputs.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size used during inference.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold for optional binary maps.")
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

    # Extension point: calibration threshold tuning can be added here later.
    # Extension point: forest-map prior features can be added to the model inputs later.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path, tile in iter_prediction_inputs(args.input_dir):
        probabilities = predict_probability_map(
            model=model,
            features=tile["features"],
            batch_size=args.batch_size,
            device=device,
        )

        output_payload = {
            "tile_id": tile.get("tile_id", np.array([input_path.stem])),
            "split": tile.get("split", np.array(["unknown"])),
            "year": tile.get("year", np.array([-1], dtype=np.int32)),
            "probabilities": probabilities.astype(np.float32, copy=False),
        }
        if args.save_binary:
            output_payload["binary_map"] = (probabilities >= args.threshold).astype(np.uint8)

        output_path = args.output_dir / input_path.name
        np.savez_compressed(output_path, **output_payload)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
