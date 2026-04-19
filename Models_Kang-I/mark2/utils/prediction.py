"""Reusable prediction helpers for Mark 2 inference and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

try:
    from .npz_data import iter_prediction_inputs
except ImportError:
    from npz_data import iter_prediction_inputs


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


def save_prediction_set(
    *,
    model: torch.nn.Module,
    input_dir: Path,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    threshold: float | None = None,
) -> list[Path]:
    """Run the model over a directory of tile artifacts and save probability outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for input_path, tile in iter_prediction_inputs(input_dir):
        probabilities = predict_probability_map(
            model=model,
            features=tile["features"],
            batch_size=batch_size,
            device=device,
        )

        output_payload = {
            "tile_id": tile.get("tile_id", np.array([input_path.stem])),
            "split": tile.get("split", np.array(["unknown"])),
            "year": tile.get("year", np.array([-1], dtype=np.int32)),
            "probabilities": probabilities.astype(np.float32, copy=False),
        }
        if "labels" in tile:
            output_payload["labels"] = tile["labels"].astype(np.int8, copy=False)
        if "valid_mask" in tile:
            output_payload["valid_mask"] = tile["valid_mask"].astype(bool, copy=False)
        if threshold is not None:
            output_payload["binary_map"] = (probabilities >= threshold).astype(np.uint8)

        output_path = output_dir / input_path.name
        np.savez_compressed(output_path, **output_payload)
        output_paths.append(output_path)

    return output_paths


def load_selected_threshold(report_path: Path) -> float:
    """Read the chosen threshold from a saved validation report JSON file."""
    payload = json.loads(report_path.read_text())
    selected_threshold = payload.get("selected_threshold")
    if not isinstance(selected_threshold, dict) or "threshold" not in selected_threshold:
        raise ValueError(f"Could not find selected_threshold.threshold in {report_path}")
    return float(selected_threshold["threshold"])
