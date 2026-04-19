"""Train a small MLP on Mark 2 embedding-only `.npz` artifacts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from ..models.mlp import build_mlp, save_model_checkpoint
    from ..utils.npz_data import count_split_pixels, infer_input_dim_from_npz, iterate_eval_batches, iterate_training_batches
except ImportError:
    from models.mlp import build_mlp, save_model_checkpoint
    from utils.npz_data import count_split_pixels, infer_input_dim_from_npz, iterate_eval_batches, iterate_training_batches


DEFAULT_TRAIN_DIR = Path("Models_Kang-I/mark2/outputs/baseline_v1/train")
DEFAULT_VALIDATION_DIR = Path("Models_Kang-I/mark2/outputs/baseline_v1/validation")
DEFAULT_MODEL_PATH = Path("Models_Kang-I/mark2/outputs/models/mlp_best.pt")
DEFAULT_HISTORY_PATH = Path("Models_Kang-I/mark2/outputs/models/mlp_history.json")
DEFAULT_HIDDEN_DIM = 64
DEFAULT_DROPOUT = 0.1
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 4096
DEFAULT_EPOCHS = 20
DEFAULT_SEED = 42
DEFAULT_THRESHOLD = 0.5


def set_deterministic_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick the training device, preferring CUDA when available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    """Compute simple validation metrics from logits and binary labels."""
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    labels_bool = labels >= 0.5

    true_positive = torch.count_nonzero(predictions & labels_bool).item()
    true_negative = torch.count_nonzero((~predictions) & (~labels_bool)).item()
    false_positive = torch.count_nonzero(predictions & (~labels_bool)).item()
    false_negative = torch.count_nonzero((~predictions) & labels_bool).item()
    total = labels.numel()

    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    positive_rate = predictions.float().mean().item() if total else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "positive_prediction_rate": positive_rate,
    }


def run_training_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_dir: Path,
    batch_size: int,
    device: torch.device,
    epoch_index: int,
    seed: int,
) -> float:
    """Run one training epoch over all valid training pixels."""
    model.train()
    total_loss = 0.0
    total_examples = 0

    for features_np, labels_np in iterate_training_batches(train_dir, batch_size=batch_size, epoch_seed=seed + epoch_index):
        features = torch.from_numpy(features_np).to(device=device, dtype=torch.float32)
        labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size_actual = labels.shape[0]
        total_loss += loss.item() * batch_size_actual
        total_examples += batch_size_actual

    return total_loss / total_examples if total_examples else 0.0


def evaluate_model(
    model: nn.Module,
    loss_fn: nn.Module,
    validation_dir: Path,
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    """Evaluate loss and binary metrics on the validation split."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for features_np, labels_np in iterate_eval_batches(validation_dir, batch_size=batch_size):
            features = torch.from_numpy(features_np).to(device=device, dtype=torch.float32)
            labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.float32)
            logits = model(features)
            loss = loss_fn(logits, labels)

            batch_size_actual = labels.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    validation_loss = total_loss / total_examples if total_examples else 0.0
    if not all_labels:
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "positive_prediction_rate": 0.0}
    else:
        metrics = compute_binary_metrics(
            logits=torch.cat(all_logits, dim=0),
            labels=torch.cat(all_labels, dim=0),
            threshold=threshold,
        )

    metrics["validation_loss"] = validation_loss
    return metrics


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for small-MLP training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", type=Path, default=DEFAULT_TRAIN_DIR, help="Directory with training `.npz` tiles.")
    parser.add_argument(
        "--validation_dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help="Directory with validation `.npz` tiles.",
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH, help="Path for the best model checkpoint.")
    parser.add_argument("--history_path", type=Path, default=DEFAULT_HISTORY_PATH, help="Path for the JSON training history.")
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM, help="MLP hidden dimension.")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout rate in the first hidden block.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Adam learning rate.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic training.")
    return parser


def main() -> None:
    """Train the small MLP and save the best checkpoint plus history."""
    args = build_argument_parser().parse_args()
    set_deterministic_seed(args.seed)
    device = get_device()

    input_dim = infer_input_dim_from_npz(args.train_dir)
    train_counts = count_split_pixels(args.train_dir)
    validation_counts = count_split_pixels(args.validation_dir)

    model = build_mlp(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    # Extension point: class weighting for imbalance can be added to BCE here later.
    # Extension point: focal loss can replace BCE here later.
    optimizer = Adam(model.parameters(), lr=args.lr)

    history: dict[str, object] = {
        "config": {
            "train_dir": str(args.train_dir),
            "validation_dir": str(args.validation_dir),
            "model_path": str(args.model_path),
            "history_path": str(args.history_path),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed": args.seed,
            "threshold": DEFAULT_THRESHOLD,
            "device": str(device),
        },
        "data_summary": {
            "train": train_counts,
            "validation": validation_counts,
        },
        "epochs": [],
        "best_epoch": None,
        "best_validation_loss": None,
    }

    best_validation_loss = float("inf")
    for epoch_index in range(args.epochs):
        train_loss = run_training_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_dir=args.train_dir,
            batch_size=args.batch_size,
            device=device,
            epoch_index=epoch_index,
            seed=args.seed,
        )
        validation_metrics = evaluate_model(
            model=model,
            loss_fn=loss_fn,
            validation_dir=args.validation_dir,
            batch_size=args.batch_size,
            device=device,
            threshold=DEFAULT_THRESHOLD,
        )

        epoch_record = {
            "epoch": epoch_index + 1,
            "train_loss": train_loss,
            "validation_loss": validation_metrics["validation_loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "positive_prediction_rate": validation_metrics["positive_prediction_rate"],
        }
        history["epochs"].append(epoch_record)

        if validation_metrics["validation_loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["validation_loss"]
            history["best_epoch"] = epoch_index + 1
            history["best_validation_loss"] = best_validation_loss
            save_model_checkpoint(
                model,
                args.model_path,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            )

        print(
            f"epoch={epoch_index + 1} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={validation_metrics['validation_loss']:.6f} "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"val_prec={validation_metrics['precision']:.4f} "
            f"val_rec={validation_metrics['recall']:.4f} "
            f"pos_rate={validation_metrics['positive_prediction_rate']:.4f}"
        )

    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    args.history_path.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
