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
    from ..evaluation.report_predictions import load_labels_and_probabilities
    from ..models.mlp import build_mlp, save_model_checkpoint
    from ..utils.evaluation import DEFAULT_THRESHOLD_SWEEP, build_validation_report, save_json_report
    from ..utils.npz_data import count_split_pixels, infer_input_dim_from_npz, iterate_eval_batches, iterate_training_batches
    from ..utils.prediction import save_prediction_set
except ImportError:
    from evaluation.report_predictions import load_labels_and_probabilities
    from models.mlp import build_mlp, save_model_checkpoint
    from utils.evaluation import DEFAULT_THRESHOLD_SWEEP, build_validation_report, save_json_report
    from utils.npz_data import count_split_pixels, infer_input_dim_from_npz, iterate_eval_batches, iterate_training_batches
    from utils.prediction import save_prediction_set


DEFAULT_TRAIN_DIR = Path("Models_Kang-I/mark2/outputs/baseline_v1/train")
DEFAULT_VALIDATION_DIR = Path("Models_Kang-I/mark2/outputs/baseline_v1/validation")
DEFAULT_MODEL_PATH = Path("Models_Kang-I/mark2/outputs/models/mlp_best.pt")
DEFAULT_HISTORY_PATH = Path("Models_Kang-I/mark2/outputs/models/mlp_history.json")
DEFAULT_VALIDATION_PREDICTION_DIR = Path("Models_Kang-I/mark2/outputs/predictions/mlp_validation")
DEFAULT_VALIDATION_REPORT_PATH = Path("Models_Kang-I/mark2/outputs/models/mlp_validation_report.json")
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
) -> dict[str, object]:
    """Evaluate loss and return raw validation probabilities for later reporting."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for features_np, labels_np in iterate_eval_batches(validation_dir, batch_size=batch_size):
            features = torch.from_numpy(features_np).to(device=device, dtype=torch.float32)
            labels = torch.from_numpy(labels_np).to(device=device, dtype=torch.float32)
            logits = model(features)
            loss = loss_fn(logits, labels)
            probabilities = torch.sigmoid(logits)

            batch_size_actual = labels.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    validation_loss = total_loss / total_examples if total_examples else 0.0
    if not all_labels:
        return {
            "validation_loss": validation_loss,
            "labels": np.array([], dtype=np.int64),
            "probabilities": np.array([], dtype=np.float32),
        }

    return {
        "validation_loss": validation_loss,
        "labels": np.concatenate(all_labels).astype(np.int64, copy=False),
        "probabilities": np.concatenate(all_probabilities).astype(np.float32, copy=False),
        "default_threshold": threshold,
    }


def compute_positive_class_weight(train_counts: dict[str, int], device: torch.device) -> torch.Tensor:
    """Compute the positive class weight from split counts for weighted BCE."""
    positive_count = int(train_counts["positive"])
    negative_count = int(train_counts["negative"])
    if positive_count <= 0:
        raise ValueError("Training split contains zero positive pixels; cannot compute pos_weight")
    if negative_count <= 0:
        raise ValueError("Training split contains zero negative pixels; cannot compute pos_weight")
    return torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)


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
    parser.add_argument(
        "--validation_prediction_dir",
        type=Path,
        default=DEFAULT_VALIDATION_PREDICTION_DIR,
        help="Directory where best-model validation prediction `.npz` files will be saved.",
    )
    parser.add_argument(
        "--validation_report_path",
        type=Path,
        default=DEFAULT_VALIDATION_REPORT_PATH,
        help="Path for the best-model validation report JSON.",
    )
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
    pos_weight = compute_positive_class_weight(train_counts, device=device)

    model = build_mlp(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
            "threshold_sweep": list(DEFAULT_THRESHOLD_SWEEP),
            "device": str(device),
            "pos_weight": float(pos_weight.item()),
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
        validation_report = build_validation_report(
            labels=validation_metrics["labels"],
            probabilities=validation_metrics["probabilities"],
            validation_loss=float(validation_metrics["validation_loss"]),
            default_threshold=DEFAULT_THRESHOLD,
            thresholds=DEFAULT_THRESHOLD_SWEEP,
        )
        default_metrics = validation_report["default_threshold_metrics"]

        epoch_record = {
            "epoch": epoch_index + 1,
            "train_loss": train_loss,
            "validation_loss": float(validation_metrics["validation_loss"]),
            "validation_accuracy": default_metrics["accuracy"],
            "validation_precision": default_metrics["precision"],
            "validation_recall": default_metrics["recall"],
            "validation_f1": default_metrics["f1"],
            "validation_average_precision": default_metrics["average_precision"],
            "positive_prediction_rate": default_metrics["positive_prediction_rate"],
            "true_positive": default_metrics["true_positive"],
            "true_negative": default_metrics["true_negative"],
            "false_positive": default_metrics["false_positive"],
            "false_negative": default_metrics["false_negative"],
            "positive_probability_mean": default_metrics["positive_probability_mean"],
            "negative_probability_mean": default_metrics["negative_probability_mean"],
            "positive_probability_p10": default_metrics["positive_probability_p10"],
            "positive_probability_p50": default_metrics["positive_probability_p50"],
            "positive_probability_p90": default_metrics["positive_probability_p90"],
            "negative_probability_p10": default_metrics["negative_probability_p10"],
            "negative_probability_p50": default_metrics["negative_probability_p50"],
            "negative_probability_p90": default_metrics["negative_probability_p90"],
        }
        history["epochs"].append(epoch_record)

        if float(validation_metrics["validation_loss"]) < best_validation_loss:
            best_validation_loss = float(validation_metrics["validation_loss"])
            history["best_epoch"] = epoch_index + 1
            history["best_validation_loss"] = best_validation_loss
            save_model_checkpoint(
                model,
                args.model_path,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            )
            save_prediction_set(
                model=model,
                input_dir=args.validation_dir,
                output_dir=args.validation_prediction_dir,
                batch_size=args.batch_size,
                device=device,
                threshold=None,
            )
            saved_labels, saved_probabilities = load_labels_and_probabilities(args.validation_prediction_dir)
            saved_report = build_validation_report(
                labels=saved_labels,
                probabilities=saved_probabilities,
                validation_loss=best_validation_loss,
                default_threshold=DEFAULT_THRESHOLD,
                thresholds=DEFAULT_THRESHOLD_SWEEP,
            )
            save_json_report(saved_report, args.validation_report_path)

        print(
            f"epoch={epoch_index + 1} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={float(validation_metrics['validation_loss']):.6f} "
            f"val_acc={default_metrics['accuracy']:.4f} "
            f"val_prec={default_metrics['precision']:.4f} "
            f"val_rec={default_metrics['recall']:.4f} "
            f"val_f1={default_metrics['f1']:.4f} "
            f"val_ap={default_metrics['average_precision']:.4f} "
            f"pos_rate={default_metrics['positive_prediction_rate']:.4f}"
        )

    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    args.history_path.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
