"""LightGBM training for the Kaite v1 baseline.

Reads cached per-tile feature parquets from ``artifacts/features_v1/``,
concatenates the training tiles, trains a single ``LGBMClassifier`` on a
soft-label binarisation of Cini's weak labels, and saves the model
together with its feature list and configuration for reproducibility.

Training target:

- ``y = (soft_target >= 0.5).astype(int)`` — majority-vote binarisation of
  the weak-label consensus.
- ``weight = sample_weight`` — Cini's source-agreement weight.
- Eligible pixels: ``train_mask == 1`` (already filtered in ``data.py``).

The eval harness (``src/eval.py``) imports :func:`train_classifier` to run
per-fold cross-validation. When invoked directly, this module trains a
final model on all 16 training tiles and writes it to
``artifacts/models_v1/baseline_v1.lgb`` alongside ``metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from .data import FEATURES_ROOT, feature_names, load_tile_features, load_tile_manifest

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "artifacts" / "models_v1"
DEFAULT_MODEL_NAME = "baseline_v1"

POSITIVE_THRESHOLD = 0.5


@dataclass
class TrainConfig:
    """Hyperparameters for the LightGBM baseline."""

    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 100
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    reg_alpha: float = 0.0
    reg_lambda: float = 0.1
    seed: int = 42
    is_unbalance: bool = True
    early_stopping_rounds: int = 30
    internal_val_fraction: float = 0.2
    n_jobs: int = -1


@dataclass
class TrainArtifacts:
    """Everything we save alongside the serialised model."""

    feature_list: list[str]
    train_tiles: list[str]
    config: dict
    positive_threshold: float
    n_rows_train: int
    n_rows_val: int
    n_positives_train: int
    n_positives_val: int
    git_sha: str
    best_iteration: int | None = None
    metrics: dict = field(default_factory=dict)


def load_training_dataframe(
    tile_ids: list[str],
    features_root: Path = FEATURES_ROOT,
) -> pd.DataFrame:
    """Concatenate per-tile feature parquets into a single dataframe."""
    frames: list[pd.DataFrame] = []
    for tile_id in tile_ids:
        path = features_root / f"{tile_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing feature parquet for tile {tile_id}. "
                "Run `python -m src.data` to regenerate."
            )
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    logger.info("loaded %d rows from %d tiles", len(df), len(tile_ids))
    return df


def prepare_xy(
    df: pd.DataFrame,
    threshold: float = POSITIVE_THRESHOLD,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Split the feature dataframe into ``(X, y, weight)``.

    Rows with missing ``soft_target`` or non-positive ``sample_weight``
    are dropped to keep the objective well-defined.
    """
    if "soft_target" not in df.columns:
        raise ValueError("soft_target column missing — did you pass test-split features?")
    keep = df["soft_target"].notna() & (df["sample_weight"] > 0)
    df = df.loc[keep]
    feat_cols = feature_names()
    X = df[feat_cols].astype(np.float32)
    y = (df["soft_target"].to_numpy() >= threshold).astype(np.int8)
    w = df["sample_weight"].to_numpy(dtype=np.float32)
    return X, y, w


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _internal_train_val_split(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    groups: pd.Series | None,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, val_idx)`` for an internal early-stopping split.

    When ``groups`` is provided (per-row tile_id), pick whole tiles for
    the validation set so the model can't memorise spatial neighbours.
    Falls back to a per-row random split if the tile count is too small
    to honour ``val_fraction``.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if groups is None:
        idx = np.arange(n)
        rng.shuffle(idx)
        val_size = max(1, int(round(n * val_fraction)))
        return idx[val_size:], idx[:val_size]

    tile_counts = groups.value_counts()
    shuffled_tiles = rng.permutation(tile_counts.index.to_numpy())
    target = int(round(n * val_fraction))
    val_tiles: list[str] = []
    running = 0
    for t in shuffled_tiles:
        if running >= target:
            break
        val_tiles.append(t)
        running += int(tile_counts.loc[t])
    if not val_tiles or running == n:
        idx = np.arange(n)
        rng.shuffle(idx)
        val_size = max(1, int(round(n * val_fraction)))
        return idx[val_size:], idx[:val_size]
    val_mask = groups.isin(val_tiles).to_numpy()
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def train_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    weight: np.ndarray,
    config: TrainConfig,
    groups: pd.Series | None = None,
) -> tuple[LGBMClassifier, dict]:
    """Train an :class:`LGBMClassifier` with an internal early-stopping split.

    Returns ``(model, info)``. ``info`` carries sizes, positive counts,
    and the best iteration chosen by early stopping.
    """
    if set(np.unique(y)) - {0, 1}:
        raise ValueError(f"Expected binary target; got unique values {np.unique(y)}")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Training set is single-class — check label pipeline.")

    train_idx, val_idx = _internal_train_val_split(
        X, y, weight, groups, config.internal_val_fraction, config.seed
    )
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    w_tr, w_val = weight[train_idx], weight[val_idx]

    model = LGBMClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        feature_fraction=config.feature_fraction,
        bagging_fraction=config.bagging_fraction,
        bagging_freq=config.bagging_freq,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        objective="binary",
        is_unbalance=config.is_unbalance,
        random_state=config.seed,
        n_jobs=config.n_jobs,
        verbosity=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[
            early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=False),
            log_evaluation(period=0),
        ],
    )
    info = {
        "n_rows_train": int(len(y_tr)),
        "n_rows_val": int(len(y_val)),
        "n_positives_train": int(y_tr.sum()),
        "n_positives_val": int(y_val.sum()),
        "best_iteration": int(model.best_iteration_) if model.best_iteration_ else None,
    }
    return model, info


def save_model(
    model: LGBMClassifier,
    artifacts: TrainArtifacts,
    output_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    """Persist the booster text file and a JSON metadata sidecar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{name}.lgb"
    metadata_path = output_dir / f"{name}.json"
    model.booster_.save_model(str(model_path))
    with open(metadata_path, "w") as fh:
        json.dump(asdict(artifacts), fh, indent=2, sort_keys=True)
    logger.info("saved model -> %s", model_path)
    logger.info("saved metadata -> %s", metadata_path)
    return model_path, metadata_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiles",
        nargs="*",
        help="Train tile IDs (default: all with fold_id assigned)",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--output-dir", type=Path, default=MODELS_ROOT, help="Directory for saved model + metadata"
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=POSITIVE_THRESHOLD,
        help="soft_target >= threshold is treated as a positive label",
    )
    parser.add_argument("--n-estimators", type=int, default=TrainConfig.n_estimators)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--num-leaves", type=int, default=TrainConfig.num_leaves)
    parser.add_argument(
        "--min-child-samples", type=int, default=TrainConfig.min_child_samples
    )
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    manifest = load_tile_manifest()
    train_tiles = args.tiles or manifest.loc[
        (manifest["split"] == "train") & manifest["fold_id"].notna(), "tile_id"
    ].tolist()

    df = load_training_dataframe(train_tiles)
    X, y, w = prepare_xy(df, threshold=args.positive_threshold)
    logger.info(
        "training set: rows=%d positives=%d (%.2f%%)",
        len(y),
        int(y.sum()),
        100.0 * y.mean(),
    )

    config = TrainConfig(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        seed=args.seed,
    )
    groups = df.loc[X.index, "tile_id"] if "tile_id" in df.columns else None
    model, info = train_classifier(X, y, w, config, groups=groups)

    artifacts = TrainArtifacts(
        feature_list=feature_names(),
        train_tiles=list(train_tiles),
        config=asdict(config),
        positive_threshold=float(args.positive_threshold),
        n_rows_train=info["n_rows_train"],
        n_rows_val=info["n_rows_val"],
        n_positives_train=info["n_positives_train"],
        n_positives_val=info["n_positives_val"],
        git_sha=_git_sha(),
        best_iteration=info["best_iteration"],
    )
    save_model(model, artifacts, args.output_dir, args.model_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
