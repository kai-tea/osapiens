"""Starter CLI behavior shared by personal model sandboxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_simple_config
from .contract import TEST_PREDICTION_COLUMNS, VALIDATION_PREDICTION_COLUMNS, output_dir_for_owner
from .data import load_manifests, split_tiles
from .evaluation import binary_metrics, validate_prediction_schema


def _default_config_path(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parent / "config.example.yaml"


def _parser(description: str, script_file: str | Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=_default_config_path(script_file).as_posix())
    parser.add_argument("--active-fold", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default=None)
    return parser


def _load_run_context(owner: str, script_file: str | Path, argv: list[str] | None, description: str):
    args = _parser(description, script_file).parse_args(argv)
    config = load_simple_config(args.config)
    active_fold = args.active_fold if args.active_fold is not None else int(config.get("active_fold", 0))
    model_name = args.model_name or str(config.get("model_name", f"{owner}_starter"))
    output_dir = Path(args.output_dir) if args.output_dir else output_dir_for_owner(owner)
    return config, active_fold, model_name, output_dir


def run_train(owner: str, script_file: str | Path, argv: list[str] | None = None) -> int:
    config, active_fold, model_name, output_dir = _load_run_context(
        owner,
        script_file,
        argv,
        f"Smoke-train starter for {owner}'s sandbox.",
    )
    manifests = load_manifests()
    folds = split_tiles(manifests.tile_manifest, active_fold=active_fold)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "owner": owner,
        "model_name": model_name,
        "active_fold": active_fold,
        "train_tiles": int(len(folds.train_tiles)),
        "val_tiles": int(len(folds.val_tiles)),
        "test_tiles": int(len(folds.test_tiles)),
        "pixel_index_rows": int(len(manifests.pixel_index)),
        "note": "Starter only: replace with real model training logic.",
        "config": config,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote_summary={summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_predict(owner: str, script_file: str | Path, argv: list[str] | None = None) -> int:
    _, _, model_name, output_dir = _load_run_context(
        owner,
        script_file,
        argv,
        f"Write prediction templates for {owner}'s sandbox.",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_template = pd.DataFrame(columns=VALIDATION_PREDICTION_COLUMNS)
    test_template = pd.DataFrame(columns=TEST_PREDICTION_COLUMNS)
    validation_template["model_name"] = validation_template.get("model_name", model_name)
    test_template["model_name"] = test_template.get("model_name", model_name)

    val_path = output_dir / "validation_predictions_template.csv"
    test_path = output_dir / "test_predictions_template.csv"
    validation_template.to_csv(val_path, index=False)
    test_template.to_csv(test_path, index=False)
    print(f"wrote_validation_template={val_path}")
    print(f"wrote_test_template={test_path}")
    return 0


def run_evaluate(owner: str, script_file: str | Path, argv: list[str] | None = None) -> int:
    _, _, _, output_dir = _load_run_context(
        owner,
        script_file,
        argv,
        f"Evaluate validation predictions for {owner}'s sandbox.",
    )
    prediction_path = output_dir / "validation_predictions.csv"
    if not prediction_path.exists():
        print(f"missing_predictions={prediction_path}")
        print("Expected validation schema:")
        print(",".join(VALIDATION_PREDICTION_COLUMNS))
        return 0

    predictions = pd.read_csv(prediction_path)
    validate_prediction_schema(predictions, split="validation")
    metrics = binary_metrics(predictions["y_true"], predictions["score"])
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"wrote_metrics={metrics_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0
