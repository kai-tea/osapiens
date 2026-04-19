"""Runner script for the Mark 2 embedding-only baseline pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .pipeline import run_embedding_label_pipeline
except ImportError:
    from pipeline import run_embedding_label_pipeline


DEFAULT_DATA_ROOT = Path("data/makeathon-challenge")
DEFAULT_OUTPUT_DIR = Path("Models_Kang-I/mark2/outputs/baseline_v1")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the Mark 2 baseline runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT, help="Challenge data root directory.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where normalized features, labels, and stats will be saved.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Embedding year to process. Defaults to the latest year common to train and test.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.25,
        help="Deterministic fraction of train tiles reserved for validation.",
    )
    parser.add_argument(
        "--include_test",
        action="store_true",
        help="Also export normalized test embeddings using the train-fit normalization statistics.",
    )
    return parser


def main() -> None:
    """Run the embedding preprocessing and weak-label generation pipeline."""
    parser = build_argument_parser()
    args = parser.parse_args()
    summary = run_embedding_label_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        year=args.year,
        val_fraction=args.val_fraction,
        include_test=args.include_test,
    )

    print(f"Embedding year: {summary['year']}")
    print(f"Normalization stats: {summary['stats_path']}")
    for split_name, counts in summary["summaries"].items():
        print(
            f"{split_name}: "
            f"positive={counts['positive']} "
            f"negative={counts['negative']} "
            f"uncertain={counts['uncertain']}"
        )


if __name__ == "__main__":
    main()
