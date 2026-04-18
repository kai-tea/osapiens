"""CLI for building weak-label artifacts."""

from __future__ import annotations

import argparse

from .build import run_build


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build labelpack artifacts for the osapiens challenge.")
    parser.add_argument(
        "--data-root",
        default="data/makeathon-challenge",
        help="Path to the downloaded challenge dataset root.",
    )
    parser.add_argument(
        "--split-dir",
        default="splits/split_v1",
        help="Path to the frozen split directory.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/labels_v1",
        help="Path where labelpack artifacts will be written.",
    )
    parser.add_argument(
        "--tile-id",
        default=None,
        help="Optional tile_id to build for a single-tile smoke test.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_build(
        data_root=args.data_root,
        split_dir=args.split_dir,
        output_root=args.output_root,
        tile_id=args.tile_id,
        force=args.force,
    )
    return 0
