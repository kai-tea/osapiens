"""Prepare raw model-input manifests for train and test tiles."""

from __future__ import annotations

import argparse
from pathlib import Path

from label_pipeline.model_inputs import DEFAULT_BUCKET_NAME, prepare_model_inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare full model-input handoff manifests.")
    parser.add_argument("--data-root", default="data/makeathon-challenge")
    parser.add_argument("--split-dir", default="splits/split_v1")
    parser.add_argument("--label-root", default="artifacts/labels_v1")
    parser.add_argument("--output-root", default="artifacts/model_inputs_v1")
    parser.add_argument("--bucket-name", default=DEFAULT_BUCKET_NAME)
    parser.add_argument("--verify-only", action="store_true", help="Do not download missing files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifest outputs.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel S3 downloads.")
    args = parser.parse_args(argv)

    result = prepare_model_inputs(
        data_root=Path(args.data_root),
        split_dir=Path(args.split_dir),
        label_root=Path(args.label_root),
        output_root=Path(args.output_root),
        bucket_name=args.bucket_name,
        verify_only=args.verify_only,
        force=args.force,
        max_workers=args.max_workers,
    )

    print(f"downloaded={result['downloaded']}")
    print(f"tiles={result['tile_count']}")
    print(f"complete_tiles={result['complete_tile_count']}")
    print(f"files={result['file_count']}")
    print(f"tile_manifest={result['tile_manifest_path']}")
    print(f"file_manifest={result['file_manifest_path']}")
    print(f"handoff={result['handoff_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
