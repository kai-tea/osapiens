"""Generate handoff and slide-ready label quality summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from label_pipeline.report import write_label_quality_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write tile/region/markdown label quality summaries.")
    parser.add_argument(
        "--output-root",
        default="artifacts/labels_v1",
        help="Label artifact root containing manifest.parquet, pixel_index.parquet, and source_overlap.csv.",
    )
    args = parser.parse_args(argv)

    result = write_label_quality_report(Path(args.output_root))
    overview = result["overview"]

    print(f"tiles={overview['n_tiles']}")
    print(f"regions={overview['n_regions']}")
    print(f"train_pixels={overview['total_train_pixels']}")
    print(f"hard_pos={overview['total_hard_pos']}")
    print(f"hard_neg={overview['total_hard_neg']}")
    print(f"ambiguous={overview['total_ambiguous']}")
    print(f"seed={overview['total_seed']}")
    print(f"tiles_missing_glads2={overview['n_tiles_missing_glads2']}")
    print(f"tile_summary={result['tile_summary_path']}")
    print(f"region_summary={result['region_summary_path']}")
    print(f"report={result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
