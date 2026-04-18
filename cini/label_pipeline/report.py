"""Reporting helpers for label quality summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SOURCE_OBS_BANDS = {
    "radd": 7,
    "gladl": 11,
    "glads2": 15,
}


@dataclass(frozen=True)
class LabelQualityPaths:
    manifest: Path
    pixel_index: Path
    source_overlap: Path
    by_tile_csv: Path
    by_region_csv: Path
    report_md: Path


def resolve_label_quality_paths(output_root: str | Path) -> LabelQualityPaths:
    output_root = Path(output_root)
    return LabelQualityPaths(
        manifest=output_root / "manifest.parquet",
        pixel_index=output_root / "pixel_index.parquet",
        source_overlap=output_root / "source_overlap.csv",
        by_tile_csv=output_root / "label_quality_by_tile.csv",
        by_region_csv=output_root / "label_quality_by_region.csv",
        report_md=output_root / "label_quality_report.md",
    )


def _ensure_artifacts_exist(paths: LabelQualityPaths) -> None:
    for path in (paths.manifest, paths.pixel_index, paths.source_overlap):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")


def load_label_artifacts(output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = resolve_label_quality_paths(output_root)
    _ensure_artifacts_exist(paths)

    manifest = pd.read_parquet(paths.manifest)
    pixel_index = pd.read_parquet(
        paths.pixel_index,
        columns=["tile_id", "region_id", "fold_id", "obs_count", "hard_label", "seed_mask"],
    )
    overlap = pd.read_csv(paths.source_overlap)
    return manifest, pixel_index, overlap


def collect_tile_source_stats(manifest: pd.DataFrame, repo_root: str | Path | None = None) -> pd.DataFrame:
    import rasterio

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    repo_root = Path(repo_root)

    rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        labelpack_path = repo_root / row.labelpack_path
        if not labelpack_path.exists():
            raise FileNotFoundError(f"Labelpack not found: {labelpack_path}")

        with rasterio.open(labelpack_path) as src:
            tile_row = {"tile_id": row.tile_id}
            for source_name, band_index in SOURCE_OBS_BANDS.items():
                obs_pixels = int(src.read(band_index).sum())
                tile_row[f"{source_name}_obs_pixels"] = obs_pixels
                tile_row[f"{source_name}_available"] = int(obs_pixels > 0)
            rows.append(tile_row)

    return pd.DataFrame(rows).sort_values("tile_id").reset_index(drop=True)


def build_tile_summary(
    manifest: pd.DataFrame,
    pixel_index: pd.DataFrame,
    tile_source_stats: pd.DataFrame,
) -> pd.DataFrame:
    pixel_index = pixel_index.copy()
    pixel_index["seed_positive"] = (
        (pixel_index["seed_mask"] == 1) & (pixel_index["hard_label"] == 1)
    ).astype(int)
    pixel_index["seed_negative"] = (
        (pixel_index["seed_mask"] == 1) & (pixel_index["hard_label"] == 0)
    ).astype(int)

    pixel_summary = (
        pixel_index.groupby("tile_id")
        .agg(
            n_train_pixels=("obs_count", "size"),
            n_obs2=("obs_count", lambda s: int((s == 2).sum())),
            n_obs3=("obs_count", lambda s: int((s == 3).sum())),
            n_hard_pos=("hard_label", lambda s: int((s == 1).sum())),
            n_hard_neg=("hard_label", lambda s: int((s == 0).sum())),
            n_ambiguous=("hard_label", lambda s: int((s == 255).sum())),
            n_seed=("seed_mask", lambda s: int((s == 1).sum())),
            n_seed_pos=("seed_positive", "sum"),
            n_seed_neg=("seed_negative", "sum"),
        )
        .reset_index()
    )

    summary = manifest.merge(pixel_summary, on="tile_id", how="left", suffixes=("", "_pixel"))
    summary = summary.merge(tile_source_stats, on="tile_id", how="left")

    summary["n_train_pixels"] = summary["n_train_pixels_pixel"].fillna(summary["n_train_pixels"]).astype(int)
    summary = summary.drop(columns=["n_train_pixels_pixel"])
    for column in [
        "n_obs2",
        "n_obs3",
        "n_hard_pos",
        "n_hard_neg",
        "n_ambiguous",
        "n_seed",
        "n_seed_pos",
        "n_seed_neg",
        "radd_obs_pixels",
        "gladl_obs_pixels",
        "glads2_obs_pixels",
        "radd_available",
        "gladl_available",
        "glads2_available",
    ]:
        summary[column] = summary[column].fillna(0).astype(int)

    summary["obs2_share"] = summary["n_obs2"] / summary["n_train_pixels"]
    summary["obs3_share"] = summary["n_obs3"] / summary["n_train_pixels"]
    summary["hard_pos_rate"] = summary["n_hard_pos"] / summary["n_train_pixels"]
    summary["ambiguous_rate"] = summary["n_ambiguous"] / summary["n_train_pixels"]
    summary["missing_glads2"] = (summary["glads2_available"] == 0).astype(int)
    summary["n_seed"] = summary["n_seed"].astype(int)

    columns = [
        "tile_id",
        "fold_id",
        "region_id",
        "labelpack_path",
        "s2_ref_path",
        "n_train_pixels",
        "n_hard_pos",
        "n_hard_neg",
        "n_seed",
        "n_seed_pos",
        "n_seed_neg",
        "n_ambiguous",
        "hard_pos_rate",
        "ambiguous_rate",
        "n_obs2",
        "n_obs3",
        "obs2_share",
        "obs3_share",
        "radd_available",
        "gladl_available",
        "glads2_available",
        "missing_glads2",
        "radd_obs_pixels",
        "gladl_obs_pixels",
        "glads2_obs_pixels",
        "radd_pos_rate",
        "gladl_pos_rate",
        "glads2_pos_rate",
    ]
    return summary[columns].sort_values(["fold_id", "tile_id"]).reset_index(drop=True)


def build_region_summary(tile_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        tile_summary.groupby("region_id")
        .agg(
            n_tiles=("tile_id", "size"),
            total_train_pixels=("n_train_pixels", "sum"),
            total_hard_pos=("n_hard_pos", "sum"),
            total_hard_neg=("n_hard_neg", "sum"),
            total_ambiguous=("n_ambiguous", "sum"),
            total_seed=("n_seed", "sum"),
            total_seed_pos=("n_seed_pos", "sum"),
            total_seed_neg=("n_seed_neg", "sum"),
            tiles_missing_glads2=("missing_glads2", "sum"),
            tiles_with_radd=("radd_available", "sum"),
            tiles_with_gladl=("gladl_available", "sum"),
            tiles_with_glads2=("glads2_available", "sum"),
            obs2_pixels=("n_obs2", "sum"),
            obs3_pixels=("n_obs3", "sum"),
        )
        .reset_index()
    )
    grouped["hard_pos_rate"] = grouped["total_hard_pos"] / grouped["total_train_pixels"]
    grouped["ambiguous_rate"] = grouped["total_ambiguous"] / grouped["total_train_pixels"]
    grouped["obs2_share"] = grouped["obs2_pixels"] / grouped["total_train_pixels"]
    grouped["obs3_share"] = grouped["obs3_pixels"] / grouped["total_train_pixels"]
    return grouped.sort_values(["total_train_pixels", "region_id"], ascending=[False, True]).reset_index(
        drop=True
    )


def build_overview(
    tile_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    overlap: pd.DataFrame,
) -> dict[str, object]:
    overall_overlap = overlap.loc[overlap["scope_type"] == "overall"].copy()
    total_train_pixels = int(tile_summary["n_train_pixels"].sum())
    total_hard_pos = int(tile_summary["n_hard_pos"].sum())
    total_hard_neg = int(tile_summary["n_hard_neg"].sum())
    total_ambiguous = int(tile_summary["n_ambiguous"].sum())
    total_seed = int(tile_summary["n_seed"].sum())
    total_seed_pos = int(tile_summary["n_seed_pos"].sum())
    total_seed_neg = int(tile_summary["n_seed_neg"].sum())
    missing_glads2_tiles = tile_summary.loc[tile_summary["missing_glads2"] == 1, "tile_id"].tolist()

    overview = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "n_tiles": int(len(tile_summary)),
        "n_regions": int(len(region_summary)),
        "total_train_pixels": total_train_pixels,
        "total_hard_pos": total_hard_pos,
        "total_hard_neg": total_hard_neg,
        "total_ambiguous": total_ambiguous,
        "total_seed": total_seed,
        "total_seed_pos": total_seed_pos,
        "total_seed_neg": total_seed_neg,
        "hard_pos_rate": (total_hard_pos / total_train_pixels) if total_train_pixels else 0.0,
        "ambiguous_rate": (total_ambiguous / total_train_pixels) if total_train_pixels else 0.0,
        "n_tiles_missing_glads2": int(len(missing_glads2_tiles)),
        "tiles_missing_glads2": missing_glads2_tiles,
        "no_hard_negatives": total_hard_neg == 0,
        "top_overlap_pairs": overall_overlap.sort_values("disagreement_rate", ascending=False)
        .head(3)[["pair", "n_obs_overlap", "disagreement_rate", "jaccard_pos"]]
        .to_dict("records"),
    }
    return overview


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def _fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    parts = [_fmt_row(headers), separator]
    parts.extend(_fmt_row(row) for row in rows)
    return "\n".join(parts)


def format_markdown_report(
    overview: dict[str, object],
    tile_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
) -> str:
    missing_glads2 = overview["tiles_missing_glads2"]
    top_pairs = overview["top_overlap_pairs"]
    top_tiles = tile_summary.sort_values("n_train_pixels", ascending=False).head(16).copy()
    top_tiles = top_tiles[
        [
            "tile_id",
            "fold_id",
            "region_id",
            "n_train_pixels",
            "n_hard_pos",
            "hard_pos_rate",
            "obs2_share",
            "obs3_share",
            "missing_glads2",
        ]
    ]

    region_view = region_summary[
        [
            "region_id",
            "n_tiles",
            "total_train_pixels",
            "total_hard_pos",
            "hard_pos_rate",
            "obs2_share",
            "obs3_share",
            "tiles_missing_glads2",
        ]
    ].copy()

    for frame in (top_tiles, region_view):
        for column in [col for col in frame.columns if col.endswith("_rate") or col.endswith("_share")]:
            frame[column] = frame[column].map(_format_percent)

    handoff_notes = [
        "- Use `train_mask == 1` rows from `pixel_index.parquet` for tabular baselines.",
        "- Use `soft_target` and `sample_weight` for the main baseline; v1 currently has no hard negatives.",
        "- `hard_label == 1` is usable as a conservative positive set.",
        "- `seed_mask == 1` is the precision-oriented seed subset for patch sampling or teacher-style warm starts.",
        "- Missing GLAD-S2 is already encoded as `obs=0` for that source. Do not impute it in v1.",
    ]
    if overview["n_regions"] == overview["n_tiles"]:
        handoff_notes.append(
            "- `region_id` currently falls back to tile-prefix grouping for many tiles, so the region table is coarse and close to tile-level."
        )

    lines = [
        "# Label Quality Report",
        "",
        f"Generated: {overview['generated_at_utc']}",
        "",
        "## Overview",
        f"- Tiles: {overview['n_tiles']}",
        f"- Regions: {overview['n_regions']}",
        f"- Train pixels: {overview['total_train_pixels']:,}",
        f"- Hard positives: {overview['total_hard_pos']:,} ({_format_percent(overview['hard_pos_rate'])})",
        f"- Hard negatives: {overview['total_hard_neg']:,}",
        f"- Ambiguous train pixels: {overview['total_ambiguous']:,} ({_format_percent(overview['ambiguous_rate'])})",
        f"- Seed pixels: {overview['total_seed']:,} "
        f"(pos={overview['total_seed_pos']:,}, neg={overview['total_seed_neg']:,})",
        f"- Tiles missing GLAD-S2: {overview['n_tiles_missing_glads2']}",
        f"- Missing-GLAD-S2 tile ids: {', '.join(missing_glads2) if missing_glads2 else 'none'}",
        "",
        "## Handoff Notes",
    ]
    lines.extend(handoff_notes)
    lines.extend(
        [
            "",
            "## Top Overlap Pairs",
        ]
    )

    for pair in top_pairs:
        lines.append(
            "- "
            f"{pair['pair']}: overlap={int(pair['n_obs_overlap']):,}, "
            f"disagreement={_format_percent(float(pair['disagreement_rate']))}, "
            f"positive_jaccard={_format_percent(float(pair['jaccard_pos']))}"
        )

    lines.extend(
        [
            "",
            "## Tile Summary",
            _dataframe_to_markdown(top_tiles),
            "",
            "## Region Summary",
            _dataframe_to_markdown(region_view),
            "",
        ]
    )

    return "\n".join(lines)


def write_label_quality_report(output_root: str | Path) -> dict[str, object]:
    output_root = Path(output_root)
    paths = resolve_label_quality_paths(output_root)
    manifest, pixel_index, overlap = load_label_artifacts(output_root)
    tile_source_stats = collect_tile_source_stats(manifest)
    tile_summary = build_tile_summary(manifest, pixel_index, tile_source_stats)
    region_summary = build_region_summary(tile_summary)
    overview = build_overview(tile_summary, region_summary, overlap)
    report_md = format_markdown_report(overview, tile_summary, region_summary)

    tile_summary.to_csv(paths.by_tile_csv, index=False)
    region_summary.to_csv(paths.by_region_csv, index=False)
    paths.report_md.write_text(report_md)

    return {
        "overview": overview,
        "tile_summary_path": paths.by_tile_csv,
        "region_summary_path": paths.by_region_csv,
        "report_path": paths.report_md,
    }
