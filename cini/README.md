# Cini Handoff

This folder contains the Person A data, validation, and weak-label pipeline work.
Generated data stays outside this folder:

- Downloaded challenge data: `data/`
- Generated label artifacts: `artifacts/labels_v1/`
- Generated raw-input handoff: `artifacts/model_inputs_v1/`

Those folders are intentionally ignored by git because they are large or reproducible.

## Folder Contents

- `label_pipeline/`: implementation for weak-label decoding, labelpack construction, split loading, reports, and raw model-input manifests.
- `scripts/`: runnable helper scripts for split creation, label quality reports, and model-input handoff generation.
- `tests/`: unit tests for decoders, label synthesis, split loading, reporting, and model-input manifests.
- `splits/split_v1/`: frozen fold assignment files. These are meant to be committed and shared.

The repo root still has a tiny `label_pipeline/` compatibility wrapper. That means existing imports and commands continue to work:

```bash
PYTHONPATH=. .venv/bin/python3 -m label_pipeline --help
```

## Main Artifacts For ML Team

- `artifacts/labels_v1/manifest.parquet`: one row per train tile labelpack.
- `artifacts/labels_v1/pixel_index.parquet`: trainable S2-grid pixels where `train_mask == 1`.
- `artifacts/labels_v1/source_overlap.csv`: weak-source overlap/disagreement summary.
- `artifacts/model_inputs_v1/tile_manifest.csv`: one row per train/test tile with modality paths and completeness.
- `artifacts/model_inputs_v1/file_manifest.csv`: one row per S1/S2/AEF file with sequence order.
- `artifacts/model_inputs_v1/DATA_HANDOFF.md`: loader contract for B/C.

## Current Completeness

All expected raw model-input files are present and size-verified locally:

```text
train:
  S1:  2012 / 2012 files
  S2:  1150 / 1150 files
  AEF:   96 /   96 files

test:
  S1:   563 /  563 files
  S2:   343 /  343 files
  AEF:   30 /   30 files

tiles:
  train: 16 / 16 complete
  test:   5 / 5 complete
```

## Label Contract

Each train tile has one 18-band labelpack:

```text
artifacts/labels_v1/tiles/{tile_id}_labelpack.tif
```

Band order:

```text
1  train_mask
2  hard_label
3  seed_mask
4  soft_target
5  sample_weight
6  obs_count
7  radd_obs
8  radd_alert
9  radd_conf
10 radd_days_since_2020
11 gladl_obs
12 gladl_alert
13 gladl_conf
14 gladl_days_since_2020
15 glads2_obs
16 glads2_alert
17 glads2_conf
18 glads2_days_since_2020
```

Recommended v1 training setup:

- Use pixels where `train_mask == 1`.
- Prefer `soft_target` with `sample_weight`.
- Treat `hard_label == 255` as ambiguous, not negative.
- Use `seed_mask == 1` only for high-confidence seed experiments.

## Split Contract

Use `cini/splits/split_v1/fold_assignments.csv`.

`fold_id` means the tile's held-out validation fold:

```python
active_fold = 0

train_tiles = tile_manifest[
    (tile_manifest["split"] == "train")
    & (tile_manifest["fold_id"] != active_fold)
]

val_tiles = tile_manifest[
    (tile_manifest["split"] == "train")
    & (tile_manifest["fold_id"] == active_fold)
]
```

## Reproduction Commands

Run tests:

```bash
PYTHONPATH=. .venv/bin/python3 -m unittest discover -s cini/tests
```

Regenerate labels:

```bash
PYTHONPATH=. .venv/bin/python3 -m label_pipeline \
  --data-root data/makeathon-challenge \
  --split-dir cini/splits/split_v1 \
  --output-root artifacts/labels_v1 \
  --force
```

Regenerate model-input handoff:

```bash
make prepare_model_inputs_v1
```
