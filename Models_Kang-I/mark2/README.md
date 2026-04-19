# Mark 2 Embedding-Only Baseline

This folder contains a simple, transparent preprocessing and weak-label pipeline for an embedding-only baseline.

Version 1 intentionally keeps the pipeline conservative:

- input features are only the raw normalized embedding vector per pixel
- normalization statistics are fit on the training split only
- weak labels use a plain 3-source vote
- uncertain pixels are kept as `-1` and excluded through `valid_mask`

## Folder Structure

```text
Models_Kang-I/mark2/
├── inference/
│   └── predict_mlp.py
├── labels/
│   └── weak_labels.py
├── models/
│   └── mlp.py
├── preprocessing/
│   └── embeddings.py
├── training/
│   └── train_mlp.py
├── utils/
│   ├── io.py
│   └── npz_data.py
├── pipeline.py
├── run_pipeline.py
└── README.md
```

## What the Pipeline Produces

For each processed tile, the runner saves a compressed `.npz` artifact containing:

- `features`: normalized per-pixel embedding vectors with shape `(height, width, channels)`
- `labels`: combined weak labels with values `1`, `0`, or `-1`
- `valid_mask`: `labels != -1`

It also saves:

- train-fit normalization statistics in JSON format
- a JSON run summary

## Weak-Label Rule Set

The label combiner uses exactly three sources:

- `RADD`
- `GLAD-L`
- `GLAD-S2`

Rules:

- `1` if at least 2 of the 3 sources indicate deforestation
- `0` only if none of the 3 sources indicates deforestation
- `-1` otherwise

No single-source overrides are used in v1.

## How to Run

From the repository root:

```bash
python3 Models_Kang-I/mark2/run_pipeline.py \
  --data_root data/makeathon-challenge \
  --output_dir Models_Kang-I/mark2/outputs/baseline_v1
```

Optional flags:

- `--year 2023` to force a specific embedding year
- `--val_fraction 0.25` to control the deterministic validation split
- `--include_test` to export normalized test embeddings with the saved train-fit statistics

If `--year` is omitted, the runner selects the latest embedding year available for every train and test tile.

## Output Layout

Example output structure:

```text
Models_Kang-I/mark2/outputs/baseline_v1/
├── stats/
│   └── normalization_year_2025.json
├── train/
├── validation/
├── test/
└── summary.json
```

## Notes

- NaN and infinite embedding values are replaced with `0.0` before statistics and normalization.
- Standardization uses `x_norm = (x - mean) / (std + 1e-6)`.
- The code includes comments marking future extension points for spatial features, weighted fusion, forest-map relabeling, and multimodal fusion.

## Model: Small MLP Baseline

This baseline adds a very small feedforward neural network trained on the preprocessed pixel embeddings.

What it is:

- a compact MLP applied independently to each pixel embedding
- input is the normalized embedding vector per pixel
- output is one deforestation probability per pixel

Why this model:

- it stays simple and easy to inspect
- it is slightly more expressive than logistic regression
- it leaves a clean path toward future multimodal fusion

Training setup:

- training data comes from `Models_Kang-I/mark2/outputs/baseline_v1/train/`
- validation data comes from `Models_Kang-I/mark2/outputs/baseline_v1/validation/`
- each tile is flattened into pixel rows
- only `valid_mask == True` pixels are used
- uncertain weak labels are ignored
- loss is `BCEWithLogitsLoss`
- optimizer is `Adam`

Saved outputs:

- best checkpoint: `Models_Kang-I/mark2/outputs/models/mlp_best.pt`
- training history: `Models_Kang-I/mark2/outputs/models/mlp_history.json`

How to train:

```bash
python3 Models_Kang-I/mark2/training/train_mlp.py
```

How to predict:

```bash
python3 Models_Kang-I/mark2/inference/predict_mlp.py
```

Prediction outputs are saved as `.npz` files with dense probability maps and, if requested, thresholded binary maps.
