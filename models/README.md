# Team Model Sandboxes

Use this folder for model experiments. Ownership is by person, not by model type, so each teammate can work without creating merge conflicts.

```text
models/
  shared/   shared loaders, split helpers, schemas, and metrics
  cini/     Cini-owned experiments
  kaite/    Kaite-owned experiments
  kangi/    Kangi-owned experiments
  tomy/     Tomy-owned experiments
```

## Rules

- Edit your own folder freely.
- Edit `models/shared/` carefully because it affects everyone.
- Do not write checkpoints, predictions, or logs into git-tracked folders.
- Write generated outputs to `artifacts/models/<owner>/`.

## Shared Inputs

All model folders should use the same data contract:

```text
artifacts/model_inputs_v1/tile_manifest.csv
artifacts/model_inputs_v1/file_manifest.csv
artifacts/labels_v1/manifest.parquet
artifacts/labels_v1/pixel_index.parquet
cini/splits/split_v1/fold_assignments.csv
```

## Shared Prediction Schemas

Validation predictions:

```text
tile_id,row,col,y_true,score,fold_id,model_name
```

Test predictions:

```text
tile_id,row,col,score,model_name
```

## Commands

```bash
make train_cini
make predict_cini
make eval_cini

make train_kaite
make predict_kaite
make eval_kaite

make train_kangi
make predict_kangi
make eval_kangi

make train_tomy
make predict_tomy
make eval_tomy
```

The starter scripts only validate the shared contract and create templates/summaries. Replace each personal folder's scripts with real model logic as needed.
