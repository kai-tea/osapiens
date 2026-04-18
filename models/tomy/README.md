# Tomy Model Sandbox

Owner: `tomy`

Use this folder for Tomy-owned model experiments. Keep generated files out of git by writing them to:

```text
artifacts/models/tomy/
```

Starter commands:

```bash
make train_tomy
make predict_tomy
make eval_tomy
```

Replace the starter logic in `train.py`, `predict.py`, and `evaluate.py` with your actual model when ready. Keep shared loaders and output schemas in `models/shared/`.
