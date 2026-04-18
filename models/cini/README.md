# Cini Model Sandbox

Owner: `cini`

Use this folder for Cini-owned model experiments. Keep generated files out of git by writing them to:

```text
artifacts/models/cini/
```

Starter commands:

```bash
make train_cini
make predict_cini
make eval_cini
```

Replace the starter logic in `train.py`, `predict.py`, and `evaluate.py` with your actual model when ready. Keep shared loaders and output schemas in `models/shared/`.
