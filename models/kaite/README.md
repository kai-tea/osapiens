# Kaite Model Sandbox

Owner: `kaite`

Use this folder for Kaite-owned model experiments. Keep generated files out of git by writing them to:

```text
artifacts/models/kaite/
```

Starter commands:

```bash
make train_kaite
make predict_kaite
make eval_kaite
```

Replace the starter logic in `train.py`, `predict.py`, and `evaluate.py` with your actual model when ready. Keep shared loaders and output schemas in `models/shared/`.
