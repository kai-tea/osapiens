# Kangi Model Sandbox

Owner: `kangi`

Use this folder for Kangi-owned model experiments. Keep generated files out of git by writing them to:

```text
artifacts/models/kangi/
```

Starter commands:

```bash
make train_kangi
make predict_kangi
make eval_kangi
```

Replace the starter logic in `train.py`, `predict.py`, and `evaluate.py` with your actual model when ready. Keep shared loaders and output schemas in `models/shared/`.
