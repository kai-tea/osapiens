# Kaite v1 — LightGBM baseline checkpoint

Trained model from the v1 LightGBM pipeline on `feature/kaite`. See `reports/baseline_v1.md` for the full post-mortem.

## What's here

- `baseline_v1.lgb` — LightGBM booster, saved via `booster.save_model(...)` by `src/train.py`.
- `baseline_v1.json` — config, feature list (402 names), git SHA, per-split sizes.

## Known caveats (do not submit as-is)

- Two of five test tiles blow up at 99% positive (47QMA_6_2, 48PWA_0_6) — see baseline_v1.md.
- Lost to naive `majority_2of3` and `hansen_lossyear_post2020` on ungated CV. Current ship candidate is `submit_heuristic` (Hansen GFC loss), not this model.

## Continuing training (warm-start + more rounds)

Use `tools/continue_train_lgbm.py` (see `remote/continue_kaite.sh` for the droplet-orchestrated version). LightGBM is CPU-only — MI300X's GPU sits idle for this, but a DO droplet with many CPU cores still trains faster than a laptop.
