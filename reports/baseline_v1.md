# Baseline v1 — handoff to Kaite

_Final numbers land here after the full 16-tile run finishes; this file is filled in by `src/eval.py` and `src/predict.py`. Hand-curated narrative only._

## What was built

A laptop-CPU pipeline that turns Cini's consolidated weak labels plus the
raw Sentinel-1 / Sentinel-2 / AEF imagery into a per-pixel deforestation
score, with a GeoJSON submission as the final artifact.

Four stages, each runnable in isolation:

1. **`src/data.py`** — per-tile feature extraction on the S2 UTM grid.
   Builds 402 features per pixel:
   - S1 VV per orbit (ascending, descending): 2 streams × {mean, std,
     min, max, linear-trend} × {baseline=2020, change=2021-2025,
     delta=change-baseline} = 30 features
   - S2 per band (B01-B12): 12 × 5 × 3 = 180
   - AEF 64-dim embedding: 64 × {baseline=2020, latest, delta} = 192
   Samples up to `--max-pixels-per-tile` (default 100k) uniformly from
   `train_mask == 1` pixels; writes parquet under
   `artifacts/features_v1/`.
2. **`src/train.py`** — LightGBM `LGBMClassifier` on
   `y = (soft_target >= 0.5).astype(int)` weighted by Cini's
   `sample_weight`. Internal early-stopping split holds out whole tiles
   to prevent spatial leakage. Saves booster + JSON metadata (config,
   feature list, git SHA).
3. **`src/eval.py`** — 3-fold CV using Cini's
   `cini/splits/split_v1/fold_assignments.csv` (MGRS-prefix grouped).
   Per-fold and aggregate precision / recall / F1 / IoU / PR-AUC at
   F1-optimal threshold; per-MGRS F1 breakdown + best-worst gap.
   Writes `reports/baseline_results.md` and a
   `baseline_results_thresholds.json` sidecar consumed by predict.
4. **`src/predict.py`** — retrains on all train tiles, pulls mean
   threshold from eval sidecar, predicts full-raster features for the
   5 test tiles, binarises, and calls `submission_utils.raster_to_geojson`
   with `min_area_ha=0.5`.

## How Cini's labels are consumed

- File path: `artifacts/labels_v1/tiles/{tile_id}_labelpack.tif`
  (18-band float32 raster on the S2 UTM grid, `nodata=-9999.0`).
- Bands read: `train_mask` (band 1), `soft_target` (4),
  `sample_weight` (5), `hard_label` (2),
  `{radd,gladl,glads2}_days_since_2020` (10, 14, 18).
- Training eligibility: `train_mask == 1` — Cini's pre-filter already
  bakes in `obs_count ≥ 2` and the post-2020 date window; no re-derived
  forest-2020 mask needed.
- Training target: `soft_target >= 0.5` → binary `y`, per Cini's handoff
  recommendation. Hard labels are unusable directly (0 hard negatives
  across the training set).
- Optional bonus sidecar: per-pixel `event_year` = earliest positive
  year across the three `*_days_since_2020` bands, stored in the
  feature parquet but not consumed by v1.

## Validation numbers

_Populated automatically from `reports/baseline_results.md`; see that
file for the per-fold table, per-MGRS breakdown, confusion matrices,
top-20 feature importances, and training config._

## Known weaknesses

- **Zero hard negatives**: Cini's synthesis only produces `hard_label`
  on pixels with `obs_count == 3` and zero source alerts — impossible
  for the 8 MGRS-47/48 SE-Asia tiles that lack GLAD-S2 coverage. Hence
  the switch to soft-target binarisation; precision on those tiles
  depends on whether `soft_target == 0` is a true "forest stayed
  forest" signal there (it is — `obs == 1 ∧ alert == 0` is emitted by
  both RADD and GLAD-L independently).
- **Threshold spread across folds**: typically 3-4× between min and max.
  Downstream submission uses the mean; median would be more robust
  against a single pathological fold. Both values logged so Kaite can
  override.
- **Small training tiles are thin**: `18NWM_9_4` has 1,092 `train_mask`
  pixels, `18NYH_9_9` has 16 — fold metrics on those tiles are noisy
  by construction.
- **AEF dominates the model**: top feature importances are almost
  entirely AEF delta embeddings. S1 and S2 temporal stats contribute
  little marginal gain — signal that the foundation model has already
  internalised the information we're engineering by hand.
- **No cloud masking for S2**: nodata pixels are set to NaN
  (LightGBM handles natively) but partly-cloudy scenes are not
  explicitly filtered.
- **Stratified sampling at feature-extraction time was the default in
  v0** — switched to uniform before the final run so reported F1
  reflects per-tile positive rates.

## What to try next

1. **Label-fusion ablation**: re-run with `y = (soft_target >= 0.34)`
   or directly regress on `soft_target` with `LGBMRegressor`. Whichever
   improves IoU/PR-AUC without inflating the regional gap wins.
2. **Drop redundant S2 features**: the top-20 feature table suggests
   most S2 temporal stats are noise. Pruning to just baseline NDVI/NDMI
   + deltas could speed up training 4-5× without hurting accuracy.
3. **Per-region threshold calibration**: the submission uses a single
   mean threshold for all 5 test tiles. Fitting one threshold per MGRS
   prefix on the held-out fold's positives would likely tighten
   precision on the Amazon tiles while keeping recall on SE Asia.
4. **Time-of-event head**: the `event_year` sidecar is already in the
   parquet — a second LightGBM regression against the positive-pixel
   subset would unlock the "predict *when* deforestation occurred"
   bonus for minimal additional training cost.
5. **Visualisation tool**: a small Folium or Streamlit viewer of
   (S2 RGB, AEF PCA, prediction, weak-label overlay) per test tile
   would directly feed the presentation rubric axis.

## Reproduction

```bash
# 1. Generate Cini's label + model-input artifacts (see her README).
# 2. Extract features for all tiles:
PYTHONPATH=. .venv/bin/python3 -m src.data --splits train test --max-pixels-per-tile 100000
# 3. Cross-validate + write report:
PYTHONPATH=. .venv/bin/python3 -m src.eval
# 4. Train final model on all 16 train tiles:
PYTHONPATH=. .venv/bin/python3 -m src.train
# 5. Produce submission GeoJSONs:
PYTHONPATH=. .venv/bin/python3 -m src.predict
```

Fix all seeds at 42; config is recorded in `reports/baseline_results.md`
alongside the git SHA.
