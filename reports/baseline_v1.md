# Baseline v1 — handoff to Kaite

_Run of 2026-04-18; code at `feature/kaite` HEAD. Headline numbers below are cross-validated on Cini's fold split but optimistic (see "Known weaknesses"); polygon-level leaderboard score will be lower._

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

## Validation numbers (3-fold CV, 16 train tiles)

| fold | val rows | positives | threshold | F1 | IoU | PR-AUC | best_iter |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 311,985 | 248,091 | 0.326 | 0.973 | 0.948 | 0.992 | 2 |
| 1 | 500,000 | 240,139 | 0.315 | 0.956 | 0.916 | 0.988 | 152 |
| 2 | 406,902 | 214,243 | 0.663 | 0.974 | 0.950 | 0.996 | — |
| **mean** |  |  | **0.435** | **0.968** | **0.938** | **0.992** |  |

Top-20 feature importances are **all AEF embeddings** — `aef_e12_baseline`
dominates (gain 3.7M), followed by `aef_e52_latest`, `aef_e36_baseline`
and various `aef_*_delta` features. Neither S1 nor any S2 band temporal
stat appears in the top 20.

### Per-MGRS-prefix F1 breakdown (regional gap)

| strongest 4 | weakest 4 |
|---|---|
| 48PWV 0.998 | 18NXJ 0.561 |
| 48QWD 0.993 | 18NWH 0.587 |
| 48PYB 0.993 | 18NWJ 0.684 |
| 48PXC 0.993 | 19NBD 0.814 |

**Regional generalisation gap** = 48PWV (0.998) − 18NXJ (0.561) = **0.437 F1**.
Pathological near-zero regions (`18NWM` 1,092 rows, `18NYH` 16 rows) are
thin tiles whose fold F1 is statistical noise.

### Test-set submission (mean threshold = 0.4348)

| tile | region | positive fraction | polygons | note |
|---|---|---:|---:|---|
| 18NVJ_1_6 | 18N (Amazon) | 0.03% | 1 | near-zero, under-predict |
| 18NYH_2_1 | 18N (Amazon) | 11.1% | 187 | plausible |
| 33NTE_5_1 | 33N (Africa) | 23.7% | 130 | plausible (out-of-training region) |
| 47QMA_6_2 | 47Q (SE Asia) | **99.3%** | 5 | **catastrophic over-predict** |
| 48PWA_0_6 | 48P (SE Asia) | **99.6%** | 1 | **catastrophic over-predict** |

See `reports/baseline_results.md` for the full confusion matrices and
training config.

## Known weaknesses

- **Optimistic CV F1**: 0.97 is not a leaderboard-equivalent number.
  `train_mask == 1` restricts eval to pixels where *some* source
  observed the land with `obs_count ≥ 2`, which heavily biases toward
  the alert frontier. The classifier is not being asked to score empty
  background. Expect polygon IoU at submission time to be materially
  lower.
- **Catastrophic regional calibration failure on SE Asia** — the
  headline issue. Test tiles `47QMA_6_2` and `48PWA_0_6` predict
  99.3% and 99.6% positive. Root cause: training positive rate in
  47Q/48P tiles is very high (those tiles miss GLAD-S2, so
  `soft_target ≥ 0.5` fires whenever RADD+GLAD-L agree, which is
  frequent in actively-deforested areas). The mean threshold 0.435 is
  then too low for the test tiles' wider geographic distribution.
  Fixing this is the **#1 priority for v2** — see "What to try next".
- **Zero hard negatives**: Cini's synthesis only produces `hard_label`
  on pixels with `obs_count == 3` and zero source alerts — impossible
  for the 8 MGRS-47/48 SE-Asia tiles that lack GLAD-S2 coverage. Hence
  the switch to soft-target binarisation; `soft_target == 0` is a
  genuine weak negative (`obs == 1 ∧ alert == 0` emitted by at least
  one source).
- **Regional generalisation gap F1 = 0.44** (48PWV 0.998 vs 18NXJ
  0.561). This shows up in validation and is consistent with the
  test-set blow-up.
- **Threshold spread across folds**: 0.315–0.663, factor 2.1×. Median
  (0.326) might be more robust than the mean (0.435) we ship —
  switching keys is a one-line CLI flag in `src/predict.py`.
- **Small training tiles are thin**: `18NWM_9_4` has 1,092 `train_mask`
  pixels, `18NYH_9_9` has 16 — fold metrics on those tiles are noisy
  by construction (both score 0.000).
- **AEF dominates the model**: all top-20 features are AEF embeddings;
  no S1 or S2 temporal stats survive. The foundation model has
  internalised the signal we were engineering by hand.
- **No cloud masking for S2**: nodata pixels are set to NaN
  (LightGBM handles natively) but partly-cloudy scenes are not
  explicitly filtered.
- **Parallel-predict summary.json clobbering**: when `src/predict.py`
  is run in multiple parallel processes (each on a subset of test
  tiles) the last writer wins. Sequentially invoking it with all test
  tiles avoids this; the shipped `submission/baseline_v1/summary.json`
  was reconstructed manually from the worker logs.

## What to try next (ordered by expected leaderboard impact)

1. **Fix the SE-Asia over-prediction — per-region threshold
   calibration**. The 99.3% / 99.6% positive fractions on `47QMA_6_2`
   and `48PWA_0_6` are the single biggest leaderboard risk. Fit one
   threshold per MGRS-prefix group on each held-out fold's positives,
   store as `baseline_results_thresholds.json` → `per_region`, and
   route `src/predict.py` by tile's `region_id`. Fallback to the
   current mean threshold for unseen prefixes (e.g. 33N).
2. **Drop `train_mask` restriction during eval**. The high CV F1 is
   an artefact of only scoring on the alert frontier. Re-run eval
   against a full-tile held-out raster (predict everywhere, compute
   IoU vs `hard_label == 1` after excluding `hard_label == 255`) to
   get a number that tracks the leaderboard.
3. **Regress on `soft_target` with `LGBMRegressor`** instead of the
   0.5 binarisation. Uses all of Cini's confidence information and
   avoids the single-threshold-at-training-time lossiness. Evaluate
   against both the soft target (MSE) and the hard-0.5 binarisation
   (F1).
4. **Time-of-event head**: the `event_year` sidecar is already in the
   parquet. A second `LGBMRegressor` against positive-pixel `event_year`
   unlocks the "predict *when* deforestation occurred" bonus for
   minimal additional training cost; preserving the column in
   `data.py` was intentional for this.
5. **Prune S1 + S2 temporal features**. Top-20 importances use zero of
   them; dropping all 210 S1+S2 features (keep just the 192 AEF dims)
   should cut training time ~5× and memory ~2× without hurting
   accuracy. Fast to ablate.
6. **Visualisation tool**: a small Folium or Streamlit viewer of
   (S2 RGB, AEF PCA, prediction, weak-label overlay) per test tile
   would directly feed the presentation rubric axis. Cheap bonus points.

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
