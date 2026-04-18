# Baseline v1 — handoff to Kaite

> **DO NOT SUBMIT `submission/baseline_v1/`.** Two of five test tiles
> (`47QMA_6_2` at 99.3%, `48PWA_0_6` at 99.6% positive) are broken in
> ways that will tank any metric that penalises false positives. The
> pipeline, eval plumbing, and GeoJSON outputs are structurally correct
> — the model is not production-usable. v2 is required before any
> leaderboard upload.

_Run of 2026-04-18; code at `feature/kaite` HEAD._

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

- **The CV metric measures the wrong task**. F1=0.968 answers
  "given this pixel was labelable (weak sources agreed enough to gate
  it through `train_mask == 1`), did we call it deforestation?". The
  leaderboard asks "for every pixel in a 1000×1000 tile, is this
  deforestation?". Those are different problems. The classifier was
  implicitly trained on the gated task and blows up when asked to score
  ungated pixels — that's precisely what the test-tile failure shows.
  Honest eval requires scoring over all pixels with ungated labels;
  see v2 priority #2.
- **SE-Asia test tiles are broken, not miscalibrated**. `47QMA_6_2`
  99.3%, `48PWA_0_6` 99.6% positive. A forest tile cannot be 99%
  deforested in five years. This is not a threshold issue — the model
  is extrapolating into an embedding region it was never trained on
  and picking "positive" as the safe bet. See v2 priority #1.
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
- **AEF dominance on CV is a trap, not a win**. Top-20 features are
  all AEF embeddings — but that's importance on the *gated* CV, where
  AEF has the easy signal. On the OOD test tiles AEF is precisely what
  extrapolates wildly (its embedding manifold is region-specific and
  the test distribution isn't on it). S1 and S2 temporal change
  signals are physically grounded and region-agnostic; they are the
  best candidate to save generalisation, not dead weight. **Do not
  prune them.**
- **No cloud masking for S2**: nodata pixels are set to NaN
  (LightGBM handles natively) but partly-cloudy scenes are not
  explicitly filtered.
- **Parallel-predict summary.json clobbering**: when `src/predict.py`
  is run in multiple parallel processes (each on a subset of test
  tiles) the last writer wins. Sequentially invoking it with all test
  tiles avoids this; the shipped `submission/baseline_v1/summary.json`
  was reconstructed manually from the worker logs.

## v2 priorities — reviewed order

1. **Forest-in-2020 gate at inference.** Hard mask: any pixel that
   wasn't forest in 2020 gets `prediction = 0`. Use S2 NDVI in 2020
   with a threshold (fast, in-repo), or a public 2020 forest-cover
   product (Hansen GFC, ESA WorldCover). This alone probably fixes
   `47QMA_6_2` and `48PWA_0_6` because most of those "positive"
   pixels aren't even forest.
2. **Honest eval without `train_mask` gating.** Re-run `src/eval.py`
   with predictions over the full held-out raster; label any pixel
   outside `train_mask` as `0` (true-negative for scoring). The
   resulting F1 is the number to trust — expect it materially below
   0.968.
3. **Keep S1 and S2 features. Reconsider AEF.** Current importance
   ranking inverts what generalises. S1 VV / S2 temporal stats are
   physically grounded change signals; AEF is a learned embedding
   whose manifold doesn't cover the test distribution. Ablate by
   training (a) AEF-only, (b) S1+S2-only, (c) both, and compare on
   the *ungated* eval from step 2. Consider per-tile AEF
   normalisation to neutralise manifold shift.
4. **Submission sanity-check / refusal rule.** Reject any test tile
   whose `positive_fraction` falls outside `[0.0001, 0.10]` pending
   manual inspection. Deforestation rates in a single 5-year window
   rarely exceed ~10% of a tile's area. Cheap to add to `predict.py`.
5. **Visual QA loop.** Before any further submission, render each
   test tile as Sentinel-2 RGB + prediction overlay (Folium,
   Streamlit, or static PNGs). The two broken tiles here are obvious
   from a glance; don't ship what hasn't been looked at. Doubles as
   presentation-rubric material.
6. **Per-region stratified training.** If training has 2 S-America
   tiles for every SE-Asia tile, the model's prior is South America.
   Sample-weight per MGRS prefix (e.g. inverse region frequency) so
   each region contributes equally.
7. **Per-region threshold calibration.** Still useful, but *only
   after* steps 1–5. On its own it can't rescue a model that's
   extrapolating.
8. **Time-of-event bonus head.** The `event_year` sidecar is already
   in the parquet. A second `LGBMRegressor` against positive-pixel
   `event_year` unlocks the "predict *when* deforestation occurred"
   bonus once the main model is trustworthy.

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
