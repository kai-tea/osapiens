# Tomy — Deforestation Detection Pipeline

Pixel-level deforestation detection from multimodal satellite data for the
osapiens Makeathon 2026. Takes Sentinel-2 optical, Sentinel-1 radar, and
AlphaEarth Foundations (AEF) pre-trained embeddings for each tile/year,
fuses three noisy weak-label sources (RADD, GLAD-L, GLAD-S2), trains a
per-pixel XGBoost classifier on change-detection features, applies
morphological post-processing and a 2020-forest gate, and emits the
submission GeoJSON via the provided `submission_utils`.

## End-to-end pipeline

```
S2 + S1 + AEF rasters ─┐
                       ├─► per-year feature stack ─┐
2020 forest mask ──────┤                           ├─► XGBoost ─► prob raster
RADD+GLAD-L+GLAD-S2 ───┴─► fused weak labels ──────┘                 │
                                                                     ▼
                                                         forest-gate + threshold
                                                         opening / closing
                                                         connected-component filter
                                                                     │
                                                                     ▼
                                                         pred_{tile}.tif (binary)
                                                                     │
                                                        raster_to_geojson + 0.5 ha filter
                                                                     │
                                                                     ▼
                                                            submission.geojson
```

## Inputs on disk

All data lives under `data/makeathon-challenge/` after `make download_data_from_s3`.

| modality | path pattern | resolution | CRS | role |
|---|---|---|---|---|
| Sentinel-2 | `sentinel-2/{split}/{tile}__s2_l2a/{tile}__s2_l2a_{year}_{month}.tif` | 10 m, monthly, 12 bands | local UTM | optical time series |
| Sentinel-1 | `sentinel-1/{split}/{tile}__s1_rtc/{tile}__s1_rtc_{year}_{month}_{asc\|desc}.tif` | ~30 m, monthly × 2 orbits, VV | local UTM | radar, cloud-proof |
| AEF | `aef-embeddings/{split}/{tile}_{year}.tiff` | annual, 64 dims | EPSG:4326 | pretrained embeddings |
| RADD | `labels/train/radd/radd_{tile}_labels.tif` | — | EPSG:4326 | weak label (radar-based) |
| GLAD-L | `labels/train/gladl/gladl_{tile}_alert{YY}.tif` + `alertDate{YY}.tif` | coarser (~30 m) | EPSG:4326 | weak label (Landsat), yearly |
| GLAD-S2 | `labels/train/glads2/glads2_{tile}_alert.tif` + `alertDate.tif` | — | EPSG:4326 | weak label (S2) |
| metadata | `metadata/{train,test}_tiles.geojson` | — | — | tile footprints |

Ground truth: **no `labels/test/` exists** — the test ground truth is held by
the organisers and only accessible via the leaderboard. Our local comparison
baseline is the fused weak labels on the train tiles.

## Layer-by-layer walkthrough

For each layer: what goes in, what comes out, where in `tomy/` to look, and
why it earns its keep.

### 1. Data loader — `tomy/src/data_loader.py`

- **In**: a `tile_id`, a `year`, and a `split` (`"train"` or `"test"`).
- **Out**: a feature tensor of shape `(C, H, W)` in float32, aligned to the
  canonical UTM grid of the tile (picked from the first available S2 scene).
- **Why**: every modality lives in a different CRS / resolution. This module
  resamples S1 and AEF onto the S2 grid so we can concatenate them
  pixel-for-pixel.
- **Key functions**
  - `list_tiles(split)` — enumerate tiles under `sentinel-2/{split}/`
  - `tile_grid(tile_id, split)` — `TileGrid(transform, crs, shape)` from S2
  - `s2_year_stats(tile_id, year, split)` → `(36, H, W)`: 12 bands × {median, min, max} across months
  - `s1_year_stats(tile_id, year, split)` → `(4, H, W)`: VV (dB) {median, mean, min, max}, reprojected onto tile grid
  - `load_aef_on_tile_grid(tile_id, year, split)` → `(64, H, W)`: AEF reprojected from EPSG:4326 to UTM, NaN-sanitised
  - `per_year_features(tile_id, year)` → `(104, H, W)` = S2 36 + S1 4 + AEF 64
  - `year_over_year_features(tile_id, year, baseline_year=2020)` → `(272, H, W)` = features 2020 ∥ features(year) ∥ AEF delta — this is the input to V1

### 2. Label fusion — `tomy/src/label_fusion.py`

- **In**: the per-tile label rasters in their native EPSG:4326 grids.
- **Out** (all aligned to the tile's UTM grid):
  - `any_alert` uint8 — at least one source fired post-2020 (forest-gated)
  - `confident_pos` uint8 — ≥ `min_sources` agreed post-2020 (training positives)
  - `confident_neg` uint8 — no source fired AND pixel was forest in 2020
  - `soft` float32 — mean confidence of sources that fired
  - `event_days` int32 — earliest post-2020 alert day (days since 1970-01-01)
  - `source_count` uint8 — how many sources agreed
- **Why**: the hidden leaderboard truth is unknown to us; our best training
  signal is "multiple independent weak sources agree". GLAD-L ships per-year
  files, so we merge them first to avoid inflating source counts.
- **Key functions**
  - `decode_radd`, `decode_gladl`, `decode_glads2` — normalise each source to `(alert, date_days, confidence)` at the pixel level
  - `_merge_decoded(items, source)` — collapses multiple years of the same source (currently used by GLAD-L) into a single `DecodedLabels`
  - `load_all_labels(tile_id, split)` → `dict[str, DecodedLabels]`
  - `forest_mask_2020(tile_id, split, ndvi_threshold=0.6)` — 2020 NDVI gate from S2 B4 + B8 monthly medians
  - `fuse_post2020(tile_id, split, gate_by_forest=True, min_sources=2)` — the payload above

### 3. CV split — `tomy/src/cv_split.py`

- **In**: a list of tile IDs (from `list_tiles("train")`).
- **Out**: `splits/mgrs_5fold.json` with five geographically coherent folds.
- **Why**: test tiles mostly live in MGRS zones the training set never saw.
  Grouping tiles by MGRS prefix (e.g. `18NWG`) for CV simulates OOD
  generalisation and prevents leakage between adjacent tiles.
- **Key functions**
  - `mgrs_prefix(tile_id)` — extracts the first MGRS token
  - `split_by_mgrs(tiles, n_folds)` — balanced bin-packing by prefix
  - `fold_iter(tiles, n_folds)` — yields `(fold, train_tiles, val_tiles)`
  - `write_splits(path, n_folds, split)` — persists the split
  - CLI: `.venv/bin/python -m tomy.src.cv_split`

### 4. Evaluation — `tomy/src/evaluate.py`

- **In**: a `predict_fn(tile_id) -> bool array` plus a target grid.
- **Out**: per-tile and aggregate `PixelMetrics` (`precision`, `recall`, `f1`, `iou`) — plus macro-averaged across folds.
- **Why**: the only local quality signal we have; drives all hyperparameter tuning.
- **Key functions**
  - `pixel_metrics(pred, target, mask)` — `tp/fp/fn/tn` counts into a `PixelMetrics`
  - `target_for_tile(tile_id, split, min_sources=2)` — returns the `confident_pos` proxy target
  - `evaluate_fold(predict_fn, tile_ids)` — per-tile and summed metrics
  - `cv_evaluate(predict_fn_factory, tiles, n_folds)` — geographic CV

### 5. V1 baseline — `tomy/src/baseline_v1.py`

- **In**: a list of training tiles, a `BaselineConfig` (years, pos/neg sample budgets, XGB params, `PostprocessConfig`).
- **Out**:
  - a trained classifier (XGBoost, or sklearn LogisticRegression fallback)
  - per-tile probability raster (`predict_proba_max` — max over `analysis_years`)
  - per-tile post-processed **binary** raster (`predict_tile_binary`)
  - `pred_{tile}.tif` GeoTIFFs via `save_prediction_raster`
- **Why**: 272-channel per-pixel classifier is cheap to train (feature build
  dominates wall time) and gets a real leaderboard submission up fast.
- **Key functions / flow**
  - `build_training_set(train_tiles, cfg, split)` — samples `pos_per_tile`/`neg_per_tile` pixels per (tile, year) where event_year matches the analysis year, assembles `(X, y)`
  - `train_model(X, y, cfg)` — XGBClassifier with LogReg fallback
  - `predict_proba_tile(model, tile, year, split)` — per-pixel probabilities for one year
  - `predict_proba_max(model, tile, split, cfg)` — max-reduce across years
  - `predict_tile_binary(model, tile, split, cfg)` — full post-processing chain
  - `save_prediction_raster`, `save_model`, `load_model`
  - `train_and_predict(train_tiles, predict_tiles, ...)` — convenience driver

### 6. Post-processing — `tomy/src/postprocess.py`

- **In**: a probability raster `(H, W) float32`, a forest mask, and a `PostprocessConfig`.
- **Out**: a clean `(H, W) uint8` binary raster ready for vectorisation.
- **Chain**
  1. `threshold` — probability → bool
  2. `binary_opening(kernel=opening_kernel)` — drop salt noise
  3. `binary_closing(kernel=closing_kernel)` — fill pinholes in real patches
  4. `filter_components_by_size(min_component_px)` — 8-connectivity, drop small blobs
  5. AND with `forest_mask_2020` — enforce the challenge's "forest in 2020" rule
- **Why**: pixel classifiers are inherently noisy; IoU-style scoring
  rewards contiguous patches, so cleaning up before vectorisation matters.
- **Key functions**
  - `postprocess(prob, forest_mask, config)` — the full chain
  - `filter_components_by_size(binary, min_px)` — uses `scipy.ndimage.label`
  - `best_threshold(prob, target, mask, candidates, metric)` — single-tile threshold sweep
  - `best_threshold_aggregate([(prob, target, mask), ...])` — micro-averaged sweep across tiles

### 7. Submission wiring — `tomy/scripts/make_submission.py`

- **In**: a directory of `pred_{tile_id}.tif` rasters (from V1 inference).
- **Out**:
  - `{out_dir}/{tile_id}.geojson` — per-tile FeatureCollection in EPSG:4326
  - `{out_dir}/submission.geojson` — combined FeatureCollection across tiles (each feature tagged with `tile_id`)
  - `{out_dir}/submission_summary.json` — per-tile polygon counts + any skips
- **Why**: closes the loop with the existing `submission_utils.raster_to_geojson` (vectorisation, UTM → EPSG:4326 reprojection, 0.5 ha minimum area filter).

## Scripts & CLI entry points

All scripts are invoked with `.venv/bin/python -m tomy.scripts.<name>`.

| script | purpose |
|---|---|
| `smoke_test.py` | End-to-end alignment + fusion sanity check on one tile. Auto-picks the latest year with both S2 and S1 coverage. |
| `eda_labels.py` | Per-source and cross-source positive-rate stats per train tile. Writes `reports/label_eda.json`. |
| `train_v1.py --fold {N\|all}` | Train on fold `N`'s training tiles, validate on held-out tiles, sweep 33 thresholds between 0.1–0.9, report best IoU/F1. Saves model to `models/v1_fold{N}.joblib`, metrics to `reports/v1_fold{N}_metrics.json`. `--fold all` trains on all 16 train tiles for the final submission model. |
| `predict_test.py --model ... [--threshold-file ...]` | Run a trained model on the 5 test tiles, apply the full post-processing chain, write `pred_{tile}.tif` rasters. |
| `make_submission.py --pred-dir ...` | Convert prediction rasters → per-tile + combined GeoJSON. |
| `wait_and_bootstrap.sh` | Polls `data/makeathon-challenge/` until S2+S1+AEF+labels+metadata are all present, then fires the smoke test + CV split. |

### End-to-end command chain

```bash
# one-time setup
make install                            # builds .venv, installs requirements
make download_data_from_s3              # ~43 GB download

# sanity
.venv/bin/python -m tomy.scripts.smoke_test
.venv/bin/python -m tomy.src.cv_split   # writes splits/mgrs_5fold.json

# per-fold training + threshold tuning (gives CV IoU/F1)
for fold in 0 1 2 3 4; do
  .venv/bin/python -m tomy.scripts.train_v1 --fold $fold
done

# final model on all 16 train tiles
.venv/bin/python -m tomy.scripts.train_v1 --fold all

# test-tile inference using chosen threshold from whichever fold scored best
.venv/bin/python -m tomy.scripts.predict_test \
    --model models/v1_foldall.joblib \
    --threshold-file reports/v1_fold0_metrics.json \
    --out-dir predictions/v1_test

# convert to leaderboard-ready GeoJSON
.venv/bin/python -m tomy.scripts.make_submission \
    --pred-dir predictions/v1_test \
    --out-dir submission/v1
# → upload submission/v1/submission.geojson to the leaderboard
```

## Key findings from label EDA

From `reports/label_eda.json` (computed by `eda_labels.py`):

- **Coverage**: 16 train tiles have RADD + GLAD-L; only **8 also have GLAD-S2** (the other 8 can reach ≤ 2 sources).
- **Per-source post-2020 positive rate (mean across tiles)**: RADD 13.7%, GLAD-S2 17.5% (where present), GLAD-L 8.1%. Union of sources: **18.8%**.
- **Agreement**: RADD ∩ GLAD-S2 = 4.8%, RADD ∩ GLAD-L = 5.7%, GLAD-S2 ∩ GLAD-L = 3.3%.
- **`confident_pos` (≥ 2 sources)**: **9.7%** — rich training signal.
- **`confident_pos` (≥ 3 sources)**: 2.1%, and 0% on the 8 tiles without GLAD-S2 — unusable as a universal target.
- **Decision**: `min_sources=2` is the right default.
- **Tile heterogeneity**: per-tile "any alert" ranges from 1.3% to 54%. The problem is wildly region-dependent.
- **Test-set geography**: 5 tiles. Only `18NYH` shares an MGRS zone with training; the other 4 (`18NVJ`, `33NTE`, `47QMA`, `48PWA`) are in zones the training set never saw — genuine OOD.

## Known caveats & open decisions

- **Post-2020 cutoff** is currently `>= 2020-01-01`. The challenge text says "events after 2020" — arguably strictly 2021+. About 20 % of current positives carry calendar-2020 dates; the 2020 NDVI forest gate drops most but not all of them. Worth tightening to `>= 2021-01-01` if the jury reads "after 2020" strictly.
- **V2 U-Net** deferred until V1 is on the leaderboard. Spatial context will likely give +5–10 % IoU; V1 is intentionally small so it ships.
- **Threshold tuning** is per-fold via `best_threshold_aggregate`. Per-region calibration remains for V2.
- **Submission metric** isn't stated explicitly in the repo. Pixel-level IoU is the most likely scoring method for binary-segmentation polygon submissions — that's what our CV already optimises.
- **Two `__init__.py` files** — `tomy/__init__.py` and `tomy/src/__init__.py`. Python 3.3+ accepts namespace packages so one is redundant. Deferred.
- **Commit history** — the two local `"make me win, make no mistakes"` commits should be collapsed into a single descriptive commit before the branch ships. Deferred.

## Repo layout snapshot

```
tomy/
├── __init__.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── label_fusion.py
│   ├── cv_split.py
│   ├── evaluate.py
│   ├── baseline_v1.py
│   └── postprocess.py
└── scripts/
    ├── smoke_test.py
    ├── eda_labels.py
    ├── train_v1.py
    ├── predict_test.py
    ├── make_submission.py
    └── wait_and_bootstrap.sh
```

Supporting top-level artefacts: `Makefile` (python3.12 venv + data download),
`requirements.txt`, `submission_utils.py` (provided — converts binary raster
to GeoJSON), `download_data.py` (provided — S3 fetcher), `challenge.ipynb`
(provided — dataset walkthrough), and the `reports/`, `splits/`, `models/`,
`predictions/`, `submission/` output directories created on demand.
