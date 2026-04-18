# Cini

Placeholder file for Cini.

L Team Handoff Summary
I implemented the full Person A data + label handoff. You now have a complete, reproducible weak-label package plus a complete raw-input manifest for S1, S2, and AEF across train and test tiles.

Main entry points:

Labels: artifacts/labels_v1
Model input handoff: artifacts/model_inputs_v1
Frozen split: splits/split_v1
Implementation package: label_pipeline
What Was Implemented

Weak-label decoding for RADD, GLAD-L, and GLAD-S2.
Conservative label synthesis into 18-band labelpack GeoTIFFs on the Sentinel-2 grid.
Frozen geo/tile split with fold_id and region_id.
Dense labelpacks for all 16 train tiles.
manifest.parquet, pixel_index.parquet, and source_overlap.csv.
Full S1/S2/AEF train+test download verification.
tile_manifest.csv and file_manifest.csv for model loaders.
DATA_HANDOFF.md for loader contract.
Unit tests for decoders, synthesis, splits, reports, and model-input manifests.
Data Completeness
All expected model inputs are present locally and size-verified against S3.

train:
  S1:  2012 / 2012 files
  S2:  1150 / 1150 files
  AEF:   96 /   96 files

test:
  S1:   563 /  563 files
  S2:   343 /  343 files
  AEF:   30 /   30 files

tiles:
  train: 16 / 16 complete
  test:   5 / 5 complete
How To Use The Data
Use tile_manifest.csv as the main table. It has one row per tile and tells you where the raw inputs and labels are.

Important columns:

split: train or test
tile_id: challenge tile id
fold_id: held-out validation fold for train tiles
region_id: coarse region id, mostly tile-prefix based
labelpack_path: path to the train labelpack, empty for test
s2_ref_path: canonical reference raster grid
s1_dir_path: Sentinel-1 tile directory
s2_dir_path: Sentinel-2 tile directory
aef_first_path: first AEF embedding file
*_complete: modality completeness flags
all_modalities_complete: 1 for every tile now
Fold usage:

active_fold = 0
train_tiles = tile_manifest[
    (tile_manifest["split"] == "train") &
    (tile_manifest["fold_id"] != active_fold)
]

val_tiles = tile_manifest[
    (tile_manifest["split"] == "train") &
    (tile_manifest["fold_id"] == active_fold)
]

test_tiles = tile_manifest[tile_manifest["split"] == "test"]
Use file_manifest.csv when you need exact temporal ordering. It has one row per raw file.

Important columns:

modality: s1, s2, or aef
year, month, orbit
sequence_order: deterministic order for time-series loading
local_path: repo-relative local path
exists_local: verified complete local file
is_reference_s2: 1 for the canonical S2 raster per tile
Ordering contract:

S2 order: (year, month)
S1 order: (year, month, orbit), with ascending before descending
AEF order: year
Canonical Grid
The canonical grid is Sentinel-2.

For every tile:

Load s2_ref_path.
Load labels directly from the labelpack; they are already on the S2 grid.
Reproject/resample S1 and AEF onto the S2 grid inside your loader if needed.
Do not compare raw S1/S2/AEF pixel positions unless they are aligned to s2_ref_path.
Training Labels
Each train tile has one 18-band labelpack:

artifacts/labels_v1/tiles/{tile_id}_labelpack.tif
Band order:

1  train_mask
2  hard_label
3  seed_mask
4  soft_target
5  sample_weight
6  obs_count
7  radd_obs
8  radd_alert
9  radd_conf
10 radd_days_since_2020
11 gladl_obs
12 gladl_alert
13 gladl_conf
14 gladl_days_since_2020
15 glads2_obs
16 glads2_alert
17 glads2_conf
18 glads2_days_since_2020
Recommended training target for v1:

Use train_mask == 1.
Prefer soft_target with sample_weight.
Treat hard_label == 255 as ambiguous, not negative.
Use seed_mask == 1 only for high-confidence seed experiments.
Do not assume hard negatives are abundant; this dataset/label strategy is intentionally conservative.
Label meaning:

train_mask = 1: at least two weak sources observed the pixel.
soft_target: confidence-weighted weak deforestation target in [0, 1].
sample_weight: agreement-based weight.
hard_label = 1: at least two strong positive weak-source votes.
hard_label = 0: all three sources observed and no source alerted.
hard_label = 255: ambiguous.
obs_count: number of weak sources covering the pixel.
*_days_since_2020: alert timing, -1 if no valid post-2020 alert.
Fast tabular baseline path:
Use pixel_index.parquet. It stores only train_mask == 1 pixels with S2-grid row/col indices.

Columns:

tile_id, row, col, region_id, fold_id, hard_label, seed_mask,
soft_target, sample_weight, obs_count, radd_alert, gladl_alert, glads2_alert
Weak-Label Caveats
The label package models noisy supervision; it does not treat any weak source as ground truth.

Current caveats:

GLAD-S2 coverage is missing for some train tiles; this is encoded via glads2_obs == 0.
hard_label == 0 is deliberately strict, so hard negatives may be sparse.
For first baseline, use soft_target + sample_weight, not only hard_label.
For slide/error analysis, use source_overlap.csv and label_quality_report.md.
Recommended Loader Flow

Read artifacts/model_inputs_v1/tile_manifest.csv.
Pick active fold.
For each train/val tile, load:
S2 files from file_manifest.csv where modality == "s2".
S1 files where modality == "s1".
AEF files where modality == "aef".
Labelpack from labelpack_path.
Align everything to s2_ref_path.
Sample pixels or patches where train_mask == 1.
Train with soft_target and sample_weight.
Validate only on tiles where fold_id == active_fold.
For test, use the same S1/S2/AEF loading path but no labels.
Reproduction Commands
Run tests:

PYTHONPATH=. .venv/bin/python3 -m unittest discover -s tests
Regenerate labels:

PYTHONPATH=. .venv/bin/python3 -m label_pipeline \
  --data-root data/makeathon-challenge \
  --split-dir splits/split_v1 \
  --output-root artifacts/labels_v1 \
  --force
Regenerate model-input handoff:

make prepare_model_inputs_v1
Current validation result:

Ran 13 tests
OK
One practical note: data/ and artifacts/ are ignored by git because they are large/generated. splits/ is intentionally not ignored, because the frozen split must be shared with the team.

