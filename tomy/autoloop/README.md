# Tomy's autoloop

Fire-and-forget 4 h pipeline that, without operator intervention, produces
a set of ranked candidate submission GeoJSONs under `submission/autoloop/`.

## What it does

1. **Safety-net heuristic variants** — six Hansen-GFC grid points over
   `treecover_threshold` × `lossyear window` × morphological cleanup.
   No training; runs in minutes. Also adds per-polygon `time_step`
   (mode lossyear → YYMM, mid-year month) which Kaite's shipped heuristic
   did not.
2. **Two MLP configs** — 256×128 feed-forward with:
   - Soft-IoU + BCE loss (aligned with the Union-IoU primary metric).
   - AEF feature-group dropout (kills v1's regional over-fit to AEF).
   - Per-MGRS inverse-frequency sample weighting (closes the 0.44 F1
     regional gap).
   - Config A: `pos_weight=3`, `iou_weight=0.5`, AEF drop 50%.
   - Config B (recall-tilted): `pos_weight=5`, `iou_weight=0.7`, AEF drop 75%.
3. **Ungated 3-fold CV** per config — same rules as `src.eval --ungated
   --forest-gate hansen`. Threshold is the median of per-fold F1-optima.
4. **Self-training** — take the winning config, pseudo-label training
   pixels where (model > 0.7 AND any weak source agrees) or
   (model < 0.1 AND no alert fires), retrain.
5. **Ensemble** — union of (winning-config binary) with
   `hansen_lossyear_post2020`, polygonised with time_step from Hansen.
6. **Summary** — `submission/autoloop/summary.md` ranks every candidate
   by ungated CV F1 alongside the naive-baseline anchors, with a
   recommended submit order.

The script never submits to the leaderboard — the operator picks a
GeoJSON and uploads via the osapiens web UI. Every candidate is a
single merged GeoJSON at `submission/autoloop/<tag>.geojson`, plus the
per-tile files at `submission/autoloop/<tag>/<tile_id>.geojson`.

## Hard constraints enforced

All per the pinned protocol:

- Ungated CV — pixels outside `train_mask` labelled 0 at eval time.
- Hansen GFC 2020 forest gate at inference (`treecover >= 25%` by default).
- Refusal rule: `positive_fraction ∉ [5e-5, 0.10]` raises on that tile.
- `min_area_ha = 0.5` on polygonisation.
- `time_step`: mode Hansen lossyear within polygon → YYMM; absent if no
  lossyear pixel overlaps (never invented, since Year Accuracy penalises
  wrong years by area).

## Run

On the MI300X droplet:

```bash
cd ~/osapiens
bash tomy/autoloop/run.sh
```

The script does its own pre-flight checks. Expect ~2–3 h wall clock,
of which the first ~30 min is `src.data` building the per-tile feature
parquets (skipped if they already exist).

Environment variables:

- `EPOCHS` (default `20`) — per-config training epochs.
- `RUN_TAG` (default `autoloop-<utc-timestamp>`) — used for log names
  and the results branch.
- `GIT_PAT` — optional; when set, the results branch is pushed to origin.

## Outputs

- `submission/autoloop/summary.md` — candidate table + recommended order.
- `submission/autoloop/summary.json` — machine-readable version.
- `submission/autoloop/<tag>.geojson` — merged, one per candidate.
- `submission/autoloop/<tag>/<tile_id>.geojson` — per-tile.
- `artifacts/models_autoloop/*.pt` + `*.json` — checkpoints + metadata.
- `logs/<run_tag>_*.log` — stage logs.

## What this does *not* do

- No raw-temporal architecture (1D TCN / ConvLSTM). Kaite's 402-dim
  parquet cache makes an MLP with soft-IoU loss the fastest buildable
  upgrade over LightGBM v1 in a 4 h window. Raw-stack temporal models
  are the v3 priority.
- No leaderboard-in-the-loop. Submissions are rate-limited to 10 total
  — the operator judges which to burn.
- No automatic pushing to `main` / `feature/kaite`. Results go to a fresh
  `results/autoloop-<timestamp>` branch.
