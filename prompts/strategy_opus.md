# osapiens deforestation challenge — strategy advisory prompt

You are Claude Opus being consulted as an outside expert. You are **not** writing code and not being handed a branch. Your one job is to return a concrete, ranked strategy for **maximizing the live-leaderboard score** on the osapiens deforestation challenge, given the constraints, data, and hardware described below. The team will do the implementation; they want your plan, your reasoning, and your priority order.

The deadline is **2026-04-19 11:00 Europe/Berlin** — roughly 18 hours from when this prompt is issued. Optimize recommendations for *what is achievable in that window*, not for what would be ideal with a week.

## What the leaderboard actually measures

The live leaderboard scores **polygon-level**, not pixel-level. Ground truth is a union of polygons, and predictions are a GeoJSON `FeatureCollection` of Polygon / MultiPolygon. Four metrics:

1. **Union IoU** — *primary ranking metric*. `area(pred_union ∩ gt_union) / area(pred_union ∪ gt_union)`. Optimize this first.
2. **Polygon Recall** — `area(pred_union ∩ gt_union) / area(gt_union)`. Pushed up by predicting more; traded off against FPR.
3. **Polygon-level FPR** — `area(pred_union \ gt_union) / area(pred_union)`. Pushed down by predicting less.
4. **Year Accuracy** — correctly dated overlap area / union of predictions and the temporal GT subset. Wrong year, missing `time_step`, missed detections, and false detections all penalize by area. `time_step` is optional YYMM (e.g., `2204` = April 2022).

Constraints worth internalizing:
- **10 counted submissions total** (failed submissions don't count). We can't grid-search the leaderboard — we get at most ~10 probes.
- Submission file is a single GeoJSON FeatureCollection covering all test tiles. Polygon or MultiPolygon only.
- `time_step` is optional but Year Accuracy is part of the score, so emitting it is strictly dominant *if* we can estimate year reliably.

Because Union IoU is area-based on the union of polygons, **one large false-positive polygon over a non-forest area tanks the score**. Conversely, shaving true-positive area by over-eroding hurts Recall. The scoring is unforgiving to both over- and under-prediction.

## The data we have

16 labeled training tiles + 5 unlabeled test tiles. Each tile is ~1000² pixels at Sentinel-2 10 m resolution. Per pixel we have:

- **Sentinel-1 RTC** — monthly ascending + descending (VV, VH), ~60 months covering 2020-2024.
- **Sentinel-2 L2A** — monthly optical, 12 bands, ~60 months (gappy due to clouds).
- **AEF foundation-model embeddings** — annual per-pixel, one vector per tile-year. **Warning**: v1 post-mortem showed AEF dominates feature importance on CV but extrapolates catastrophically on held-out regions (pushed two test tiles to 99% positive). Use only with care.
- **Weak labels** — three independent sources, all as rasters with per-pixel event flags and rough dates:
  - `RADD` — Sentinel-1-based alerts, near-real-time, high precision.
  - `GLAD-L` — Landsat-based, annual.
  - `GLAD-S2` — Sentinel-2-based.
- **Hansen GFC** — global forest cover / lossyear raster (pulled separately). Already used as a gating mask.
- **Cini's label packs** at `artifacts/labels_v1/tiles/{tile_id}_labelpack.tif` — 18-band consolidated label stacks (soft targets + per-source flags + train_mask).

## What's already been tried (and what beat what)

The team ran an honest 3-fold grouped CV (grouped by MGRS tile to prevent spatial leakage) with **ungated scoring** (pixels outside `train_mask` labeled 0) and a **Hansen 2020 tree-cover ≥ 25%** gate applied at inference. On that harness:

| strategy | mean F1 | mean IoU | mean PR-AUC | notes |
|---|---:|---:|---:|---|
| `copy_radd` (just RADD alerts) | 0.902 | 0.826 | 0.854 | very high precision, decent recall |
| `majority_2of3` (≥2 of RADD/GLAD-L/GLAD-S2) | 0.744 | 0.597 | 0.943 | near-perfect precision, low recall |
| `copy_gladl` | 0.585 | 0.414 | 0.474 | weak |
| `copy_glads2` | 0.522 | 0.380 | 0.495 | weak |
| **v1 LightGBM (Kaite)** | **lost to both `copy_radd` and `majority_2of3` on ungated CV** | — | — | dropped from shipping |

Two failure modes jurisdictionally confirmed by v1:
- **AEF overfits regionally.** Feature importance top-1 in-fold; in out-of-distribution MGRS tiles (48P/47Q) it saturated positive.
- **Regional gap.** v1 had F1 0.998 on 48PWV vs 0.561 on 18NXJ — 0.44 F1 gap. RADD alone is near-zero on `18NWM`, `18NYH`, and ~0.27 on `19NBD` but ~0.99 on Southeast Asia tiles.

Kaite's current ship candidate for a heuristic submission is **`hansen_lossyear_post2020`** (i.e., Hansen GFC pixels where `lossyear >= 21`), and a checkpoint of v1 LightGBM is parked at `checkpoints/kaite_v1/` but is not to be submitted.

## Hardware and time budget

One remote AMD MI300X droplet: **1× MI300X, 192 GB VRAM, 20 vCPU, 240 GB RAM, 720 GB boot NVMe, 5 TB scratch**. PyTorch-ROCm 6.1 via `remote/bootstrap.sh`. `torch.cuda.is_available()` returns `True`; no AMD-specific code needed.

Dataset total is small (16 tiles × ~10⁶ pixels × ~60 months × ~14 channels ≈ 13 GB of floats). Memory is not the bottleneck; **overfit is**, and **wall-clock training time** is (we have one droplet, one person to operate it, and ~18 hours).

No GPU locally. Local machine runs CPU smoke tests only.

## Hard constraints the team will not break

These are load-bearing — any plan violating them will be rejected:

- **Validation = grouped 3-fold CV** on `cini/splits/split_v1/fold_assignments.csv`, grouped by MGRS. No new splits, no random splits.
- **Ungated scoring** during CV (non-`train_mask` pixels labeled 0). Gated eval inflated v1 F1 0.515 → 0.968 — it lies.
- **Hansen forest gate at inference**: zero out any pixel where `treecover2000 < 25%` before polygonization.
- **Refusal rule**: if a tile's `positive_fraction ∉ [5e-5, 0.10]`, refuse to emit a polygon for that tile.
- **`min_area_ha=0.5`** in `raster_to_geojson`.
- **Never claim a metric on the 5 test tiles.** They have no labels. Only ungated CV on the 16 train tiles is truth.
- **Must beat `hansen_lossyear_post2020` AND `majority_2of3` by ≥3 F1 points on ungated CV** before shipping a learned model. If it doesn't beat them, ship the heuristic.

## The leaderboard-vs-CV gap is the real risk

The leaderboard scores polygon Union IoU on a hidden test set of 5 tiles; our CV scores pixel-level F1/IoU on the 16 train tiles. These are **not the same metric**. Specifically:

- Pixel IoU ≠ polygon Union IoU — polygonization (threshold + `min_area_ha` + morphological ops) is a large and separately-tunable part of the pipeline.
- 3 of our 16 train tiles show near-zero RADD recall, implying ground-truth distribution shift already present *within* CV. Test tiles may be worse.
- We only get 10 submissions. We cannot tune the polygonization pipeline against the leaderboard — we have to pick it from CV and accept the gap.

## What we want from you

Return a single plan, structured as follows, in one markdown document. Be concrete (name layers, loss functions, hyperparameters, thresholds). Rank by expected Union-IoU lift per hour of implementation.

### 1. Dominant strategy in 18 hours (≤ 200 words)
One paragraph. What's the single best overall bet given the time budget, the 10-submission cap, the regional-generalization risk, and the fact that `copy_radd` already scores IoU 0.826 on CV? Should we ship a souped-up heuristic, a learned model, or an ensemble of both? Why?

### 2. Architecture recommendation
Pick one primary architecture and defend it against the others. Candidates to consider (feel free to propose another):

- Per-pixel 1D temporal transformer / TCN over monthly S1+S2.
- Temporal U-Net (3D convs or ConvLSTM) over monthly stacks with spatial context.
- Small U-Net on the v1 402-dim feature stack.
- Weak-label ensemble + thin calibration layer, no deep model.
- Hansen + RADD + weak-label fusion rule (symbolic, no training).

For the chosen architecture, specify: input tensor shape, temporal encoder, spatial encoder (if any), loss (BCE / Dice / Tversky / Lovász — which and why for **Union IoU**), class balancing strategy, regularization, and how the AEF regional-overfit failure is avoided.

### 3. Training recipe for the MI300X
Concrete: batch construction (per-tile, per-patch, per-pixel?), sampling strategy across the 16 tiles with the known regional imbalance, optimizer + LR schedule, augmentations (temporal jitter, band dropout, spatial flips — which specifically?), epoch count given the ~13 GB dataset and 192 GB VRAM, mixed precision yes/no on ROCm, and a stopping criterion tied to ungated CV.

### 4. Temporal head for Year Accuracy
Should we predict `time_step`? If yes, how — regression vs classification, per-pixel vs per-polygon, trained jointly or decoupled? What's the expected lift and downside? Year Accuracy penalizes wrong years *and* missing `time_step` — so emitting a noisy year is not strictly dominant over omitting it.

### 5. Inference + polygonization pipeline
The step between per-pixel probability and the GeoJSON is where most Union IoU is won or lost. Specify: threshold selection (per-tile vs global; F1-optimal vs Youden-J vs calibrated), morphological post-processing (open/close radii), minimum polygon area, simplification tolerance, and how to assign `time_step` to a polygon (mode of per-pixel predicted month? weighted by area?). Explicitly account for Hansen gate + refusal rule.

### 6. Ensembling strategy
With only 10 submissions, ensembling has to be decided pre-submission. Should we ensemble (learned model ∪ RADD ∪ Hansen lossyear)? Union, intersection, or weighted vote? Thresholds? Specify.

### 7. Submission ladder (10 slots, ordered)
Given we can submit at most 10 times, list the exact ordered sequence of submissions you'd make. For each: what it is, what it tests, and the *decision rule* for whether to continue or fall back. Example shape:
> 1. `hansen_lossyear_post2020` baseline — anchor.
> 2. `copy_radd` — anchor.
> 3. `majority_2of3 ∪ hansen_lossyear_post2020` — heuristic ensemble ceiling.
> 4. v2 learned model, threshold = CV-F1-optimal.
> 5. v2 learned model ∪ `copy_radd`. …

### 8. What to explicitly *not* do
Call out the traps. AEF as primary signal, gated eval, leaderboard-only tuning without CV anchor, training without augmentation, shipping a model that loses to `majority_2of3` on CV, etc. Short list.

### 9. Honest risk assessment
One paragraph: what's the single most likely reason the final score underperforms what CV predicts, and what is the cheapest pre-submission check that would catch it?

## Tone and format

Technical, terse, opinionated. No hedging ("it depends", "could consider") without a concrete recommendation attached. Assume the reader is a strong ML engineer who needs your plan, not a tutorial. No code blocks longer than 10 lines. Cite specific numbers from the CV table above when defending a choice.

When a trade-off is genuinely close, state both options, name the one you'd pick, and give the one-sentence reason. Do not produce a balanced-both-sides analysis.
