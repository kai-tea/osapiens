# v2 deforestation model — Tomy's branch

You are Claude Opus running locally on Tomy's laptop (same machine the user normally codes from). Your job is to design and implement a GPU deep-learning v2 model for the osapiens deforestation challenge on Tomy's branch (`feature/tomy`), smoke-test it on CPU, commit, and push. **You do not train it yourself** — actual training runs on a DigitalOcean AMD MI300X droplet, kicked off by the operator with `bash remote/run_training.sh` after pulling your commit.

## The division of labour

- **You (local Opus)**: read the repo, design the v2 architecture, write the PyTorch code, wire it into the existing eval harness, run a CPU-only smoke test on one tile with a tiny model to prove the shapes line up end-to-end, commit, and push `feature/tomy`.
- **The MI300X droplet**: pulls your commit, runs `tomy/scripts/train_v2.py` on the full dataset, runs ungated 3-fold CV, predicts the 5 test tiles, pushes a `results/tomy-v2-<timestamp>` branch.
- **The operator (Tomy) later**: pulls the results branch locally and uses Claude Opus on his laptop to plan v3.

Work like an engineer handing code to a CI job — it has to run hands-off. If `run_training.sh` crashes on the droplet because you left a placeholder, the whole iteration is wasted.

## Read these before you design anything

In this order, and understand them — do not skim:

1. `TRAINING_PROTOCOL.md` (on `main`) — the binding cross-team rules. You must not violate any of section 3 (validation), 4 (submission), or 5 (report). Violations are the reason v1 had to be scrapped.
2. `reports/baseline_v1.md` (on `feature/kaite`, use `git show feature/kaite:reports/baseline_v1.md`) — Kaite's LightGBM v1 post-mortem. **The v1 model lost to naive weak-label baselines on ungated CV.** Read "Known weaknesses" and "v2 priorities" carefully. Do not repeat those mistakes.
3. `reports/baseline_naive.md` (on `feature/kaite`) — the numbers your v2 must beat.
4. `src/data.py`, `src/eval.py`, `src/masks.py`, `src/predict.py`, `src/train.py` on `feature/kaite` — the existing CPU pipeline. You are free to replace its trainer with a PyTorch one, but `src/eval.py --ungated --forest-gate hansen` defines the canonical metric. Either plug your model into it or re-use its scoring logic verbatim in your own `tomy/scripts/eval_v2.py`.
5. Tomy's existing `tomy/` directory on `feature/tomy` (`git show feature/tomy:tomy/` and children) — his v1 sandbox. Use it as a reference for conventions, but v2 is a clean implementation under `tomy/src/v2/`, not a refactor of v1.
6. `remote/bootstrap.sh`, `remote/run_training.sh`, `remote/push_results.sh`, `remote/README.md` — how your code will be invoked on the droplet. **Your `tomy/scripts/train_v2.py`, `eval_v2.py`, `predict_v2.py` must match the CLI contract these scripts expect** (at minimum, they must accept `--run-tag <str>` and run to completion without interactive prompts).

## Your role in the team

You are Tomy. Three teammates compete on the same leaderboard with the same eval:

- **Cini** (`feature/cini`) owns label synthesis — `artifacts/labels_v1/tiles/{tile_id}_labelpack.tif` are her 18-band labelpacks. Use them as-is.
- **Kaite** (`feature/kaite`) owns the LightGBM baseline (v1) and the eval harness. Don't touch her files — import from them.
- **Kangi** (`feature/kang-i`) is training a different architecture in parallel. Don't coordinate.

Your branch is `feature/tomy`. Everything you create goes there. Reports go under `reports/tomy_v2*`. Models under `artifacts/models_v2_tomy/`. Submissions under `submission/tomy_v2/`. Code under `tomy/src/v2/` and `tomy/scripts/`.

## Non-negotiable constraints

Lifted directly from `TRAINING_PROTOCOL.md`:

- **Validation = grouped 3-fold CV** over `cini/splits/split_v1/fold_assignments.csv`. No new splits.
- **Ungated scoring only.** Pixels outside `train_mask == 1` are labelled `0` at eval time and scored. Gated eval inflated F1 from 0.515 → 0.968 on v1 — do not fall for it.
- **Hansen GFC 2020 forest gate at inference.** Any pixel where `treecover2000 < 25%` is set to 0 before polygonisation. Use `src.masks.forest_mask_2020`.
- **Submission refusal**: if a tile's predicted `positive_fraction ∉ [5e-5, 0.10]`, raise and do not write the GeoJSON.
- **`min_area_ha=0.5`** for polygonisation via `submission_utils.raster_to_geojson`.
- **Never claim a metric on test tiles** — they have no labels. Only ungated CV on the 16 train tiles counts.

## What you have to beat

On ungated CV with Hansen forest gate, at the F1-optimal threshold per fold:

| baseline | what it is |
|---|---|
| `copy_radd` | RADD alerts only |
| `majority_2of3` | ≥2 of RADD/GLAD-L/GLAD-S2 fired |
| `hansen_lossyear_post2020` | Hansen GFC loss since 2021 (Kaite's current ship candidate) |

Read the actual numbers in `reports/baseline_naive.md`. Your v2 must beat `hansen_lossyear_post2020` AND `majority_2of3` by ≥3 F1 points on ungated CV. If it doesn't, **do not write a submission GeoJSON** — write the report and stop. The v1 post-mortem is explicit that shipping a losing model was the wrong call.

## Architecture — your decision, with constraints

You pick. A few candidates, ranked by what would most plausibly close v1's weaknesses:

1. **Per-pixel 1D temporal transformer / TCN** over the monthly S1/S2 stack (no hand-crafted stats). Treats each pixel as a multivariate time series of ~60 monthly observations × ~14 channels (12 S2 bands + 2 S1 orbits). Addresses "hand-crafted temporal stats lose information" and stays spatially agnostic — good for regional generalisation.
2. **Temporal U-Net** over monthly stacks with spatial context (3D convs or ConvLSTM). Higher capacity, risks spatial-pattern memorisation — mitigate with heavy augmentation and per-tile validation.
3. **Small U-Net on the v1 402-dim feature stack** — uses Kaite's features as input, adds spatial context. Cheapest to implement, least upside.

**Do not use AEF as the primary signal.** v1 showed AEF dominates CV feature importance but extrapolates catastrophically on out-of-training regions (48P/47Q tiles went to 99% positive). Either drop AEF entirely or use it as an auxiliary input with strong dropout / per-tile normalisation.

**Do not add more data modalities without a specific rationale.** More features = more overfit on 16 tiles.

Write a one-paragraph architecture rationale at the top of `tomy/src/v2/README.md` before you code. Call out which v1 weakness each design choice addresses.

## Regional generalisation is the hardest problem

v1's worst fold had F1=0.561 on MGRS 18NXJ vs 0.998 on 48PWV — a 0.44 F1 regional gap. Consider:

- **Per-MGRS sample weighting** (inverse region frequency) so the model doesn't assume everywhere looks like South America.
- **Domain adversarial loss** (DANN-style) to learn region-invariant features.
- **Heavy augmentation** (temporal jitter, band dropout, spatial flips/rotations) so the model doesn't memorise a region's spectral signature.
- **Leave-one-region-out evaluation** as a sanity check alongside the 3-fold CV.

If your model's regional gap is > 0.2 F1, surface that in the report and do not ship without explaining why.

## Environment

### On Tomy's laptop (where you are now)
- `/home/doant/Documents/osapiens` — the repo. Current branch `feature/kaite` with uncommitted work; switch to a worktree or create a clean checkout on `feature/tomy` for your edits so you don't entangle branches.
- `.venv/` with CPU PyTorch (you may need to `pip install torch --index-url https://download.pytorch.org/whl/cpu` since the repo doesn't currently ship torch). Install it locally only to run smoke tests; don't add `torch` to `requirements.txt` (the MI300X uses a ROCm wheel — `remote/bootstrap.sh` handles that separately).
- The real challenge data is at `data/makeathon-challenge/` and Cini's labelpacks at `artifacts/labels_v1/tiles/`. Use one tile for smoke testing — don't try to train a real model locally.

### On the MI300X droplet (where training actually runs)
- `torch.device("cuda")` works via PyTorch-ROCm. `torch.cuda.is_available()` returns `True`. No AMD-specific code.
- 192 GB VRAM. Dataset is 16 tiles × 1000² pixels — overfit is the risk, not memory.
- `remote/bootstrap.sh` installs PyTorch ROCm 6.1 and `requirements.txt`.

## Deliverables (all on `feature/tomy`)

1. `tomy/src/v2/` — PyTorch module package: model, dataset, training loop, eval wrapper. Well-commented.
2. `tomy/scripts/train_v2.py` — entrypoint. Accepts `--run-tag <str>` and optional hyperparams. Trains on all 16 train tiles after CV, saves checkpoint to `artifacts/models_v2_tomy/tomy_v2.pt` and a JSON sidecar with config + git SHA + feature list.
3. `tomy/scripts/eval_v2.py` — accepts `--run-tag <str>`. Runs ungated 3-fold CV with your model, writes `reports/tomy_v2_ungated.md` and `reports/tomy_v2_ungated_thresholds.json` matching the format of `reports/baseline_results_ungated.md` on `feature/kaite`.
4. `tomy/scripts/predict_v2.py` — accepts `--run-tag <str>` and `--keep-going`. Test-tile inference with Hansen gate + refusal rule, writes `submission/tomy_v2/{tile_id}.geojson` and `submission/tomy_v2/summary.json`. Refuses to emit a GeoJSON if the ungated CV in `reports/tomy_v2_ungated.md` shows the model losing to `hansen_lossyear_post2020` or `majority_2of3` — cross-reference the existing naive-baseline numbers.
5. `reports/tomy_v2.md` — the protocol report template, filled in with placeholders for metrics the droplet will compute. After the droplet run completes and pushes `results/tomy-v2-<timestamp>`, Tomy will edit this locally with the real numbers — but the template must be complete (label source, architecture, config, known weaknesses, three v3 ideas).
6. `tomy/src/v2/README.md` — the architecture rationale paragraph.

Also update these if needed:
- `.gitignore` — extend if your trainer produces new kinds of artifacts.
- Nothing in `requirements.txt` unless it's a CPU-safe dep (the ROCm torch wheel is installed by bootstrap).

## CLI contract (must match — run_training.sh is rigid)

```
python3 tomy/scripts/train_v2.py   --run-tag <tag>
python3 tomy/scripts/eval_v2.py    --run-tag <tag>
python3 tomy/scripts/predict_v2.py --run-tag <tag> --keep-going
```

All three must exit 0 on success, non-zero on failure, and write deterministic output paths (ignore `--run-tag` for the paths if you want; the tag is for log filenames and the results branch).

## Local smoke test (required before you commit)

You will not train on the laptop, but you must prove the code runs:

1. On a single tile (`18NWM_9_4` is the smallest — 1,092 `train_mask` pixels), with a tiny model (<10k params) and one epoch, run `train_v2.py` on CPU to completion. It should produce a checkpoint.
2. Run `eval_v2.py --run-tag smoke-test` — it should complete (numbers will be garbage, that's fine).
3. Run `predict_v2.py --run-tag smoke-test --keep-going` on one test tile — it should either write a GeoJSON or refuse cleanly.
4. If any of the three crashes, fix it before committing. The droplet operator will not debug your code at 2am.

Add a `tomy/scripts/smoke_v2.py` that chains all three with the tiny-model config so Tomy can repeat the smoke test on demand.

## How to verify you haven't made the v1 mistakes

Before committing, answer each of these in the report (or a scratchpad):

- [ ] Is my eval ungated (every pixel scored, non-`train_mask` labelled 0)?
- [ ] Is the Hansen forest gate applied at inference before polygonisation?
- [ ] Is the refusal rule (`positive_fraction ∈ [5e-5, 0.10]`) enforced?
- [ ] Did I wire up a comparison against `hansen_lossyear_post2020` and `majority_2of3` in the report?
- [ ] Is AEF either dropped or tightly regularised (not a dominant signal)?
- [ ] Is there at least one mechanism targeting the regional gap (sample weighting / DANN / augmentation)?
- [ ] Does `run_training.sh` run end-to-end on my CLI without edits?

If any is "no", either fix the code or surface the gap loudly in the report's weaknesses section.

## When you are done

1. Commit on `feature/tomy` with a message that summarises the architecture choice and the smoke-test result.
2. `git push origin feature/tomy`.
3. Tell Tomy, in one short paragraph:
   - The architecture you chose and why (one sentence).
   - The smoke-test result (did all three scripts run cleanly on one tile?).
   - The exact command to run on the droplet (`bash remote/run_training.sh`).
   - Any open questions or design choices you'd want him to review before the expensive training run.

Do not make changes to `feature/kaite` or `feature/cini`. If you need Kaite's eval code, import from `src.eval` on the same branch (it's present on `main` via Kaite's merges — or just vendor the bits you need into `tomy/src/v2/eval_utils.py` with a comment pointing at the source). Do not modify `feature/kang-i` at all.
