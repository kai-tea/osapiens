# Remote training on DigitalOcean AMD MI300X

One-shot workflow for training v2 models (Tomy's, initially) on a DO AMD GPU droplet, pushing results back to GitHub on a `results/<run-id>` branch, and pulling them locally for the next round of Claude-Opus-assisted improvements.

The scripts here do **not** implement the model. They set up the droplet, run whatever trainer lives at `tomy/scripts/train_v2.py`, and ship the results. The model itself is implemented by Claude Opus running on the droplet, guided by `prompts/tomy_v2_opus.md`.

## One-time setup

1. Provision a DO AMD GPU droplet (MI300X flavour). Ubuntu 22.04 default image works.
2. SSH in as `ubuntu` (or whichever user DO creates). Clone just enough to run bootstrap:
   ```bash
   sudo apt-get update && sudo apt-get install -y git
   git clone https://github.com/kai-tea/osapiens.git
   cd osapiens && git checkout feature/tomy
   bash remote/bootstrap.sh
   ```
   This installs ROCm PyTorch, pulls the rest of the repo dependencies, and downloads the challenge data from S3 (~15 GB).
3. From your **laptop**, rsync the two things that live outside the challenge bucket:
   ```bash
   rsync -av /home/doant/Documents/osapiens/artifacts/labels_v1/ \
     <droplet>:osapiens/artifacts/labels_v1/
   rsync -av /home/doant/Documents/osapiens/cini/splits/ \
     <droplet>:osapiens/cini/splits/
   ```
   These are Cini's label synthesis outputs (~15 MB) and her fold assignments (~1 KB). They aren't in git and aren't in the challenge S3 bucket.
4. Mint a GitHub PAT with `repo` write scope. Export on the droplet:
   ```bash
   export GIT_PAT=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

## Per-run loop

On the droplet, inside a `tmux` session so the run survives your SSH dropping:

1. Run an Opus coding session against `prompts/tomy_v2_opus.md` until it has written the three entrypoints under `tomy/scripts/`:
   - `train_v2.py`
   - `eval_v2.py`
   - `predict_v2.py`
   and a `tomy/src/v2/` package with the actual model. (If you don't have a Claude Code subscription on the droplet, copy the prompt into a local session, then `git push` the resulting code to `feature/tomy` and `git pull` it on the droplet.)
2. Kick off the pipeline:
   ```bash
   bash remote/run_training.sh
   ```
   This trains, runs ungated 3-fold CV, runs test-tile inference with the Hansen forest gate and refusal rule, and pushes everything worth keeping to a fresh `results/tomy-v2-<utc-timestamp>` branch.
3. On your laptop, pull:
   ```bash
   git fetch origin
   git checkout results/tomy-v2-<timestamp>
   ```
   Open `reports/tomy_v2.md` and `reports/tomy_v2_ungated.md`. Use Claude Opus locally to diagnose weaknesses and propose a v3 — then iterate.

## What gets pushed vs what stays on the droplet

Pushed (small, sharable):
- `reports/tomy_v2*.md` and `*.json`
- `submission/tomy_v2/*.geojson` and `summary.json`
- `artifacts/models_v2_tomy/*.json` (metadata only)
- `logs/*.log`

Not pushed (too big, regeneratable, or live):
- `artifacts/models_v2_tomy/*.pt` — stays on droplet. SCP manually if you want it on your laptop.
- `artifacts/features_v1/*.parquet` — regenerate from `src.data` if needed.
- `data/makeathon-challenge/` — re-download from S3 via `download_data.py`.

## Cost hygiene

- MI300X droplets are expensive. Stop the droplet (keep the volume) when not actively training.
- `remote/run_training.sh` writes all logs to `logs/` so you can `scp` them later if the SSH session drops.
- If you want multi-run parallelism, snapshot the droplet once the venv is ready and stamp new droplets from that snapshot rather than re-running `bootstrap.sh`.

## Extending to other teammates

The scripts are Tomy-shaped by default (paths, branch, model names). For Kangi's branch:
- Copy `prompts/tomy_v2_opus.md` to `prompts/kangi_v2_opus.md`, edit the role/branch/paths.
- Point `BRANCH=feature/kang-i` to `bootstrap.sh`.
- Replace `tomy/` paths in `run_training.sh` and `push_results.sh` with `kangi/`.
- Push to `results/kangi-v2-<timestamp>`.
