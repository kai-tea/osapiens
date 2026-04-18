#!/usr/bin/env bash
# tomy/autoloop/run.sh — fire-and-forget 4 h autoloop on the MI300X droplet.
#
# Prerequisites (bootstrap must have been run once):
#   - data/makeathon-challenge/         (bootstrap.sh downloads this)
#   - artifacts/labels_v1/tiles/        (rsync from laptop, see remote/README.md)
#   - cini/splits/split_v1/fold_assignments.csv  (in git)
#   - artifacts/model_inputs_v1/tile_manifest.csv (needed by src.data — see below)
#
# What this script does, in order:
#   1. Ensure Kaite's per-tile feature parquets exist under artifacts/features_v1/.
#      If not, produce them for all 21 tiles (16 train + 5 test). ~30 min.
#   2. Run the autoloop orchestrator: heuristic variants → 2 model configs →
#      self-train → ensemble → summary. ~1.5–2.5 h.
#   3. Push the results branch (submission + reports + model JSON sidecars).
#
# Safe to re-run: each stage checks for existing outputs and skips if present.
set -euo pipefail

: "${REPO_DIR:=$HOME/osapiens}"
: "${RUN_TAG:=autoloop-$(date -u +%Y%m%d-%H%M%S)}"
: "${RESULTS_BRANCH:=results/${RUN_TAG}}"
: "${EPOCHS:=20}"
: "${MAX_PIXELS_PER_TILE:=100000}"

cd "$REPO_DIR"

# shellcheck disable=SC1091
source .venv-gpu/bin/activate

mkdir -p logs artifacts/features_v1 artifacts/models_autoloop \
    artifacts/predictions_autoloop submission/autoloop

echo ">> pre-flight"
[ -d data/makeathon-challenge/sentinel-2 ] || {
    echo "ERROR: challenge data missing at data/makeathon-challenge/. Run bootstrap.sh."
    exit 1
}
[ -d artifacts/labels_v1/tiles ] || {
    echo "ERROR: Cini's labelpacks missing. rsync them — see remote/README.md."
    exit 1
}
[ -f cini/splits/split_v1/fold_assignments.csv ] || {
    echo "ERROR: fold assignments missing."
    exit 1
}
[ -f artifacts/model_inputs_v1/tile_manifest.csv ] || {
    echo "ERROR: tile_manifest.csv missing at artifacts/model_inputs_v1/. rsync it."
    exit 1
}

echo ">> stage 1: train-tile feature parquets (skip if already present)"
N_EXPECTED=$(awk -F',' 'NR>1 && $1=="train" {print $2}' artifacts/model_inputs_v1/tile_manifest.csv | wc -l)
N_PRESENT=$(awk -F',' 'NR>1 && $1=="train" {print $2}' artifacts/model_inputs_v1/tile_manifest.csv \
    | while read -r t; do [ -f "artifacts/features_v1/${t}.parquet" ] && echo 1; done | wc -l)
if [ "$N_PRESENT" -lt "$N_EXPECTED" ]; then
    echo "   present=$N_PRESENT expected=$N_EXPECTED — running src.data on train split"
    PYTHONPATH=. python3 -m src.data \
        --splits train \
        --max-pixels-per-tile "$MAX_PIXELS_PER_TILE" \
        2>&1 | tee "logs/${RUN_TAG}_features.log"
else
    echo "   all $N_PRESENT train parquets present — skipping"
fi

echo ">> stage 2: autoloop orchestrator"
PYTHONPATH=. python3 -m tomy.autoloop.main \
    --epochs "$EPOCHS" \
    2>&1 | tee "logs/${RUN_TAG}_autoloop.log"

echo ">> stage 3: collect reports + submissions into results branch"
if [ -n "${GIT_PAT:-}" ]; then
    RESULTS_BRANCH="$RESULTS_BRANCH" bash remote/push_results_autoloop.sh "$RESULTS_BRANCH"
else
    echo "   GIT_PAT unset — skipping push. Results live in submission/autoloop/."
fi

echo
echo "Done. Candidates in submission/autoloop/ (summary.md + *.geojson)."
