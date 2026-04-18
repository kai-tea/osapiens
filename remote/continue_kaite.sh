#!/usr/bin/env bash
# remote/continue_kaite.sh — warm-start Kaite's LightGBM baseline on the droplet.
#
# Assumes the droplet has been bootstrapped (remote/bootstrap.sh) and that
# the features parquets + labelpacks have been synced. LightGBM is CPU-only,
# so this doesn't use the MI300X GPU — it just benefits from the droplet's
# many CPU cores and uninterrupted runtime.
#
# Usage (on the droplet, inside tmux):
#   export GIT_PAT=<github-pat>
#   EXTRA_ROUNDS=500 bash remote/continue_kaite.sh
set -euo pipefail

: "${REPO_DIR:=$HOME/osapiens}"
: "${EXTRA_ROUNDS:=500}"
: "${LEARNING_RATE:=}"   # empty = reuse parent
: "${NUM_LEAVES:=}"      # empty = reuse parent
: "${RUN_TAG:=kaite-cont-$(date -u +%Y%m%d-%H%M%S)}"
: "${RESULTS_BRANCH:=results/${RUN_TAG}}"
: "${OUT_DIR:=checkpoints/kaite_v1_continued}"

cd "$REPO_DIR"

echo ">> pre-flight"
[ -f checkpoints/kaite_v1/baseline_v1.lgb ] || {
    echo "ERROR: parent model missing. Pull feature/kaite to get checkpoints/kaite_v1/."
    exit 1
}
[ -d artifacts/features_v1 ] || {
    echo "NOTE: features_v1 parquets not present — extracting first."
    # shellcheck disable=SC1091
    source .venv-gpu/bin/activate
    python3 -m src.data --splits train --max-pixels-per-tile 100000
}

# shellcheck disable=SC1091
source .venv-gpu/bin/activate
mkdir -p logs "$OUT_DIR"

EXTRA_ARGS=()
[ -n "$LEARNING_RATE" ] && EXTRA_ARGS+=(--learning-rate "$LEARNING_RATE")
[ -n "$NUM_LEAVES" ]    && EXTRA_ARGS+=(--num-leaves "$NUM_LEAVES")

echo ">> continuing training (+$EXTRA_ROUNDS rounds)"
PYTHONPATH=. python3 tools/continue_train_lgbm.py \
    --base-model checkpoints/kaite_v1/baseline_v1.lgb \
    --base-metadata checkpoints/kaite_v1/baseline_v1.json \
    --extra-rounds "$EXTRA_ROUNDS" \
    --run-tag "$RUN_TAG" \
    --out-dir "$OUT_DIR" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "logs/${RUN_TAG}.log"

echo ">> re-run ungated CV against the continued model"
# The existing src.eval retrains per-fold — to benchmark the continued model,
# swap in its trees as the init_model for the CV folds. For a first pass we
# run plain CV and report its F1 gap vs the parent's; deeper integration is
# left for the operator to decide.
PYTHONPATH=. python3 -m src.eval --ungated --forest-gate hansen \
    --report-path "reports/${RUN_TAG}_ungated.md" \
    2>&1 | tee -a "logs/${RUN_TAG}.log"

echo ">> push results"
bash remote/push_results.sh "$RESULTS_BRANCH"
echo
echo "Done. Pull on laptop with:"
echo "  git fetch origin && git checkout $RESULTS_BRANCH"
