#!/usr/bin/env bash
# remote/run_training.sh — end-to-end v2 run on the MI300X droplet.
#
# Activates the ROCm venv, runs Tomy's v2 training + eval + prediction,
# then pushes the reports + submission to a fresh results branch.
#
# Expected to be invoked after:
#   1. bash remote/bootstrap.sh
#   2. rsync of artifacts/labels_v1/ and cini/splits/ from the operator's laptop
#   3. export GIT_PAT=<github PAT with repo write scope>
#
# The trainer/eval/predict entrypoints under tomy/scripts/ are expected to
# exist — they are produced by the Opus agent running on the droplet with
# prompts/tomy_v2_opus.md. If they don't exist yet, this script prints a
# clear error pointing at the prompt.
set -euo pipefail

: "${REPO_DIR:=$HOME/osapiens}"
: "${RUN_TAG:=tomy-v2-$(date -u +%Y%m%d-%H%M%S)}"
: "${RESULTS_BRANCH:=results/${RUN_TAG}}"

cd "$REPO_DIR"

echo ">> pre-flight checks"
[ -d data/makeathon-challenge/sentinel-2 ] || {
    echo "ERROR: challenge data missing. Run bootstrap.sh."
    exit 1
}
[ -d artifacts/labels_v1/tiles ] || {
    echo "ERROR: Cini's labelpacks missing at artifacts/labels_v1/tiles/."
    echo "       rsync them from your laptop — see remote/README.md."
    exit 1
}
[ -f cini/splits/split_v1/fold_assignments.csv ] || {
    echo "ERROR: Cini's fold assignments missing at cini/splits/split_v1/."
    echo "       rsync them from your laptop — see remote/README.md."
    exit 1
}
for f in tomy/scripts/train_v2.py tomy/scripts/eval_v2.py tomy/scripts/predict_v2.py; do
    [ -f "$f" ] || {
        echo "ERROR: missing $f."
        echo "       Have the Opus agent on the droplet follow prompts/tomy_v2_opus.md first."
        exit 1
    }
done

# shellcheck disable=SC1091
source .venv-gpu/bin/activate

mkdir -p logs artifacts/models_v2_tomy submission/tomy_v2

echo ">> train (run tag: $RUN_TAG)"
python3 tomy/scripts/train_v2.py --run-tag "$RUN_TAG" 2>&1 \
    | tee "logs/${RUN_TAG}_train.log"

echo ">> ungated CV"
python3 tomy/scripts/eval_v2.py --run-tag "$RUN_TAG" 2>&1 \
    | tee "logs/${RUN_TAG}_eval.log"

echo ">> test-tile inference (Hansen-gated, refusal-rule enforced)"
python3 tomy/scripts/predict_v2.py --run-tag "$RUN_TAG" --keep-going 2>&1 \
    | tee "logs/${RUN_TAG}_predict.log" \
    || echo "   predict exited non-zero — check log before blaming the push"

echo ">> push results"
RESULTS_BRANCH="$RESULTS_BRANCH" bash remote/push_results.sh "$RESULTS_BRANCH"

echo
echo "Done. Pull on laptop with:"
echo "  git fetch origin && git checkout $RESULTS_BRANCH"
