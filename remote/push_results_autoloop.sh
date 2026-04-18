#!/usr/bin/env bash
# remote/push_results_autoloop.sh — autoloop-shaped sibling of push_results.sh.
#
# Commits the autoloop's reports + combined GeoJSONs + model-metadata JSON
# sidecars (NOT the *.pt checkpoints — too big) to a fresh results branch
# and pushes to origin using GIT_PAT auth.
set -euo pipefail

RESULTS_BRANCH="${1:?Usage: $0 <results-branch-name>}"
: "${REPO_DIR:=$HOME/osapiens}"
: "${GIT_PAT:?Set GIT_PAT to a GitHub PAT with 'repo' write scope}"

cd "$REPO_DIR"

REMOTE_URL="$(git remote get-url origin)"
case "$REMOTE_URL" in
    https://*)
        AUTH_URL="${REMOTE_URL/https:\/\//https://${GIT_PAT}@}"
        ;;
    *)
        echo "ERROR: origin is not an https URL"
        exit 1
        ;;
esac

SOURCE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
SOURCE_SHA="$(git rev-parse HEAD)"

git checkout -B "$RESULTS_BRANCH"

git add -f \
    submission/autoloop/ \
    artifacts/models_autoloop/*.json \
    reports/autoloop*.md reports/autoloop*.json \
    logs/*.log 2>/dev/null || true

if git diff --cached --quiet; then
    echo "Nothing to commit on $RESULTS_BRANCH — autoloop produced no artifacts?"
    exit 1
fi

git commit -m "autoloop results: ${RESULTS_BRANCH}

Source branch: ${SOURCE_BRANCH}
Source SHA:    ${SOURCE_SHA}
Host:          $(hostname)
Date (UTC):    $(date -u +%Y-%m-%dT%H:%M:%SZ)
"

git push "$AUTH_URL" "$RESULTS_BRANCH" --set-upstream --force-with-lease

echo "Pushed $RESULTS_BRANCH."
echo "Pull locally with:"
echo "  git fetch origin && git checkout $RESULTS_BRANCH"
