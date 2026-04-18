#!/usr/bin/env bash
# remote/push_results.sh — commit reports + submission + small model metadata
# to a fresh results branch and push to origin.
#
# Deliberately does NOT commit large artifacts:
#   - artifacts/features_v1/*.parquet (regeneratable)
#   - artifacts/models_v2_tomy/*.pt    (too big for git, stay on droplet)
#   - data/                            (gitignored, regeneratable from S3)
#
# Usage:
#   GIT_PAT=<pat> bash remote/push_results.sh <branch-name>
set -euo pipefail

RESULTS_BRANCH="${1:?Usage: $0 <results-branch-name>}"
: "${REPO_DIR:=$HOME/osapiens}"
: "${GIT_PAT:?Set GIT_PAT to a GitHub PAT with 'repo' write scope}"

cd "$REPO_DIR"

REMOTE_URL="$(git remote get-url origin)"
case "$REMOTE_URL" in
    https://*)
        # Inject PAT into the URL for push auth, without persisting it on disk.
        AUTH_URL="${REMOTE_URL/https:\/\//https://${GIT_PAT}@}"
        ;;
    *)
        echo "ERROR: origin is not an https URL — adjust auth logic for $REMOTE_URL"
        exit 1
        ;;
esac

# Keep reproducibility: record which source branch this was built off.
SOURCE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
SOURCE_SHA="$(git rev-parse HEAD)"

git checkout -B "$RESULTS_BRANCH"

# Stage the things that are meant to be shared.
git add -f \
    reports/tomy_v2*.md reports/tomy_v2*.json \
    submission/tomy_v2/ \
    artifacts/models_v2_tomy/*.json \
    logs/*.log 2>/dev/null || true

if git diff --cached --quiet; then
    echo "Nothing to commit on $RESULTS_BRANCH — did the training pipeline actually produce outputs?"
    exit 1
fi

git commit -m "results: ${RESULTS_BRANCH}

Source branch: ${SOURCE_BRANCH}
Source SHA:    ${SOURCE_SHA}
Host:          $(hostname)
Date (UTC):    $(date -u +%Y-%m-%dT%H:%M:%SZ)
"

git push "$AUTH_URL" "$RESULTS_BRANCH" --set-upstream --force-with-lease

echo
echo "Pushed $RESULTS_BRANCH."
echo "Pull locally with:"
echo "  git fetch origin && git checkout $RESULTS_BRANCH"
