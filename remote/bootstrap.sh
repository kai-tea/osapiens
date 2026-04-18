#!/usr/bin/env bash
# remote/bootstrap.sh — one-shot setup on a fresh DigitalOcean AMD MI300X droplet.
#
# Idempotent: safe to re-run. Installs system deps, clones the repo, creates a
# ROCm PyTorch venv, and downloads the challenge data from S3. Does NOT sync
# Cini's labelpacks or the splits/ dir — those come via rsync from the operator's
# laptop (see remote/README.md).
#
# Usage (as the default ubuntu user on the droplet):
#   REPO_URL=https://github.com/kai-tea/osapiens.git \
#   BRANCH=feature/tomy \
#   bash bootstrap.sh
set -euo pipefail

: "${REPO_URL:=https://github.com/kai-tea/osapiens.git}"
: "${REPO_DIR:=$HOME/osapiens}"
: "${BRANCH:=feature/tomy}"
: "${GIT_USER_NAME:=Tomy (MI300X droplet)}"
: "${GIT_USER_EMAIL:=tomy.doan@tngtech.com}"

echo ">> system packages"
sudo apt-get update -y
sudo apt-get install -y \
    git tmux htop jq curl rsync build-essential cmake \
    python3.10 python3.10-venv python3.10-dev python3-pip

echo ">> clone or update repo at $REPO_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin --prune
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo ">> git identity"
git config user.name "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

echo ">> python venv (.venv-gpu)"
if [ ! -d .venv-gpu ]; then
    python3.10 -m venv .venv-gpu
fi
# shellcheck disable=SC1091
source .venv-gpu/bin/activate
pip install -U pip wheel setuptools

echo ">> PyTorch ROCm 6.1 (MI300X-compatible)"
pip install --index-url https://download.pytorch.org/whl/rocm6.1 \
    torch torchvision torchaudio

echo ">> repo requirements"
pip install -r requirements.txt

echo ">> GPU smoke test"
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("vram GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
PY

echo ">> challenge data from S3 (skipped if present)"
if [ ! -d data/makeathon-challenge ]; then
    python3 -m download_data
else
    echo "   data/makeathon-challenge already exists — skipping"
fi

echo
echo "Bootstrap complete. Next steps (run from your laptop):"
echo
echo "  # sync Cini's labelpacks + manifest + splits to the droplet (one-time, ~15 MB):"
echo "  rsync -av /home/doant/Documents/osapiens/artifacts/labels_v1/ \\"
echo "    <droplet>:${REPO_DIR}/artifacts/labels_v1/"
echo "  rsync -av /home/doant/Documents/osapiens/artifacts/model_inputs_v1/ \\"
echo "    <droplet>:${REPO_DIR}/artifacts/model_inputs_v1/"
echo "  rsync -av /home/doant/Documents/osapiens/cini/splits/ \\"
echo "    <droplet>:${REPO_DIR}/cini/splits/"
echo
echo "Then on the droplet:"
echo "  cd ${REPO_DIR}"
echo "  export GIT_PAT=<your-github-pat>"
echo "  bash remote/run_training.sh"
