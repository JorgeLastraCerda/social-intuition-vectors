#!/bin/bash
# sync_outputs.sh — commit and push lightweight pipeline outputs to git.
#
# Tracked paths (un-ignored in .gitignore):
#   data/processed/concept_vectors*/   (direction vectors, activation matrices, meta.json)
#   results/logs/validate_probes_*.json
#   results/tables/probe_metrics*.csv
#   results/tables/layer_sweep*.csv + layer_sweep*.meta.json
#
# NOT committed: model weights (*.safetensors, *.bin, *.pt), SGE logs (*.out, *.err),
#               HF cache (/work/.../hf_cache).
#
# Usage:
#   bash jobs/sync_outputs.sh [REPO_PATH]
#   REPO_PATH defaults to /work/emrecan.ulu/normalcy-axis (SCCKN work dir).
#   On local machines, run from the repo root without an argument.
#
# One-time SCCKN credential setup (run from the login node, NOT a compute node):
#   git config --global credential.helper store
#   git push origin main        # enter your GitHub username + Personal Access Token once
#   # Token is then cached in ~/.git-credentials; subsequent pushes are passwordless.
#   # Alternatively, switch the remote to SSH and add a deploy key.
#   # ADJUST: ask Stefan if the cluster policy prefers SSH keys or tokens.
#
# Conflict handling: git pull --rebase folds in collaborator work before pushing,
# so this script never needs --force and is safe to run concurrently from multiple jobs.
# If two jobs race and both get a rejected push, rerun this script once from the login node.

set -euo pipefail

REPO="${1:-/work/emrecan.ulu/normalcy-axis}"  # ADJUST: SCCKN path if different
cd "$REPO"

git add \
    data/processed/concept_vectors*/ \
    results/logs/validate_probes_*.json \
    results/tables/probe_metrics*.csv \
    results/tables/layer_sweep*.csv \
    results/tables/layer_sweep*.meta.json \
    2>/dev/null || true   # tolerate missing globs (e.g. first run before extraction)

if git diff --cached --quiet; then
    echo "[sync] nothing to commit — outputs already up to date"
    exit 0
fi

STAMP=$(date -u +%Y-%m-%dT%H:%MZ 2>/dev/null || date -u)
git commit -m "Sync pipeline outputs from $(hostname) (${STAMP})"
git pull --rebase origin main
git push origin main
echo "[sync] pushed successfully."
