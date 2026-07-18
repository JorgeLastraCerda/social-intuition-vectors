#!/bin/bash
# sync_outputs.sh — commit and push lightweight pipeline outputs to git.
#
# Tracked paths (un-ignored in .gitignore):
#   data/processed/concept_vectors*/   (direction vectors, activation matrices, meta.json)
#   results/logs/validate_probes_*.json
#   results/tables/probe_metrics*.csv
#   results/tables/layer_sweep*.csv + layer_sweep*.meta.json
#   data/processed/gemma_scope_*/   (sparse activations and derived vectors)
#   results/logs/gemma_scope_*.json
#   results/tables/gemma_scope_*.csv
#   results/tables/steering_dense_*.csv       (dense steering summaries + raw)
#   results/logs/steering_dense_*.json        (dense steering provenance)
#   results/tables/hiring_audit_*.csv         (Phase 7 probe-vs-human)
#   results/tables/hiring_steering_raw_*.csv  (Phase 7 causal sweep)
#   results/tables/hiring_disparity_*.csv     (Phase 7 disparity + mediation)
#   results/logs/hiring_steering_*.json       (Phase 7 causal sweep provenance)
#   results/logs/hiring_probe_vs_human_*.json (Phase 7 probe-vs-human provenance)
#   results/logs/hiring_mediation_*.json      (Phase 7 mediation provenance)
#   results/tables/hiring_{group_r4,name_level}_*.csv (R4 outputs)
#   results/logs/{hiring_r4,smoke_gemma4}_*.json      (R4/smoke provenance)
#   results/logs/gemma4_stages_1_3_submission_*.json  (staged-run job manifest)
#   results/logs/gemma4_parity_*.json  (multi-GPU parity manifests/results)
#   results/logs/gemma4_stage3_retry_submission_*.json (parallel Stage 3 retry manifest)
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

shopt -s nullglob
output_paths=(
    data/processed/concept_vectors*/
    data/processed/gemma_scope_*/
    results/logs/validate_probes_*.json
    results/logs/gemma_scope_*.json
    results/tables/probe_metrics*.csv
    results/tables/layer_sweep*.csv
    results/tables/layer_sweep*.meta.json
    results/tables/gemma_scope_*.csv
    results/tables/steering_dense_*.csv
    results/logs/steering_dense_*.json
    results/tables/hiring_audit_*.csv
    results/tables/hiring_steering_raw_*.csv
    results/tables/hiring_disparity_*.csv
    results/tables/hiring_group_r4_*.csv
    results/tables/hiring_name_level_*.csv
    results/logs/hiring_steering_*.json
    results/logs/hiring_probe_vs_human_*.json
    results/logs/hiring_mediation_*.json
    results/logs/hiring_r4_*.json
    results/logs/smoke_gemma4_*.json
    results/logs/smoke_qwen36_*.json
    results/logs/qwen36_smoke_submission_*.json
    results/logs/qwen36_*_stage*.json
    results/logs/qwen36_full_submission_*.json
    results/logs/gemma4_stages_1_3_submission_*.json
    results/logs/gemma4_parity_*.json
    results/logs/gemma4_stage3_retry_submission_*.json
)
if ((${#output_paths[@]})); then
    git add "${output_paths[@]}"
fi

if git diff --cached --quiet; then
    echo "[sync] nothing to commit — outputs already up to date"
    exit 0
fi

STAMP=$(date -u +%Y-%m-%dT%H:%MZ 2>/dev/null || date -u)
git commit -m "Sync pipeline outputs from $(hostname) (${STAMP})"
git pull --rebase origin main
git push origin main
echo "[sync] pushed successfully."
