#!/bin/bash
# Validate and synchronize Qwen3.6 smoke outputs after the GPU job.
# ADJUST: module path if SCCKN changes its runtime.
#$ -q scc
#$ -l h_rt=00:15:00
#$ -l h_vmem=4G
#$ -pe smp 1
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${REPO_PATH:?REPO_PATH is required}"
: "${GIT_COMMIT:?GIT_COMMIT is required}"
: "${GPU_SENTINEL:?GPU_SENTINEL is required}"
: "${FINAL_SENTINEL:?FINAL_SENTINEL is required}"

module load conda  # ADJUST
conda activate wc-qwen36-hf
export PYTHONPATH="$REPO_PATH"
cd "$REPO_PATH"

if ! git merge-base --is-ancestor "$GIT_COMMIT" HEAD; then
  echo "Refusing Qwen3.6 finalization: submitted commit is not an ancestor of HEAD." >&2
  exit 10
fi
if ! git diff --quiet "$GIT_COMMIT" HEAD -- \
  "$CONFIG_PATH" \
  src/qwen36_smoke.py \
  src/validate_qwen36_smoke.py \
  jobs/sge/qwen36_smoke_finalize.sh \
  jobs/sync_outputs.sh; then
  echo "Refusing Qwen3.6 finalization: critical implementation changed after submission." >&2
  exit 11
fi
if [[ ! -f "$GPU_SENTINEL" ]]; then
  echo "Refusing Qwen3.6 finalization: missing GPU success sentinel: $GPU_SENTINEL" >&2
  exit 20
fi

python -m src.validate_qwen36_smoke --config "$CONFIG_PATH"
bash jobs/sync_outputs.sh "$REPO_PATH"

mkdir -p "$(dirname "$FINAL_SENTINEL")"
sentinel_tmp="${FINAL_SENTINEL}.tmp.${JOB_ID:-$$}"
: > "$sentinel_tmp"
mv "$sentinel_tmp" "$FINAL_SENTINEL"
echo "[success] Qwen3.6 smoke outputs validated and synchronized"
