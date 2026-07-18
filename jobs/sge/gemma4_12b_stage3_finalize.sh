#!/bin/bash
# Validate and synchronize the independent Gemma 4 12B Stage 3 output.
# This CPU-only job is held on the L40 job ID.
# ADJUST: module path if SCCKN changes its runtime environment.
#$ -q scc
#$ -l h_rt=00:15:00
#$ -l h_vmem=4G
#$ -pe smp 1
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${REPO_PATH:?REPO_PATH is required}"
: "${GIT_COMMIT:?GIT_COMMIT is required}"
: "${SUCCESS_SENTINEL:?SUCCESS_SENTINEL is required}"
: "${FINAL_SENTINEL:?FINAL_SENTINEL is required}"

module load conda  # ADJUST
conda activate wc-tl-g4
export PYTHONPATH="$REPO_PATH"
cd "$REPO_PATH"

if ! git merge-base --is-ancestor "$GIT_COMMIT" HEAD; then
  echo "Refusing 12B Stage 3 finalization: submitted commit is not an ancestor of checkout HEAD." >&2
  exit 10
fi
if ! git diff --quiet "$GIT_COMMIT" HEAD -- \
  src/validate_gemma4_stage.py \
  jobs/sge/gemma4_12b_stage3_finalize.sh \
  jobs/sync_outputs.sh; then
  echo "Refusing 12B Stage 3 finalization: critical implementation changed after submission." >&2
  exit 11
fi
if [[ ! -f "$SUCCESS_SENTINEL" ]]; then
  echo "Refusing 12B Stage 3 finalization: missing success sentinel: $SUCCESS_SENTINEL" >&2
  exit 20
fi

python -m src.validate_gemma4_stage \
  --stage 3 \
  --model google/gemma-4-12B-it \
  --label gemma4_12b \
  --vectors-subdir concept_vectors_gemma4_12b \
  --expected-layers 48 \
  --expected-d-model 3840

bash jobs/sync_outputs.sh "$REPO_PATH"

mkdir -p "$(dirname "$FINAL_SENTINEL")"
sentinel_tmp="${FINAL_SENTINEL}.tmp.${JOB_ID:-$$}"
: > "$sentinel_tmp"
mv "$sentinel_tmp" "$FINAL_SENTINEL"
echo "[success] Gemma 4 12B Stage 3 validated and synchronized"
