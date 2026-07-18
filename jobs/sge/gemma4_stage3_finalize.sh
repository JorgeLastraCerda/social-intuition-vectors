#!/bin/bash
# Validate and synchronize both independent Gemma 4 Stage 3 retry outputs.
# This is a CPU-only job held on both GPU job IDs.
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
: "${SENTINEL_26B:?SENTINEL_26B is required}"
: "${SENTINEL_31B:?SENTINEL_31B is required}"
: "${FINAL_SENTINEL:?FINAL_SENTINEL is required}"

module load conda  # ADJUST
conda activate wc-tl-g4
export PYTHONPATH="$REPO_PATH"
cd "$REPO_PATH"

if ! git merge-base --is-ancestor "$GIT_COMMIT" HEAD; then
  echo "Refusing Stage 3 finalization: submitted commit is not an ancestor of checkout HEAD." >&2
  exit 10
fi
if ! git diff --quiet "$GIT_COMMIT" HEAD -- \
  src/validate_gemma4_stage.py \
  jobs/sge/gemma4_stage3_finalize.sh \
  jobs/sync_outputs.sh; then
  echo "Refusing Stage 3 finalization: critical implementation changed after submission." >&2
  exit 11
fi

missing=()
[[ -f "$SENTINEL_26B" ]] || missing+=("$SENTINEL_26B")
[[ -f "$SENTINEL_31B" ]] || missing+=("$SENTINEL_31B")
if ((${#missing[@]})); then
  printf 'Refusing Stage 3 finalization: missing success sentinel: %s\n' "${missing[@]}" >&2
  exit 20
fi

python -m src.validate_gemma4_stage \
  --stage 3 \
  --model google/gemma-4-26B-A4B-it \
  --label gemma4_26b_a4b \
  --vectors-subdir concept_vectors_gemma4_26b_a4b \
  --expected-layers 30 \
  --expected-d-model 2816
python -m src.validate_gemma4_stage \
  --stage 3 \
  --model google/gemma-4-31B-it \
  --label gemma4_31b \
  --vectors-subdir concept_vectors_gemma4_31b \
  --expected-layers 60 \
  --expected-d-model 5376

bash jobs/sync_outputs.sh "$REPO_PATH"

mkdir -p "$(dirname "$FINAL_SENTINEL")"
sentinel_tmp="${FINAL_SENTINEL}.tmp.${JOB_ID:-$$}"
: > "$sentinel_tmp"
mv "$sentinel_tmp" "$FINAL_SENTINEL"
echo "[success] both Gemma 4 Stage 3 retries validated and synchronized"
