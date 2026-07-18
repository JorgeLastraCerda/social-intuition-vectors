#!/bin/bash
# CPU-only validator for all three Stage 3B jobs. Output sync happens post-qacct.
#$ -q scc
#$ -l h_rt=00:20:00
#$ -l h_vmem=4G
#$ -pe smp 1
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${REPO_PATH:?REPO_PATH is required}"
: "${GIT_COMMIT:?GIT_COMMIT is required}"
: "${SENTINEL_12B:?SENTINEL_12B is required}"
: "${SENTINEL_26B:?SENTINEL_26B is required}"
: "${SENTINEL_31B:?SENTINEL_31B is required}"
: "${FINAL_SENTINEL:?FINAL_SENTINEL is required}"

module load conda  # ADJUST
conda activate wc-tl-g4
export PYTHONPATH="$REPO_PATH"
cd "$REPO_PATH"

if ! git merge-base --is-ancestor "$GIT_COMMIT" HEAD; then
  echo "Refusing Stage 3B finalization: submitted commit is not an ancestor." >&2
  exit 10
fi
for sentinel in "$SENTINEL_12B" "$SENTINEL_26B" "$SENTINEL_31B"; do
  [[ -f "$sentinel" ]] || { echo "Missing Stage 3B sentinel: $sentinel" >&2; exit 20; }
done

python -m src.validate_gemma4_stage --stage 3 --analysis-profile stage3b \
  --model google/gemma-4-12B-it --label stage3b_gemma4_12b_l40 \
  --vectors-subdir concept_vectors_gemma4_12b --expected-layers 48 --expected-d-model 3840
python -m src.validate_gemma4_stage --stage 3 --analysis-profile stage3b \
  --model google/gemma-4-26B-A4B-it --label stage3b_gemma4_26b_a4b \
  --vectors-subdir concept_vectors_gemma4_26b_a4b --expected-layers 30 --expected-d-model 2816
python -m src.validate_gemma4_stage --stage 3 --analysis-profile stage3b \
  --model google/gemma-4-31B-it --label stage3b_gemma4_31b \
  --vectors-subdir concept_vectors_gemma4_31b --expected-layers 60 --expected-d-model 5376

mkdir -p "$(dirname "$FINAL_SENTINEL")"
sentinel_tmp="${FINAL_SENTINEL}.tmp.${JOB_ID:-$$}"
: > "$sentinel_tmp"
mv "$sentinel_tmp" "$FINAL_SENTINEL"
echo "[success] all Gemma 4 Stage 3B outputs validated; run provenance postflight"
