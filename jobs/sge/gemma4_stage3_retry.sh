#!/bin/bash
# Run one independent Gemma 4 Stage 3 retry on a single RTX 6000 GPU.
# Submit only through submit_gemma4_stage3_retry.sh.
# ADJUST: module and scratch paths if SCCKN changes its runtime environment.
#$ -q gpu@scc214
#$ -l h_rt=12:00:00
#$ -l h_vmem=96G
#$ -pe smp 2
#$ -l gpu=1
#$ -l rtx_6000=1
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${LABEL:?LABEL is required}"
: "${VECTORS_SUBDIR:?VECTORS_SUBDIR is required}"
: "${EXPECTED_LAYERS:?EXPECTED_LAYERS is required}"
: "${EXPECTED_D_MODEL:?EXPECTED_D_MODEL is required}"
: "${SUCCESS_SENTINEL:?SUCCESS_SENTINEL is required}"
: "${REPO_PATH:?REPO_PATH is required}"
: "${GIT_COMMIT:?GIT_COMMIT is required}"

module load conda  # ADJUST
conda activate wc-tl-g4
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH="$REPO_PATH"
cd "$REPO_PATH"

if ! git merge-base --is-ancestor "$GIT_COMMIT" HEAD; then
  echo "Refusing Stage 3 retry: submitted commit is not an ancestor of checkout HEAD." >&2
  exit 10
fi
if ! git diff --quiet "$GIT_COMMIT" HEAD -- \
  config/config.yaml \
  src/layer_sweep.py \
  src/utils/model_loader.py \
  src/validate_gemma4_stage.py \
  jobs/sge/gemma4_stage3_retry.sh; then
  echo "Refusing Stage 3 retry: critical implementation changed after submission." >&2
  exit 11
fi
if [[ "$(python -c 'import torch; print(torch.cuda.device_count())')" != "1" ]]; then
  echo "Refusing Stage 3 retry: Grid Engine did not expose exactly one GPU." >&2
  exit 12
fi
if [[ -e "$SUCCESS_SENTINEL" ]]; then
  echo "Refusing Stage 3 retry: success sentinel already exists: $SUCCESS_SENTINEL" >&2
  exit 13
fi

python - <<'PY'
import torch

print(f"[runtime] host GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(
    f"[runtime] visible GPUs={torch.cuda.device_count()} "
    f"total_memory={torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB",
    flush=True,
)
PY

common_args=(
  --model "$MODEL_NAME"
  --label "$LABEL"
  --vectors-subdir "$VECTORS_SUBDIR"
  --expected-layers "$EXPECTED_LAYERS"
  --expected-d-model "$EXPECTED_D_MODEL"
)

echo "[preflight] validating Stage 1 and Stage 2 inputs for $LABEL" >&2
python -m src.validate_gemma4_stage --stage 1 "${common_args[@]}"
python -m src.validate_gemma4_stage --stage 2 "${common_args[@]}"
python -m src.validate_gemma4_stage --stage 3 "${common_args[@]}" --require-absent

echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ) model=$MODEL_NAME job=${JOB_ID:-unknown}" >&2
python src/layer_sweep.py --model "$MODEL_NAME" --label "$LABEL"
python -m src.validate_gemma4_stage --stage 3 "${common_args[@]}"

mkdir -p "$(dirname "$SUCCESS_SENTINEL")"
sentinel_tmp="${SUCCESS_SENTINEL}.tmp.${JOB_ID:-$$}"
: > "$sentinel_tmp"
mv "$sentinel_tmp" "$SUCCESS_SENTINEL"
echo "[success] $(date -u +%Y-%m-%dT%H:%M:%SZ) $LABEL Stage 3 -> $SUCCESS_SENTINEL"
