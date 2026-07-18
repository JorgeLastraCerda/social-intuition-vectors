#!/bin/bash
# Enhanced Gemma 4 Stage 3B sweep on one GPU. Submit only through its submitter.
# ADJUST: module and scratch paths if SCCKN changes its runtime environment.
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${LABEL:?LABEL is required}"
: "${SOURCE_LABEL:?SOURCE_LABEL is required}"
: "${VECTORS_SUBDIR:?VECTORS_SUBDIR is required}"
: "${EXPECTED_LAYERS:?EXPECTED_LAYERS is required}"
: "${EXPECTED_D_MODEL:?EXPECTED_D_MODEL is required}"
: "${EXPECTED_GPU_KIND:?EXPECTED_GPU_KIND is required}"
: "${SUCCESS_SENTINEL:?SUCCESS_SENTINEL is required}"
: "${REPO_PATH:?REPO_PATH is required}"
: "${GIT_COMMIT:?GIT_COMMIT is required}"

module load conda  # ADJUST
conda activate wc-tl-g4
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH="$REPO_PATH"
cd "$REPO_PATH"

if ! git merge-base --is-ancestor "$GIT_COMMIT" HEAD; then
  echo "Refusing Stage 3B: submitted commit is not an ancestor of checkout HEAD." >&2
  exit 10
fi
if ! git diff --quiet "$GIT_COMMIT" HEAD -- \
  config/config.yaml src/layer_sweep.py src/validate_probes.py \
  src/validate_gemma4_stage.py src/utils/model_loader.py \
  jobs/sge/gemma4_stage3b.sh; then
  echo "Refusing Stage 3B: critical implementation changed after submission." >&2
  exit 11
fi
if [[ "$(python -c 'import torch; print(torch.cuda.device_count())')" != "1" ]]; then
  echo "Refusing Stage 3B: Grid Engine did not expose exactly one GPU." >&2
  exit 12
fi
EXPECTED_GPU_KIND="$EXPECTED_GPU_KIND" python - <<'PY'
import os
import torch

name = torch.cuda.get_device_name(0)
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
print(f"[runtime] host GPU: {name}", flush=True)
print(
    f"[runtime] free_memory={free_bytes / 1024**3:.2f} GiB "
    f"total_memory={total_bytes / 1024**3:.2f} GiB",
    flush=True,
)
expected = {
    "L40": "NVIDIA L40",
    "RTX_PRO_6000": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
}[os.environ["EXPECTED_GPU_KIND"]]
if name != expected:
    raise SystemExit(
        f"Refusing Stage 3B: expected {expected!r}, got {name!r}."
    )
PY

source_args=(
  --model "$MODEL_NAME" --label "$SOURCE_LABEL"
  --vectors-subdir "$VECTORS_SUBDIR"
  --expected-layers "$EXPECTED_LAYERS" --expected-d-model "$EXPECTED_D_MODEL"
)
output_args=(
  --model "$MODEL_NAME" --label "$LABEL"
  --vectors-subdir "$VECTORS_SUBDIR"
  --expected-layers "$EXPECTED_LAYERS" --expected-d-model "$EXPECTED_D_MODEL"
  --analysis-profile stage3b
)
python -m src.validate_gemma4_stage --stage 1 "${source_args[@]}"
python -m src.validate_gemma4_stage --stage 2 "${source_args[@]}"
python -m src.validate_gemma4_stage --stage 3 "${output_args[@]}" --require-absent

echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ) model=$MODEL_NAME label=$LABEL job=${JOB_ID:-unknown}" >&2
python src/layer_sweep.py \
  --model "$MODEL_NAME" --label "$LABEL" \
  --validation-profile stage3b --n-bootstrap 1000 --bootstrap-batch-size 100 \
  --git-commit "$GIT_COMMIT"
python -m src.validate_gemma4_stage --stage 3 "${output_args[@]}"

mkdir -p "$(dirname "$SUCCESS_SENTINEL")"
sentinel_tmp="${SUCCESS_SENTINEL}.tmp.${JOB_ID:-$$}"
: > "$sentinel_tmp"
mv "$sentinel_tmp" "$SUCCESS_SENTINEL"
echo "[success] $(date -u +%Y-%m-%dT%H:%M:%SZ) $LABEL -> $SUCCESS_SENTINEL"
