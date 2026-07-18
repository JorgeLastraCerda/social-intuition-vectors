#!/bin/bash
# Run the independent Gemma 4 12B Stage 3 retry on one SCCKN L40 GPU.
# Submit only through submit_gemma4_12b_stage3_retry.sh.
# ADJUST: module and scratch paths if SCCKN changes its runtime environment.
#$ -q gpu@scc192,gpu@scc213
#$ -l h_rt=01:00:00
#$ -l h_vmem=32G
#$ -pe smp 2
#$ -l gpu=1
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${SUCCESS_SENTINEL:?SUCCESS_SENTINEL is required}"
: "${REPO_PATH:?REPO_PATH is required}"
: "${GIT_COMMIT:?GIT_COMMIT is required}"

module load conda  # ADJUST
conda activate wc-tl-g4
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH="$REPO_PATH"
cd "$REPO_PATH"

if ! git merge-base --is-ancestor "$GIT_COMMIT" HEAD; then
  echo "Refusing 12B Stage 3 retry: submitted commit is not an ancestor of checkout HEAD." >&2
  exit 10
fi
if ! git diff --quiet "$GIT_COMMIT" HEAD -- \
  config/config.yaml \
  src/layer_sweep.py \
  src/utils/model_loader.py \
  src/validate_gemma4_stage.py \
  jobs/sge/gemma4_12b_stage3_retry.sh; then
  echo "Refusing 12B Stage 3 retry: critical implementation changed after submission." >&2
  exit 11
fi
if [[ "$(python -c 'import torch; print(torch.cuda.device_count())')" != "1" ]]; then
  echo "Refusing 12B Stage 3 retry: Grid Engine did not expose exactly one GPU." >&2
  exit 12
fi
if [[ -e "$SUCCESS_SENTINEL" ]]; then
  echo "Refusing 12B Stage 3 retry: success sentinel already exists: $SUCCESS_SENTINEL" >&2
  exit 13
fi

python - <<'PY'
import torch

name = torch.cuda.get_device_name(0)
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
free_gib = free_bytes / 1024**3
total_gib = total_bytes / 1024**3
print(f"[runtime] host GPU: {name}", flush=True)
print(
    f"[runtime] visible GPUs={torch.cuda.device_count()} "
    f"free_memory={free_gib:.2f} GiB total_memory={total_gib:.2f} GiB",
    flush=True,
)
if "L40" not in name:
    raise SystemExit(f"Refusing 12B Stage 3 retry: expected an L40, got {name!r}.")
if free_gib < 30.0:
    raise SystemExit(
        f"Refusing 12B Stage 3 retry: only {free_gib:.2f} GiB is free; "
        "at least 30 GiB is required."
    )
PY

common_args=(
  --model google/gemma-4-12B-it
  --label gemma4_12b
  --vectors-subdir concept_vectors_gemma4_12b
  --expected-layers 48
  --expected-d-model 3840
)

echo "[preflight] validating Stage 1 and Stage 2 inputs for gemma4_12b" >&2
python -m src.validate_gemma4_stage --stage 1 "${common_args[@]}"
python -m src.validate_gemma4_stage --stage 2 "${common_args[@]}"
python -m src.validate_gemma4_stage --stage 3 "${common_args[@]}" --require-absent

echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ) model=google/gemma-4-12B-it job=${JOB_ID:-unknown}" >&2
python src/layer_sweep.py --model google/gemma-4-12B-it --label gemma4_12b
python -m src.validate_gemma4_stage --stage 3 "${common_args[@]}"

mkdir -p "$(dirname "$SUCCESS_SENTINEL")"
sentinel_tmp="${SUCCESS_SENTINEL}.tmp.${JOB_ID:-$$}"
: > "$sentinel_tmp"
mv "$sentinel_tmp" "$SUCCESS_SENTINEL"
echo "[success] $(date -u +%Y-%m-%dT%H:%M:%SZ) gemma4_12b Stage 3 -> $SUCCESS_SENTINEL"
