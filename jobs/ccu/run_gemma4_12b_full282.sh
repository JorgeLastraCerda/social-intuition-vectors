#!/bin/bash
# Run one write-once Gemma 4 12B full-282 hiring-steering regime on CCU H100.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash jobs/ccu/run_gemma4_12b_full282.sh {local|broad|denoised}" >&2
  exit 2
fi
REGIME=$1
[[ "$REGIME" =~ ^(local|broad|denoised)$ ]] || { echo "Unknown regime: $REGIME" >&2; exit 2; }

REPO_PATH=${REPO_PATH:-/home/jovyan/work/normalcy-axis}
VENV_PATH=${VENV_PATH:-/home/jovyan/.venvs/normalcy-gemma4-cu124}
STATE_ROOT=${STATE_ROOT:-/home/jovyan/work/normalcy-gemma4-remaining}
export HF_HOME=${HF_HOME:-/home/jovyan/work/hf_cache}
export PYTHONPATH="$REPO_PATH"
export CUDA_VISIBLE_DEVICES=0
PYTHON="$VENV_PATH/bin/python"
CONFIG_PATH=config/gemma4_12b.yaml

case "$REGIME" in
  local)
    RUN_NAME=hiring_local
    OUTPUT_SUFFIX=local
    VECTOR_KIND=raw
    STRENGTHS=-0.1,-0.05,0,0.05,0.1
    ;;
  broad)
    RUN_NAME=hiring_broad
    OUTPUT_SUFFIX=broad
    VECTOR_KIND=raw
    STRENGTHS=-0.5,-0.25,0,0.25,0.5
    ;;
  denoised)
    RUN_NAME=hiring_denoised
    OUTPUT_SUFFIX=denoised_local
    VECTOR_KIND=denoised
    STRENGTHS=-0.1,-0.05,0,0.05,0.1
    ;;
esac

cd "$REPO_PATH"
mkdir -p "$STATE_ROOT/sentinels" "$STATE_ROOT/logs"
SENTINEL="$STATE_ROOT/sentinels/gemma4_12b_${OUTPUT_SUFFIX}_full282.success"
LABEL="gemma4_12b_${OUTPUT_SUFFIX}_full282"

if [[ -f "$SENTINEL" ]]; then
  "$PYTHON" -m src.validate_gemma4_remaining \
    --config "$CONFIG_PATH" --run "$RUN_NAME" --full282
  echo "[complete] validated existing sentinel $SENTINEL"
  exit 0
fi

read -r GPU_NAME FREE_GIB < <("$PYTHON" - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit("Exactly one visible CUDA GPU is required.")
name = torch.cuda.get_device_name(0)
free, _ = torch.cuda.mem_get_info(0)
print(name.replace(" ", "_"), free / 1024**3)
PY
)
GPU_NAME=${GPU_NAME//_/ }
if [[ "$GPU_NAME" != *H100* ]]; then
  echo "Expected H100, got $GPU_NAME" >&2
  exit 30
fi
MIN_FREE=$($PYTHON -c 'import sys; from src.utils.config import load_config; print(load_config(sys.argv[1]).smoke.min_free_vram_gib)' "$CONFIG_PATH")
"$PYTHON" -c 'import sys; assert float(sys.argv[1]) >= float(sys.argv[2]), f"Only {float(sys.argv[1]):.2f} GiB free; need {float(sys.argv[2]):.2f}."' "$FREE_GIB" "$MIN_FREE"
echo "[hardware] gpu=$GPU_NAME free=$FREE_GIB GiB"

"$PYTHON" -m src.validate_gemma4_remaining \
  --config "$CONFIG_PATH" --run "$RUN_NAME" --full282 --require-absent
"$PYTHON" -m src.hiring_steering --config "$CONFIG_PATH" \
  --vectors-subdir concept_vectors_gemma4_12b --label "$LABEL" \
  --strengths="$STRENGTHS" --vector-kind "$VECTOR_KIND" \
  --n-names 0 --prompt-format native-chat
"$PYTHON" -m src.summarize_hiring_steering summarize \
  --config "$CONFIG_PATH" --label "$LABEL" --n-boot 5000
"$PYTHON" -m src.validate_gemma4_remaining \
  --config "$CONFIG_PATH" --run "$RUN_NAME" --full282

sentinel_tmp="$SENTINEL.tmp.$$"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$sentinel_tmp"
mv "$sentinel_tmp" "$SENTINEL"
echo "[success] $LABEL"
