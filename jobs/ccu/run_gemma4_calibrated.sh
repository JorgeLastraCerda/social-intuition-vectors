#!/bin/bash
# Run one Gemma 4 smoke gate and resumable calibrated steering job on CCU H100.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash jobs/ccu/run_gemma4_calibrated.sh {12b|26b_a4b|31b}" >&2
  exit 2
fi
MODEL_KEY=$1
REPO_PATH=${REPO_PATH:-/home/jovyan/work/normalcy-axis}
VENV_PATH=${VENV_PATH:-/home/jovyan/.venvs/normalcy-gemma4-cu124}
STATE_ROOT=${STATE_ROOT:-/home/jovyan/work/normalcy-gemma4-state}
export HF_HOME=${HF_HOME:-/home/jovyan/work/hf_cache}
export PYTHONPATH="$REPO_PATH"
export CUDA_VISIBLE_DEVICES=0
PYTHON="$VENV_PATH/bin/python"

case "$MODEL_KEY" in
  12b)
    CONFIG_PATH=config/gemma4_12b.yaml
    VECTORS_SUBDIR=concept_vectors_gemma4_12b
    LABEL=gemma4_12b_calibrated_ccu_h100
    ;;
  26b_a4b)
    CONFIG_PATH=config/gemma4_26b_a4b.yaml
    VECTORS_SUBDIR=concept_vectors_gemma4_26b_a4b
    LABEL=gemma4_26b_a4b_calibrated_ccu_h100
    ;;
  31b)
    CONFIG_PATH=config/gemma4_31b.yaml
    VECTORS_SUBDIR=concept_vectors_gemma4_31b
    LABEL=gemma4_31b_calibrated_ccu_h100
    ;;
  *) echo "Unknown model key: $MODEL_KEY" >&2; exit 2 ;;
esac

cd "$REPO_PATH"
mkdir -p "$STATE_ROOT/checkpoints" "$STATE_ROOT/sentinels" results/logs
CHECKPOINT_DIR="$STATE_ROOT/checkpoints/$LABEL"
SENTINEL="$STATE_ROOT/sentinels/$LABEL.success"
SMOKE_PATH="results/logs/smoke_${LABEL}.json"

read -r GPU_NAME FREE_GIB TOTAL_GIB < <("$PYTHON" - <<'PY'
import torch
if torch.cuda.device_count() != 1:
    raise SystemExit("Exactly one visible CUDA GPU is required.")
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info(0)
print(name.replace(" ", "_"), free / 1024**3, total / 1024**3)
PY
)
GPU_NAME=${GPU_NAME//_/ }
if [[ "$GPU_NAME" != *H100* ]]; then
  echo "Expected H100, got $GPU_NAME" >&2
  exit 30
fi
MIN_FREE=$($PYTHON -c 'import sys; from src.utils.config import load_config; print(load_config(sys.argv[1]).smoke.min_free_vram_gib)' "$CONFIG_PATH")
"$PYTHON" -c 'import sys; free=float(sys.argv[1]); need=float(sys.argv[2]); assert free >= need, f"Only {free:.2f} GiB free; need {need:.2f}."' "$FREE_GIB" "$MIN_FREE"
echo "[hardware] gpu=$GPU_NAME free=$FREE_GIB total=$TOTAL_GIB GiB"

if [[ -f "$SMOKE_PATH" ]]; then
  echo "[resume] validating existing smoke $SMOKE_PATH"
else
  "$PYTHON" smoke_tests/gemma4_transformerlens/smoke_test_bridge.py \
    --config "$CONFIG_PATH" --max-logit-diff 0.02 --output "$SMOKE_PATH"
fi
"$PYTHON" -m src.validate_ccu_gemma4 --config "$CONFIG_PATH" \
  --smoke-path "$SMOKE_PATH" --total-vram-gib "$TOTAL_GIB"

if [[ -f "$SENTINEL" ]]; then
  "$PYTHON" -m src.validate_calibrated_steering \
    --config "$CONFIG_PATH" --label "$LABEL"
  echo "[complete] validated existing sentinel $SENTINEL"
  exit 0
fi

checkpoint_args=(--checkpoint-dir "$CHECKPOINT_DIR")
if [[ -f "$CHECKPOINT_DIR/manifest.json" ]]; then checkpoint_args+=(--resume); fi
if [[ ! -f "$CHECKPOINT_DIR/manifest.json" ]]; then
  "$PYTHON" -m src.validate_calibrated_steering \
    --config "$CONFIG_PATH" --label "$LABEL" --require-absent
fi
"$PYTHON" -m src.dense_steering --config "$CONFIG_PATH" \
  --vectors-subdir "$VECTORS_SUBDIR" --label "$LABEL" --vector-kind raw \
  --include-cross-axis --n-random-directions 99 \
  --strengths=-0.1,-0.05,0,0.05,0.1 --prompt-format native-chat \
  --control-scale sd_matched --interventions additive,norm_preserving \
  "${checkpoint_args[@]}"
"$PYTHON" -m src.validate_calibrated_steering \
  --config "$CONFIG_PATH" --label "$LABEL"
sentinel_tmp="$SENTINEL.tmp.$$"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$sentinel_tmp"
mv "$sentinel_tmp" "$SENTINEL"
echo "[success] $LABEL"
