#!/bin/bash
# Run one write-once Qwen3.6 calibrated-steering job on the CCU H100.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash jobs/ccu/run_qwen36_calibrated.sh {27b|35b_a3b}" >&2
  exit 2
fi

MODEL_KEY=$1
REPO_PATH=${REPO_PATH:-/home/jovyan/work/normalcy-axis}
VENV_PATH=${VENV_PATH:-/home/jovyan/.venvs/normalcy-qwen36-cu124}
STATE_ROOT=${STATE_ROOT:-/home/jovyan/work/normalcy-qwen36-state}
export HF_HOME=${HF_HOME:-/home/jovyan/work/hf_cache}
export PYTHONPATH="$REPO_PATH"
export CUDA_VISIBLE_DEVICES=0
PYTHON="$VENV_PATH/bin/python"

case "$MODEL_KEY" in
  27b)
    CONFIG_PATH=config/qwen36_27b.yaml
    LABEL=qwen36_27b_calibrated_topicfix_ccu_h100
    MIN_FREE_GIB=60
    ;;
  35b_a3b)
    CONFIG_PATH=config/qwen36_35b_a3b.yaml
    LABEL=qwen36_35b_a3b_calibrated_ccu_h100
    MIN_FREE_GIB=72
    ;;
  *) echo "Unknown model key: $MODEL_KEY" >&2; exit 2 ;;
esac

cd "$REPO_PATH"
mkdir -p "$STATE_ROOT/sentinels" results/logs
SENTINEL="$STATE_ROOT/sentinels/$LABEL.success"

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
"$PYTHON" -c 'import sys; free=float(sys.argv[1]); need=float(sys.argv[2]); assert free >= need, f"Only {free:.2f} GiB free; need {need:.2f}."' "$FREE_GIB" "$MIN_FREE_GIB"
"$PYTHON" - <<'PY'
import importlib.metadata
import importlib.util

version = importlib.metadata.version("transformers")
if version != "5.14.1":
    raise SystemExit(f"Expected transformers 5.14.1, got {version}.")
if importlib.util.find_spec("transformer_lens") is not None:
    raise SystemExit("TransformerLens must be absent from the native-HF Qwen environment.")
PY
echo "[hardware] gpu=$GPU_NAME free=$FREE_GIB total=$TOTAL_GIB GiB"

if [[ -f "$SENTINEL" ]]; then
  "$PYTHON" -m src.validate_calibrated_steering \
    --config "$CONFIG_PATH" --label "$LABEL"
  echo "[complete] validated existing sentinel $SENTINEL"
  exit 0
fi

"$PYTHON" -m src.validate_calibrated_steering \
  --config "$CONFIG_PATH" --label "$LABEL" --require-absent
"$PYTHON" -m src.qwen36_calibrated_steering \
  --config "$CONFIG_PATH" --label "$LABEL" --n-random-directions 99
"$PYTHON" -m src.validate_calibrated_steering \
  --config "$CONFIG_PATH" --label "$LABEL"

sentinel_tmp="$SENTINEL.tmp.$$"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$sentinel_tmp"
mv "$sentinel_tmp" "$SENTINEL"
echo "[success] $LABEL"
