#!/bin/bash
# Run one local-regime (raw or denoised) hiring-steering sweep for a native-TL
# model on the CCU H100. Closes the coverage gap where Llama-3.1-8B and
# Qwen3-14B only had the broad {+/-0.25,+/-0.50} sweep, never the narrow
# {+/-0.05,+/-0.10} regime the other seven models have.
#
# Uses native TransformerLens (config.model.backend = "transformer-lens"),
# matching the backend the original broad sweeps for these two models used
# (git 0e0547a), not the Bridge interface added later for Gemma 4.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash jobs/ccu/run_hiring_local_tl.sh {llama31_8b|qwen3_14b} {raw|denoised}" >&2
  exit 2
fi

MODEL_KEY=$1
KIND=$2
[[ "$MODEL_KEY" =~ ^(llama31_8b|qwen3_14b)$ ]] || { echo "Unknown model: $MODEL_KEY" >&2; exit 2; }
[[ "$KIND" =~ ^(raw|denoised)$ ]] || { echo "Unknown kind: $KIND" >&2; exit 2; }

REPO_PATH=${REPO_PATH:-/home/jovyan/work/normalcy-axis}
VENV_PATH=${VENV_PATH:-/home/jovyan/.venvs/normalcy-gemma4-cu124}
export HF_HOME=${HF_HOME:-/home/jovyan/work/hf_cache}
export PYTHONPATH="$REPO_PATH"
export CUDA_VISIBLE_DEVICES=0
PYTHON="$VENV_PATH/bin/python"

CONFIG_PATH="config/${MODEL_KEY}.yaml"
VECTORS_SUBDIR="concept_vectors_${MODEL_KEY}"
if [[ "$KIND" == "raw" ]]; then
  LABEL="${MODEL_KEY}_local"
  VECTOR_KIND=raw
else
  LABEL="${MODEL_KEY}_denoised_local"
  VECTOR_KIND=denoised
fi

cd "$REPO_PATH"
RAW_CSV="results/tables/hiring_steering_raw_${LABEL}.csv"

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
MIN_FREE=$("$PYTHON" -c 'import sys; from src.utils.config import load_config; print(load_config(sys.argv[1]).smoke.min_free_vram_gib)' "$CONFIG_PATH")
"$PYTHON" -c 'import sys; assert float(sys.argv[1]) >= float(sys.argv[2]), f"Only {float(sys.argv[1]):.2f} GiB free; need {float(sys.argv[2]):.2f}."' "$FREE_GIB" "$MIN_FREE"
echo "[hardware] gpu=$GPU_NAME free=$FREE_GIB GiB"

if [[ -f "$RAW_CSV" ]]; then
  echo "[skip] $RAW_CSV already exists" >&2
  exit 0
fi

"$PYTHON" -m src.hiring_steering --config "$CONFIG_PATH" \
  --vectors-subdir "$VECTORS_SUBDIR" --label "$LABEL" \
  --strengths=-0.1,-0.05,0,0.05,0.1 --vector-kind "$VECTOR_KIND" \
  --n-names 60 --prompt-format raw

"$PYTHON" -c "
import pandas as pd
df = pd.read_csv('$RAW_CSV')
assert len(df) == 600, f'expected 600 rows, got {len(df)}'
strengths = sorted(df[\"strength\"].unique())
expected = [-0.1, -0.05, 0.0, 0.05, 0.1]
assert strengths == expected, f'expected {expected}, got {strengths}'
print('[check] 600 rows, strengths', strengths)
"

"$PYTHON" -m src.summarize_hiring_steering summarize \
  --config "$CONFIG_PATH" --label "$LABEL" --n-boot 5000

echo "[success] $LABEL"
