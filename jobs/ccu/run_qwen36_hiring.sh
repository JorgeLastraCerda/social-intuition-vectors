#!/bin/bash
# Run one resumable native-HF Qwen3.6 hiring task on the CCU H100.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash jobs/ccu/run_qwen36_hiring.sh {27b|35b_a3b} {audit|neutral|local|broad|denoised_local|local_full282|broad_full282|denoised_local_full282}" >&2
  exit 2
fi

MODEL_KEY=$1
TASK=$2
REPO_PATH=${REPO_PATH:-/home/jovyan/work/normalcy-axis}
VENV_PATH=${VENV_PATH:-/home/jovyan/.venvs/normalcy-qwen36-cu124}
STATE_ROOT=${STATE_ROOT:-/home/jovyan/work/normalcy-qwen36-hiring-state}
export HF_HOME=${HF_HOME:-/home/jovyan/work/hf_cache}
export PYTHONPATH="$REPO_PATH"
export CUDA_VISIBLE_DEVICES=0
PYTHON="$VENV_PATH/bin/python"

case "$MODEL_KEY" in
  27b) CONFIG_PATH=config/qwen36_27b.yaml; BASE_LABEL=qwen36_27b; MIN_FREE_GIB=60 ;;
  35b_a3b) CONFIG_PATH=config/qwen36_35b_a3b.yaml; BASE_LABEL=qwen36_35b_a3b; MIN_FREE_GIB=72 ;;
  *) echo "Unknown model key: $MODEL_KEY" >&2; exit 2 ;;
esac
case "$TASK" in
  audit) RUN_TASK=audit; LABEL=$BASE_LABEL; EXTRA_ARGS=() ;;
  neutral) RUN_TASK=neutral; LABEL="${BASE_LABEL}_neutral"; EXTRA_ARGS=() ;;
  local|broad|denoised_local)
    RUN_TASK=steering
    LABEL="${BASE_LABEL}_${TASK}"
    EXTRA_ARGS=(--regime "$TASK" --n-names 60)
    ;;
  local_full282|broad_full282|denoised_local_full282)
    RUN_TASK=steering
    REGIME=${TASK%_full282}
    LABEL="${BASE_LABEL}_${TASK}"
    EXTRA_ARGS=(--regime "$REGIME" --n-names 282)
    ;;
  *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac

cd "$REPO_PATH"
mkdir -p "$STATE_ROOT/checkpoints" "$STATE_ROOT/sentinels" "$STATE_ROOT/logs"
CHECKPOINT_DIR="$STATE_ROOT/checkpoints/$LABEL"
SENTINEL="$STATE_ROOT/sentinels/$LABEL.success"

case "$RUN_TASK" in
  audit)
    OUTPUT_PATHS=("results/tables/hiring_audit_${LABEL}.csv" "results/logs/hiring_probe_vs_human_${LABEL}.json")
    ;;
  neutral)
    OUTPUT_PATHS=("data/processed/concept_vectors_${BASE_LABEL}/X_neutral.npy" "data/processed/concept_vectors_${BASE_LABEL}/neutral_meta.json")
    ;;
  steering)
    OUTPUT_PATHS=("results/tables/hiring_steering_raw_${LABEL}.csv" "results/logs/hiring_steering_${LABEL}.json")
    ;;
esac

publish_sentinel() {
  local sentinel_tmp="$SENTINEL.tmp.$$"
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$sentinel_tmp"
  mv "$sentinel_tmp" "$SENTINEL"
}

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
[[ "$GPU_NAME" == *H100* ]] || { echo "Expected H100, got $GPU_NAME" >&2; exit 30; }
"$PYTHON" -c 'import sys; assert float(sys.argv[1]) >= float(sys.argv[2]), f"Only {float(sys.argv[1]):.2f} GiB free; need {float(sys.argv[2]):.2f}."' "$FREE_GIB" "$MIN_FREE_GIB"
"$PYTHON" - <<'PY'
import importlib.metadata, importlib.util
assert importlib.metadata.version("transformers") == "5.14.1"
assert importlib.util.find_spec("transformer_lens") is None
PY
echo "[hardware] gpu=$GPU_NAME free=$FREE_GIB total=$TOTAL_GIB GiB"

if [[ -f "$SENTINEL" ]]; then
  "$PYTHON" -m src.validate_qwen36_hiring --config "$CONFIG_PATH" \
    --task "$RUN_TASK" --label "$LABEL"
  echo "[complete] $SENTINEL"
  exit 0
fi
existing_outputs=0
for output_path in "${OUTPUT_PATHS[@]}"; do
  [[ -f "$output_path" ]] && ((existing_outputs += 1))
done
if ((existing_outputs == ${#OUTPUT_PATHS[@]})); then
  "$PYTHON" -m src.validate_qwen36_hiring --config "$CONFIG_PATH" \
    --task "$RUN_TASK" --label "$LABEL"
  publish_sentinel
  echo "[recovered] validated published outputs and created $SENTINEL"
  exit 0
fi
"$PYTHON" -m src.validate_qwen36_hiring --config "$CONFIG_PATH" \
  --task "$RUN_TASK" --label "$LABEL" --require-absent

checkpoint_args=(--checkpoint-dir "$CHECKPOINT_DIR")
if [[ -f "$CHECKPOINT_DIR/manifest.json" ]]; then
  checkpoint_commit=$(
    "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"]["git_commit"])' \
      "$CHECKPOINT_DIR/manifest.json"
  )
  checkpoint_args+=(--resume --checkpoint-origin-commit "$checkpoint_commit")
fi
"$PYTHON" -m src.qwen36_hiring --config "$CONFIG_PATH" --task "$RUN_TASK" \
  --label "$LABEL" "${EXTRA_ARGS[@]}" "${checkpoint_args[@]}"
"$PYTHON" -m src.validate_qwen36_hiring --config "$CONFIG_PATH" \
  --task "$RUN_TASK" --label "$LABEL"

publish_sentinel
echo "[success] $LABEL"
