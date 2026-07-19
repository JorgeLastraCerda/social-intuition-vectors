#!/bin/bash
# Run one independent calibrated-steering pilot on an RTX PRO 6000.
set -euo pipefail

: "${MODEL_KEY:?MODEL_KEY is required}"
: "${REPO_PATH:?REPO_PATH is required}"
: "${GIT_COMMIT:?GIT_COMMIT is required}"
: "${SUCCESS_SENTINEL:?SUCCESS_SENTINEL is required}"

cd "$REPO_PATH"
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH="$REPO_PATH"
module load conda  # ADJUST if SCCKN changes the module name.

case "$MODEL_KEY" in
  gemma3_12b)
    ENV_NAME=wc-tl
    CONFIG_PATH=config/config.yaml
    LABEL=gemma3_12b_calibrated
    COMMAND=(-m src.dense_steering --config "$CONFIG_PATH" \
      --vectors-subdir concept_vectors --label "$LABEL" --vector-kind raw \
      --include-cross-axis --n-random-directions 99 \
      --strengths=-0.1,-0.05,0,0.05,0.1 --prompt-format raw \
      --control-scale sd_matched --interventions additive,norm_preserving)
    ;;
  gemma4_12b)
    ENV_NAME=wc-tl-g4
    CONFIG_PATH=config/gemma4_12b.yaml
    LABEL=gemma4_12b_calibrated
    COMMAND=(-m src.dense_steering --config "$CONFIG_PATH" \
      --vectors-subdir concept_vectors_gemma4_12b --label "$LABEL" --vector-kind raw \
      --include-cross-axis --n-random-directions 99 \
      --strengths=-0.1,-0.05,0,0.05,0.1 --prompt-format native-chat \
      --control-scale sd_matched --interventions additive,norm_preserving)
    ;;
  gemma4_26b_a4b)
    ENV_NAME=wc-tl-g4
    CONFIG_PATH=config/gemma4_26b_a4b.yaml
    LABEL=gemma4_26b_a4b_calibrated_scckn_rtx6000
    CHECKPOINT_DIR="$(dirname "$SUCCESS_SENTINEL")/checkpoint"
    checkpoint_args=(--checkpoint-dir "$CHECKPOINT_DIR")
    if [[ -f "$CHECKPOINT_DIR/manifest.json" ]]; then
      checkpoint_commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"]["git_commit"])' "$CHECKPOINT_DIR/manifest.json")
      checkpoint_args+=(--resume --checkpoint-origin-commit "$checkpoint_commit")
    fi
    COMMAND=(-m src.dense_steering --config "$CONFIG_PATH" \
      --vectors-subdir concept_vectors_gemma4_26b_a4b --label "$LABEL" --vector-kind raw \
      --include-cross-axis --n-random-directions 99 \
      --strengths=-0.1,-0.05,0,0.05,0.1 --prompt-format native-chat \
      --control-scale sd_matched --interventions additive,norm_preserving \
      "${checkpoint_args[@]}")
    ;;
  gemma4_31b)
    ENV_NAME=wc-tl-g4
    CONFIG_PATH=config/gemma4_31b.yaml
    LABEL=gemma4_31b_calibrated_scckn_rtx6000
    CHECKPOINT_DIR="$(dirname "$SUCCESS_SENTINEL")/checkpoint"
    checkpoint_args=(--checkpoint-dir "$CHECKPOINT_DIR")
    if [[ -f "$CHECKPOINT_DIR/manifest.json" ]]; then
      checkpoint_commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"]["git_commit"])' "$CHECKPOINT_DIR/manifest.json")
      checkpoint_args+=(--resume --checkpoint-origin-commit "$checkpoint_commit")
    fi
    COMMAND=(-m src.dense_steering --config "$CONFIG_PATH" \
      --vectors-subdir concept_vectors_gemma4_31b --label "$LABEL" --vector-kind raw \
      --include-cross-axis --n-random-directions 99 \
      --strengths=-0.1,-0.05,0,0.05,0.1 --prompt-format native-chat \
      --control-scale sd_matched --interventions additive,norm_preserving \
      "${checkpoint_args[@]}")
    ;;
  qwen36_27b)
    ENV_NAME=wc-qwen36-hf
    CONFIG_PATH=config/qwen36_27b.yaml
    LABEL=qwen36_27b_calibrated_topicfix_scckn_rtx6000
    CHECKPOINT_DIR="$(dirname "$SUCCESS_SENTINEL")/checkpoint"
    checkpoint_args=(--checkpoint-dir "$CHECKPOINT_DIR")
    if [[ -f "$CHECKPOINT_DIR/manifest.json" ]]; then
      checkpoint_commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"]["git_commit"])' "$CHECKPOINT_DIR/manifest.json")
      checkpoint_args+=(--resume --checkpoint-origin-commit "$checkpoint_commit")
    fi
    COMMAND=(-m src.qwen36_calibrated_steering --config "$CONFIG_PATH" \
      --label "$LABEL" --n-random-directions 99 "${checkpoint_args[@]}")
    ;;
  qwen36_35b_a3b)
    ENV_NAME=wc-qwen36-hf
    CONFIG_PATH=config/qwen36_35b_a3b.yaml
    LABEL=qwen36_35b_a3b_calibrated_scckn_rtx6000
    CHECKPOINT_DIR="$(dirname "$SUCCESS_SENTINEL")/checkpoint"
    checkpoint_args=(--checkpoint-dir "$CHECKPOINT_DIR")
    if [[ -f "$CHECKPOINT_DIR/manifest.json" ]]; then
      checkpoint_commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"]["git_commit"])' "$CHECKPOINT_DIR/manifest.json")
      checkpoint_args+=(--resume --checkpoint-origin-commit "$checkpoint_commit")
    fi
    COMMAND=(-m src.qwen36_calibrated_steering --config "$CONFIG_PATH" \
      --label "$LABEL" --n-random-directions 99 "${checkpoint_args[@]}")
    ;;
  *) echo "Unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac
PYTHON=(conda run --no-capture-output -n "$ENV_NAME" python)

if ! git merge-base --is-ancestor "$GIT_COMMIT" HEAD; then
  echo "Submitted commit is not an ancestor of SCCKN HEAD." >&2
  exit 10
fi
if ! git diff --quiet "$GIT_COMMIT" HEAD -- \
  "$CONFIG_PATH" src/steering_calibration.py src/dense_steering.py \
  src/steering_checkpoint.py \
  src/qwen36_calibrated_steering.py src/validate_calibrated_steering.py \
  jobs/sge/calibrated_steering_run.sh; then
  echo "Critical calibrated-steering implementation changed after submission." >&2
  exit 11
fi
if [[ -e "$SUCCESS_SENTINEL" ]]; then
  echo "Success sentinel already exists: $SUCCESS_SENTINEL" >&2
  exit 12
fi

"${PYTHON[@]}" - <<'PY'
import torch

if torch.cuda.device_count() != 1:
    raise SystemExit("Exactly one visible CUDA GPU is required.")
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info(0)
print(f"[runtime] gpu={name!r} free={free/1024**3:.2f} total={total/1024**3:.2f} GiB")
if "RTX PRO 6000" not in name:
    raise SystemExit(f"Expected RTX PRO 6000, got {name!r}.")
PY

"${PYTHON[@]}" -m src.validate_calibrated_steering \
  --config "$CONFIG_PATH" --label "$LABEL" --require-absent
echo "[start] model=$MODEL_KEY job=${JOB_ID:-unknown} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${PYTHON[@]}" "${COMMAND[@]}"
"${PYTHON[@]}" -m src.validate_calibrated_steering \
  --config "$CONFIG_PATH" --label "$LABEL"

mkdir -p "$(dirname "$SUCCESS_SENTINEL")"
sentinel_tmp="${SUCCESS_SENTINEL}.tmp.${JOB_ID:-$$}"
: > "$sentinel_tmp"
mv "$sentinel_tmp" "$SUCCESS_SENTINEL"
echo "[success] $MODEL_KEY -> $SUCCESS_SENTINEL"
