#!/bin/bash
# Execute one independent pinned Gemma 4 remaining-test run.
set -euo pipefail

: "${RUN_NAME:?RUN_NAME is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${SUCCESS_SENTINEL:?SUCCESS_SENTINEL is required}"
: "${REPO_PATH:?REPO_PATH is required}"
: "${GIT_COMMIT:?GIT_COMMIT is required}"
: "${EXPECTED_GPU_KIND:=CPU}"
: "${FULL282:=0}"

module load conda  # ADJUST if SCCKN changes the module name.
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH="$REPO_PATH"
cd "$REPO_PATH"
PYTHON=(conda run --no-capture-output -n wc-tl-g4 python)

if ! git merge-base --is-ancestor "$GIT_COMMIT" HEAD; then
  echo "Refusing $RUN_NAME: submitted commit is not an ancestor of HEAD." >&2
  exit 10
fi
if ! git diff --quiet "$GIT_COMMIT" HEAD -- \
  "$CONFIG_PATH" src/dense_steering.py src/hiring_audit.py src/hiring_steering.py \
  src/summarize_hiring_steering.py src/extract_neutral.py src/denoise_vectors.py \
  src/validate_gemma4_remaining.py src/utils/model_loader.py \
  jobs/sge/gemma4_remaining_run.sh; then
  echo "Refusing $RUN_NAME: critical implementation changed after submission." >&2
  exit 11
fi
if [[ -e "$SUCCESS_SENTINEL" ]]; then
  echo "Refusing $RUN_NAME: sentinel exists: $SUCCESS_SENTINEL" >&2
  exit 12
fi

if [[ "$EXPECTED_GPU_KIND" != "CPU" ]]; then
  if [[ "$("${PYTHON[@]}" -c 'import torch; print(torch.cuda.device_count())')" != "1" ]]; then
    echo "Refusing $RUN_NAME: exactly one visible GPU is required." >&2
    exit 13
  fi
  "${PYTHON[@]}" - "$CONFIG_PATH" "$EXPECTED_GPU_KIND" <<'PY'
import sys
import torch
from src.utils.config import load_config

cfg = load_config(sys.argv[1])
expected = sys.argv[2]
name = torch.cuda.get_device_name(0)
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
free_gib = free_bytes / 1024**3
print(f"[runtime] gpu={name!r} free={free_gib:.2f} GiB total={total_bytes/1024**3:.2f} GiB")
if expected == "L40" and name != "NVIDIA L40":
    raise SystemExit(f"Expected exact NVIDIA L40, got {name!r}.")
if expected == "RTX_PRO_6000" and "RTX PRO 6000" not in name:
    raise SystemExit(f"Expected RTX PRO 6000, got {name!r}.")
if free_gib < cfg.smoke.min_free_vram_gib:
    raise SystemExit(
        f"Only {free_gib:.2f} GiB free; {cfg.smoke.min_free_vram_gib:.2f} GiB required."
    )
PY
fi

full_args=()
if [[ "$FULL282" == "1" ]]; then full_args=(--full282); fi
"${PYTHON[@]}" -m src.validate_gemma4_remaining \
  --config "$CONFIG_PATH" --run "$RUN_NAME" "${full_args[@]}" --require-absent

label=$("${PYTHON[@]}" -c \
  'import sys; from src.utils.config import load_config; print(load_config(sys.argv[1]).smoke.label.removesuffix("_pinned"))' \
  "$CONFIG_PATH")
vectors_subdir="concept_vectors_${label}"
name_count=60
suffix=""
if [[ "$FULL282" == "1" ]]; then name_count=0; suffix="_full282"; fi

echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ) run=$RUN_NAME model=$label job=${JOB_ID:-unknown}" >&2
case "$RUN_NAME" in
  smoke)
    "${PYTHON[@]}" smoke_tests/gemma4_transformerlens/smoke_test_bridge.py \
      --config "$CONFIG_PATH"
    ;;
  neutral)
    "${PYTHON[@]}" src/extract_neutral.py --config "$CONFIG_PATH" \
      --vectors-subdir "$vectors_subdir"
    ;;
  pca)
    "${PYTHON[@]}" src/denoise_vectors.py --config "$CONFIG_PATH" \
      --vectors-subdir "$vectors_subdir"
    ;;
  dense_raw|dense_denoised)
    vector_kind=${RUN_NAME#dense_}
    "${PYTHON[@]}" -m src.dense_steering --config "$CONFIG_PATH" \
      --vectors-subdir "$vectors_subdir" --label "${label}_${vector_kind}" \
      --vector-kind "$vector_kind" --include-cross-axis --n-random-directions 50 \
      --strengths=-0.1,-0.05,0,0.05,0.1 --prompt-format native-chat
    ;;
  audit)
    "${PYTHON[@]}" -m src.hiring_audit --config "$CONFIG_PATH" \
      --vectors-subdir "$vectors_subdir" --label "$label" --prompt-format native-chat
    ;;
  hiring_local|hiring_broad|hiring_denoised)
    regime=${RUN_NAME#hiring_}
    vector_kind=raw
    strengths=-0.1,-0.05,0,0.05,0.1
    output_suffix="$regime"
    if [[ "$regime" == "broad" ]]; then strengths=-0.5,-0.25,0,0.25,0.5; fi
    if [[ "$regime" == "denoised" ]]; then
      vector_kind=denoised
      output_suffix=denoised_local
    fi
    output_label="${label}_${output_suffix}${suffix}"
    "${PYTHON[@]}" -m src.hiring_steering --config "$CONFIG_PATH" \
      --vectors-subdir "$vectors_subdir" --label "$output_label" \
      --strengths="$strengths" --vector-kind "$vector_kind" \
      --n-names "$name_count" --prompt-format native-chat
    "${PYTHON[@]}" -m src.summarize_hiring_steering summarize \
      --config "$CONFIG_PATH" --label "$output_label" --n-boot 5000
    ;;
  posthoc)
    "${PYTHON[@]}" -m src.hiring_disparity --config "$CONFIG_PATH" --label "$label"
    "${PYTHON[@]}" -m src.hiring_r4 --config "$CONFIG_PATH" --label "$label"
    ;;
  full282_gate)
    "${PYTHON[@]}" -m src.summarize_hiring_steering gate \
      --config "$CONFIG_PATH" --model-label "$label"
    ;;
  *)
    echo "Unknown RUN_NAME=$RUN_NAME" >&2
    exit 14
    ;;
esac

"${PYTHON[@]}" -m src.validate_gemma4_remaining \
  --config "$CONFIG_PATH" --run "$RUN_NAME" "${full_args[@]}"
mkdir -p "$(dirname "$SUCCESS_SENTINEL")"
sentinel_tmp="${SUCCESS_SENTINEL}.tmp.${JOB_ID:-$$}"
: > "$sentinel_tmp"
mv "$sentinel_tmp" "$SUCCESS_SENTINEL"
echo "[success] run=$RUN_NAME -> $SUCCESS_SENTINEL"
