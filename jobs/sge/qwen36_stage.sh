#!/bin/bash
# Execute one independent Qwen3.6 production stage. Submit through a Qwen3.6 submitter.
set -euo pipefail

: "${STAGE:?STAGE is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${SUCCESS_SENTINEL:?SUCCESS_SENTINEL is required}"
: "${REPO_PATH:?REPO_PATH is required}"
: "${GIT_COMMIT:?GIT_COMMIT is required}"

module load conda  # ADJUST if SCCKN changes its module name.
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH="$REPO_PATH"
cd "$REPO_PATH"
PYTHON=(conda run --no-capture-output -n wc-qwen36-hf python)

if ! git merge-base --is-ancestor "$GIT_COMMIT" HEAD; then
  echo "Refusing Qwen3.6 Stage $STAGE: submitted commit is not an ancestor of HEAD." >&2
  exit 10
fi
if ! git diff --quiet "$GIT_COMMIT" HEAD -- \
  "$CONFIG_PATH" requirements-qwen36.txt src/qwen36_pipeline.py \
  src/validate_qwen36_stage.py src/utils/config.py jobs/sge/qwen36_stage.sh; then
  echo "Refusing Qwen3.6 Stage $STAGE: critical implementation changed." >&2
  exit 11
fi
if [[ -e "$SUCCESS_SENTINEL" ]]; then
  echo "Refusing Qwen3.6 Stage $STAGE: sentinel exists: $SUCCESS_SENTINEL" >&2
  exit 12
fi

if [[ "$STAGE" == "1" || "$STAGE" == "3" ]]; then
  if [[ "$("${PYTHON[@]}" -c 'import torch; print(torch.cuda.device_count())')" != "1" ]]; then
    echo "Refusing Qwen3.6 Stage $STAGE: exactly one visible GPU is required." >&2
    exit 13
  fi
  "${PYTHON[@]}" - "$CONFIG_PATH" <<'PY'
import sys
import torch
from src.utils.config import load_config

cfg = load_config(sys.argv[1])
name = torch.cuda.get_device_name(0)
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
free_gib = free_bytes / 1024**3
print(f"[runtime] gpu={name!r} free={free_gib:.2f} GiB total={total_bytes/1024**3:.2f} GiB")
if "RTX PRO 6000" not in name:
    raise SystemExit(f"Expected RTX PRO 6000, got {name!r}.")
if free_gib < cfg.native_hf.min_free_vram_gib:
    raise SystemExit(
        f"Only {free_gib:.2f} GiB free; {cfg.native_hf.min_free_vram_gib:.2f} GiB required."
    )
PY
fi

"${PYTHON[@]}" -m src.validate_qwen36_stage \
  --config "$CONFIG_PATH" --stage "$STAGE" --require-absent
echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ) stage=$STAGE job=${JOB_ID:-unknown}" >&2
"${PYTHON[@]}" -m src.qwen36_pipeline --config "$CONFIG_PATH" --stage "$STAGE"
"${PYTHON[@]}" -m src.validate_qwen36_stage --config "$CONFIG_PATH" --stage "$STAGE"

mkdir -p "$(dirname "$SUCCESS_SENTINEL")"
sentinel_tmp="${SUCCESS_SENTINEL}.tmp.${JOB_ID:-$$}"
: > "$sentinel_tmp"
mv "$sentinel_tmp" "$SUCCESS_SENTINEL"
echo "[success] stage=$STAGE -> $SUCCESS_SENTINEL"
