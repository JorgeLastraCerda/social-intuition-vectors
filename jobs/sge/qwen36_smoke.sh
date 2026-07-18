#!/bin/bash
# Run the native-HF Qwen3.6 Stage 1--3 smoke on one RTX PRO 6000.
# Submit only through submit_qwen36_smoke.sh.
# ADJUST: module and scratch paths if SCCKN changes its runtime.
#$ -q gpu@scc214
#$ -l h_rt=02:00:00
#$ -l h_vmem=96G
#$ -pe smp 2
#$ -l gpu=1
#$ -l rtx_6000=1
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${SUCCESS_SENTINEL:?SUCCESS_SENTINEL is required}"
: "${REPO_PATH:?REPO_PATH is required}"
: "${GIT_COMMIT:?GIT_COMMIT is required}"

module load conda  # ADJUST
conda activate wc-qwen36-hf
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH="$REPO_PATH"
cd "$REPO_PATH"

if ! git merge-base --is-ancestor "$GIT_COMMIT" HEAD; then
  echo "Refusing Qwen3.6 smoke: submitted commit is not an ancestor of HEAD." >&2
  exit 10
fi
if ! git diff --quiet "$GIT_COMMIT" HEAD -- \
  "$CONFIG_PATH" \
  requirements-qwen36.txt \
  src/qwen36_smoke.py \
  src/validate_qwen36_smoke.py \
  src/utils/config.py \
  jobs/sge/qwen36_smoke.sh; then
  echo "Refusing Qwen3.6 smoke: critical implementation changed after submission." >&2
  exit 11
fi
if [[ "$(python -c 'import torch; print(torch.cuda.device_count())')" != "1" ]]; then
  echo "Refusing Qwen3.6 smoke: Grid Engine did not expose exactly one GPU." >&2
  exit 12
fi
if [[ -e "$SUCCESS_SENTINEL" ]]; then
  echo "Refusing Qwen3.6 smoke: success sentinel already exists: $SUCCESS_SENTINEL" >&2
  exit 13
fi

python - "$CONFIG_PATH" <<'PY'
import sys

import torch

from src.utils.config import load_config

cfg = load_config(sys.argv[1])
name = torch.cuda.get_device_name(0)
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
free_gib = free_bytes / 1024**3
total_gib = total_bytes / 1024**3
print(
    f"[runtime] gpu={name!r} visible={torch.cuda.device_count()} "
    f"free={free_gib:.2f} GiB total={total_gib:.2f} GiB",
    flush=True,
)
if "RTX PRO 6000" not in name:
    raise SystemExit(f"Expected an RTX PRO 6000, got {name!r}.")
if free_gib < cfg.smoke.min_free_vram_gib:
    raise SystemExit(
        f"Only {free_gib:.2f} GiB is free; "
        f"{cfg.smoke.min_free_vram_gib:.2f} GiB is required."
    )
PY

python -m src.validate_qwen36_smoke --config "$CONFIG_PATH" --require-absent
echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ) job=${JOB_ID:-unknown}" >&2
python -m src.qwen36_smoke --config "$CONFIG_PATH"
python -m src.validate_qwen36_smoke --config "$CONFIG_PATH"

mkdir -p "$(dirname "$SUCCESS_SENTINEL")"
sentinel_tmp="${SUCCESS_SENTINEL}.tmp.${JOB_ID:-$$}"
: > "$sentinel_tmp"
mv "$sentinel_tmp" "$SUCCESS_SENTINEL"
echo "[success] $(date -u +%Y-%m-%dT%H:%M:%SZ) -> $SUCCESS_SENTINEL"
