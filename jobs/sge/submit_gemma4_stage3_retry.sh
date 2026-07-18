#!/bin/bash
# Submit independent held Stage 3 retries for Gemma 4 26B-A4B and 31B.
set -euo pipefail

MODE="${1:-}"
if [[ -n "$MODE" && "$MODE" != "--dry-run" ]]; then
  echo "usage: bash jobs/sge/submit_gemma4_stage3_retry.sh [--dry-run]" >&2
  exit 2
fi
DRY_RUN=0
if [[ "$MODE" == "--dry-run" ]]; then DRY_RUN=1; fi

cd "$(git rev-parse --show-toplevel)"
REPO_PATH="$(pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="/work/emrecan.ulu/gemma4_stage3_retry/$RUN_ID"
MANIFEST="results/logs/gemma4_stage3_retry_submission_${RUN_ID}.json"
GIT_COMMIT="$(git rev-parse HEAD)"
QUEUE="gpu@scc214"
GPU_RESOURCES="gpu=1,rtx_6000=1,h_vmem=96G,h_rt=12:00:00"
PRIORITY="0"
SENTINEL_26B="$STATE_DIR/26b.success"
SENTINEL_31B="$STATE_DIR/31b.success"
FINAL_SENTINEL="$STATE_DIR/final.success"

outputs=(
  results/tables/layer_sweep_gemma4_26b_a4b.csv
  results/tables/layer_sweep_gemma4_26b_a4b.meta.json
  results/tables/layer_sweep_gemma4_31b.csv
  results/tables/layer_sweep_gemma4_31b.meta.json
)

if ((DRY_RUN)); then
  echo "[dry-run] held model=26b queue=$QUEUE resources=$GPU_RESOURCES priority=$PRIORITY predecessor=none"
  echo "[dry-run] held model=31b queue=$QUEUE resources=$GPU_RESOURCES priority=$PRIORITY predecessor=none"
  echo "[dry-run] finalizer queue=scc hold=26b,31b gpu=0"
  echo "[dry-run] run_id=$RUN_ID manifest=$MANIFEST"
  exit 0
fi

for path in "$MANIFEST" "$STATE_DIR" "${outputs[@]}"; do
  if [[ -e "$path" ]]; then
    echo "Refusing submission: path already exists: $path" >&2
    exit 3
  fi
done

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing submission: SCCKN worktree is not clean." >&2
  exit 4
fi
if pgrep -u "$USER" -f 'pip(3)? install|conda (create|install|update)' >/dev/null; then
  echo "Refusing submission: a pip/conda mutation is still running for $USER." >&2
  exit 5
fi
conda run -n wc-tl-g4 python -m pip check
conda run -n wc-tl-g4 python -c \
  'from importlib.metadata import version; import torch, transformers; print("torch={} transformers={} transformer-lens={}".format(torch.__version__, transformers.__version__, version("transformer-lens")))'

validate_common_26=(
  --model google/gemma-4-26B-A4B-it
  --label gemma4_26b_a4b
  --vectors-subdir concept_vectors_gemma4_26b_a4b
  --expected-layers 30
  --expected-d-model 2816
)
validate_common_31=(
  --model google/gemma-4-31B-it
  --label gemma4_31b
  --vectors-subdir concept_vectors_gemma4_31b
  --expected-layers 60
  --expected-d-model 5376
)
for stage in 1 2; do
  conda run -n wc-tl-g4 python -m src.validate_gemma4_stage --stage "$stage" "${validate_common_26[@]}"
  conda run -n wc-tl-g4 python -m src.validate_gemma4_stage --stage "$stage" "${validate_common_31[@]}"
done

available=$(qstat -F gpu -q "$QUEUE" | awk -F= '/qc:gpu=/{gsub(/[[:space:]]/, "", $2); print $2; exit}')
if [[ ! "$available" =~ ^[0-9]+$ ]]; then
  echo "Refusing submission: could not parse available GPU count for $QUEUE." >&2
  exit 6
fi
if ((available < 2)); then
  echo "Refusing submission: $QUEUE has $available available GPUs; two are required." >&2
  exit 7
fi
echo "[preflight] $QUEUE reports $available available GPUs"

mkdir -p "$STATE_DIR" results/logs
submitted_jobs=()
cleanup_partial_submission() {
  status=$?
  if ((status != 0 && ${#submitted_jobs[@]})); then
    echo "Submission failed; deleting held partial jobs: ${submitted_jobs[*]}" >&2
    qdel "${submitted_jobs[@]}" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup_partial_submission EXIT

vars_26="MODEL_NAME=google/gemma-4-26B-A4B-it,LABEL=gemma4_26b_a4b,VECTORS_SUBDIR=concept_vectors_gemma4_26b_a4b,EXPECTED_LAYERS=30,EXPECTED_D_MODEL=2816,SUCCESS_SENTINEL=$SENTINEL_26B,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT"
vars_31="MODEL_NAME=google/gemma-4-31B-it,LABEL=gemma4_31b,VECTORS_SUBDIR=concept_vectors_gemma4_31b,EXPECTED_LAYERS=60,EXPECTED_D_MODEL=5376,SUCCESS_SENTINEL=$SENTINEL_31B,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT"

job_26=$(qsub -terse -h -p "$PRIORITY" \
  -q "$QUEUE" -l "$GPU_RESOURCES" -pe smp 2 \
  -N g4_s3r_26b \
  -o "results/logs/gemma4_stage3_retry_${RUN_ID}_26b.out" \
  -e "results/logs/gemma4_stage3_retry_${RUN_ID}_26b.err" \
  -v "$vars_26" jobs/sge/gemma4_stage3_retry.sh)
submitted_jobs+=("$job_26")

job_31=$(qsub -terse -h -p "$PRIORITY" \
  -q "$QUEUE" -l "$GPU_RESOURCES" -pe smp 2 \
  -N g4_s3r_31b \
  -o "results/logs/gemma4_stage3_retry_${RUN_ID}_31b.out" \
  -e "results/logs/gemma4_stage3_retry_${RUN_ID}_31b.err" \
  -v "$vars_31" jobs/sge/gemma4_stage3_retry.sh)
submitted_jobs+=("$job_31")

final_vars="REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT,SENTINEL_26B=$SENTINEL_26B,SENTINEL_31B=$SENTINEL_31B,FINAL_SENTINEL=$FINAL_SENTINEL"
job_final=$(qsub -terse -p "$PRIORITY" -hold_jid "$job_26,$job_31" \
  -q scc -l h_vmem=4G,h_rt=00:15:00 -pe smp 1 \
  -N g4_s3r_fin \
  -o "results/logs/gemma4_stage3_retry_${RUN_ID}_final.out" \
  -e "results/logs/gemma4_stage3_retry_${RUN_ID}_final.err" \
  -v "$final_vars" jobs/sge/gemma4_stage3_finalize.sh)
submitted_jobs+=("$job_final")

export RUN_ID STATE_DIR MANIFEST GIT_COMMIT REPO_PATH QUEUE GPU_RESOURCES PRIORITY
export JOB_26="$job_26" JOB_31="$job_31" JOB_FINAL="$job_final"
export SENTINEL_26B SENTINEL_31B FINAL_SENTINEL
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "run_id": os.environ["RUN_ID"],
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "gpu-jobs-user-held",
    "git_commit": os.environ["GIT_COMMIT"],
    "repo_path": os.environ["REPO_PATH"],
    "state_dir": os.environ["STATE_DIR"],
    "release_command": f"qrls {os.environ['JOB_26']} {os.environ['JOB_31']}",
    "jobs": [
        {
            "model": "google/gemma-4-26B-A4B-it",
            "label": "gemma4_26b_a4b",
            "job_id": os.environ["JOB_26"],
            "queue": os.environ["QUEUE"],
            "resources": os.environ["GPU_RESOURCES"],
            "priority": int(os.environ["PRIORITY"]),
            "user_held": True,
            "success_sentinel": os.environ["SENTINEL_26B"],
            "outputs": [
                "results/tables/layer_sweep_gemma4_26b_a4b.csv",
                "results/tables/layer_sweep_gemma4_26b_a4b.meta.json",
            ],
        },
        {
            "model": "google/gemma-4-31B-it",
            "label": "gemma4_31b",
            "job_id": os.environ["JOB_31"],
            "queue": os.environ["QUEUE"],
            "resources": os.environ["GPU_RESOURCES"],
            "priority": int(os.environ["PRIORITY"]),
            "user_held": True,
            "success_sentinel": os.environ["SENTINEL_31B"],
            "outputs": [
                "results/tables/layer_sweep_gemma4_31b.csv",
                "results/tables/layer_sweep_gemma4_31b.meta.json",
            ],
        },
        {
            "role": "finalizer",
            "job_id": os.environ["JOB_FINAL"],
            "queue": "scc",
            "gpu_request": 0,
            "hold_job_ids": [os.environ["JOB_26"], os.environ["JOB_31"]],
            "success_sentinel": os.environ["FINAL_SENTINEL"],
        },
    ],
}
Path(os.environ["MANIFEST"]).write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
PY

trap - EXIT
echo "[submitted-held] 26b=$job_26 31b=$job_31 finalizer=$job_final"
echo "[manifest] $MANIFEST"
echo "[next] synchronize the manifest, then release both GPU jobs: qrls $job_26 $job_31"
