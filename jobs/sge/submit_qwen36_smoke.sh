#!/bin/bash
# Submit one held Qwen3.6-27B Stage 1--3 smoke plus CPU finalizer.
set -euo pipefail

MODE="${1:-}"
if [[ -n "$MODE" && "$MODE" != "--dry-run" ]]; then
  echo "usage: bash jobs/sge/submit_qwen36_smoke.sh [--dry-run]" >&2
  exit 2
fi
DRY_RUN=0
if [[ "$MODE" == "--dry-run" ]]; then DRY_RUN=1; fi

cd "$(git rev-parse --show-toplevel)"
REPO_PATH="$(pwd)"
CONFIG_PATH="config/qwen36_smoke.yaml"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="/work/emrecan.ulu/qwen36_smoke/$RUN_ID"
MANIFEST="results/logs/qwen36_smoke_submission_${RUN_ID}.json"
GIT_COMMIT="$(git rev-parse HEAD)"
QUEUE="gpu@scc214"
GPU_RESOURCES="gpu=1,rtx_6000=1,h_vmem=96G,h_rt=02:00:00"
PRIORITY="0"
GPU_SENTINEL="$STATE_DIR/gpu.success"
FINAL_SENTINEL="$STATE_DIR/final.success"

if ((DRY_RUN)); then
  echo "[dry-run] held smoke config=$CONFIG_PATH queue=$QUEUE resources=$GPU_RESOURCES priority=$PRIORITY"
  echo "[dry-run] finalizer queue=scc hold=smoke gpu=0"
  echo "[dry-run] availability policy=submit-even-when-zero"
  echo "[dry-run] run_id=$RUN_ID manifest=$MANIFEST"
  exit 0
fi

for path in "$MANIFEST" "$STATE_DIR"; do
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

module load conda  # ADJUST
PYTHON=(conda run --no-capture-output -n wc-qwen36-hf python)
"${PYTHON[@]}" -m pip check
"${PYTHON[@]}" - <<'PY'
import importlib.util

import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor

assert transformers.__version__ == "5.14.1", transformers.__version__
assert importlib.util.find_spec("transformer_lens") is None
print("torch", torch.__version__, "transformers", transformers.__version__)
print("classes", AutoProcessor.__name__, AutoModelForMultimodalLM.__name__)
PY
"${PYTHON[@]}" -m src.validate_qwen36_smoke \
  --config "$CONFIG_PATH" --require-absent

available=$(qstat -F gpu -q "$QUEUE" | awk -F= '/qc:gpu=/{gsub(/[[:space:]]/, "", $2); print $2; exit}')
if [[ ! "$available" =~ ^[0-9]+$ ]]; then
  echo "Refusing submission: could not parse available GPU count for $QUEUE." >&2
  exit 6
fi
echo "[preflight] $QUEUE reports $available available GPUs; submission will proceed"

vars="CONFIG_PATH=$CONFIG_PATH,SUCCESS_SENTINEL=$GPU_SENTINEL,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT"
final_vars="CONFIG_PATH=$CONFIG_PATH,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT,GPU_SENTINEL=$GPU_SENTINEL,FINAL_SENTINEL=$FINAL_SENTINEL"

qsub -w v -h -p "$PRIORITY" -q "$QUEUE" -l "$GPU_RESOURCES" -pe smp 2 \
  -v "$vars" jobs/sge/qwen36_smoke.sh >/dev/null
qsub -w v -p "$PRIORITY" -q scc -l h_vmem=4G,h_rt=00:15:00 -pe smp 1 \
  -v "$final_vars" jobs/sge/qwen36_smoke_finalize.sh >/dev/null

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

job_gpu=$(qsub -terse -h -p "$PRIORITY" \
  -q "$QUEUE" -l "$GPU_RESOURCES" -pe smp 2 \
  -N q36_s123_smoke \
  -o "results/logs/qwen36_smoke_${RUN_ID}.out" \
  -e "results/logs/qwen36_smoke_${RUN_ID}.err" \
  -v "$vars" jobs/sge/qwen36_smoke.sh)
submitted_jobs+=("$job_gpu")

job_final=$(qsub -terse -p "$PRIORITY" -hold_jid "$job_gpu" \
  -q scc -l h_vmem=4G,h_rt=00:15:00 -pe smp 1 \
  -N q36_smoke_fin \
  -o "results/logs/qwen36_smoke_${RUN_ID}_final.out" \
  -e "results/logs/qwen36_smoke_${RUN_ID}_final.err" \
  -v "$final_vars" jobs/sge/qwen36_smoke_finalize.sh)
submitted_jobs+=("$job_final")

qstat -j "$job_gpu" >/dev/null
qstat -j "$job_final" >/dev/null

export RUN_ID STATE_DIR MANIFEST GIT_COMMIT REPO_PATH CONFIG_PATH QUEUE
export GPU_RESOURCES PRIORITY AVAILABLE="$available" JOB_GPU="$job_gpu" JOB_FINAL="$job_final"
export GPU_SENTINEL FINAL_SENTINEL
"${PYTHON[@]}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import load_config

cfg = load_config(os.environ["CONFIG_PATH"])
available = int(os.environ["AVAILABLE"])
payload = {
    "run_id": os.environ["RUN_ID"],
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "gpu-job-user-held",
    "availability_at_submission": available,
    "availability_interpretation": "available" if available > 0 else "queued-no-gpu-available",
    "git_commit": os.environ["GIT_COMMIT"],
    "repo_path": os.environ["REPO_PATH"],
    "config_path": os.environ["CONFIG_PATH"],
    "model": cfg.model.name,
    "revision": cfg.model.revision,
    "state_dir": os.environ["STATE_DIR"],
    "release_command": f"qrls {os.environ['JOB_GPU']}",
    "jobs": [
        {
            "role": "stage-1-2-3-smoke",
            "job_id": os.environ["JOB_GPU"],
            "queue": os.environ["QUEUE"],
            "resources": os.environ["GPU_RESOURCES"],
            "priority": int(os.environ["PRIORITY"]),
            "user_held": True,
            "success_sentinel": os.environ["GPU_SENTINEL"],
        },
        {
            "role": "finalizer",
            "job_id": os.environ["JOB_FINAL"],
            "queue": "scc",
            "gpu_request": 0,
            "hold_job_ids": [os.environ["JOB_GPU"]],
            "success_sentinel": os.environ["FINAL_SENTINEL"],
        },
    ],
}
Path(os.environ["MANIFEST"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

trap - EXIT
echo "[submitted-held] gpu=$job_gpu finalizer=$job_final availability=$available"
echo "[manifest] $MANIFEST"
echo "[next] synchronize the manifest, then release: qrls $job_gpu"
