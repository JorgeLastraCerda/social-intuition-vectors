#!/bin/bash
# Submit one held Gemma 4 12B Stage 3 retry to the SCCKN L40 pool.
set -euo pipefail

MODE="${1:-}"
if [[ -n "$MODE" && "$MODE" != "--dry-run" ]]; then
  echo "usage: bash jobs/sge/submit_gemma4_12b_stage3_retry.sh [--dry-run]" >&2
  exit 2
fi
DRY_RUN=0
if [[ "$MODE" == "--dry-run" ]]; then DRY_RUN=1; fi

cd "$(git rev-parse --show-toplevel)"
REPO_PATH="$(pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="/work/emrecan.ulu/gemma4_stage3_retry_12b/$RUN_ID"
MANIFEST="results/logs/gemma4_stage3_retry_submission_12b_${RUN_ID}.json"
GIT_COMMIT="$(git rev-parse HEAD)"
QUEUE="gpu@scc192,gpu@scc213"
GPU_RESOURCES="gpu=1,h_vmem=32G,h_rt=01:00:00"
PRIORITY="0"
SUCCESS_SENTINEL="$STATE_DIR/12b.success"
FINAL_SENTINEL="$STATE_DIR/final.success"
OUTPUT_CSV="results/tables/layer_sweep_gemma4_12b.csv"
OUTPUT_META="results/tables/layer_sweep_gemma4_12b.meta.json"

if ((DRY_RUN)); then
  echo "[dry-run] held model=12b hardware=L40 queue=$QUEUE resources=$GPU_RESOURCES priority=$PRIORITY predecessor=none"
  echo "[dry-run] finalizer queue=scc hold=12b gpu=0"
  echo "[dry-run] run_id=$RUN_ID manifest=$MANIFEST"
  exit 0
fi

for path in "$MANIFEST" "$STATE_DIR" "$OUTPUT_CSV" "$OUTPUT_META"; do
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

validate_common=(
  --model google/gemma-4-12B-it
  --label gemma4_12b
  --vectors-subdir concept_vectors_gemma4_12b
  --expected-layers 48
  --expected-d-model 3840
)
conda run -n wc-tl-g4 python -m src.validate_gemma4_stage --stage 1 "${validate_common[@]}"
conda run -n wc-tl-g4 python -m src.validate_gemma4_stage --stage 2 "${validate_common[@]}"
conda run -n wc-tl-g4 python -m src.validate_gemma4_stage --stage 3 "${validate_common[@]}" --require-absent

available=0
for queue_instance in gpu@scc192 gpu@scc213; do
  count=$(qstat -F gpu -q "$queue_instance" | awk -F= '/qc:gpu=/{gsub(/[[:space:]]/, "", $2); print $2; exit}')
  if [[ ! "$count" =~ ^[0-9]+$ ]]; then
    echo "Refusing submission: could not parse available GPU count for $queue_instance." >&2
    exit 6
  fi
  available=$((available + count))
done
if ((available < 1)); then
  echo "Refusing submission: the L40 pool has no available GPU." >&2
  exit 7
fi
echo "[preflight] L40 pool reports $available available GPUs"

vars="SUCCESS_SENTINEL=$SUCCESS_SENTINEL,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT"
final_vars="SUCCESS_SENTINEL=$SUCCESS_SENTINEL,FINAL_SENTINEL=$FINAL_SENTINEL,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT"

qsub -w v -h -p "$PRIORITY" \
  -q "$QUEUE" -l "$GPU_RESOURCES" -pe smp 2 \
  -N g4_s3r_12b -v "$vars" jobs/sge/gemma4_12b_stage3_retry.sh >/dev/null
qsub -w v -p "$PRIORITY" \
  -q scc -l h_vmem=4G,h_rt=00:15:00 -pe smp 1 \
  -N g4_s3r_12f -v "$final_vars" jobs/sge/gemma4_12b_stage3_finalize.sh >/dev/null
echo "[preflight] qsub verification passed for the L40 job and CPU finalizer"

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
  -N g4_s3r_12b \
  -o "results/logs/gemma4_stage3_retry_12b_${RUN_ID}.out" \
  -e "results/logs/gemma4_stage3_retry_12b_${RUN_ID}.err" \
  -v "$vars" jobs/sge/gemma4_12b_stage3_retry.sh)
submitted_jobs+=("$job_gpu")

job_final=$(qsub -terse -p "$PRIORITY" -hold_jid "$job_gpu" \
  -q scc -l h_vmem=4G,h_rt=00:15:00 -pe smp 1 \
  -N g4_s3r_12f \
  -o "results/logs/gemma4_stage3_retry_12b_${RUN_ID}_final.out" \
  -e "results/logs/gemma4_stage3_retry_12b_${RUN_ID}_final.err" \
  -v "$final_vars" jobs/sge/gemma4_12b_stage3_finalize.sh)
submitted_jobs+=("$job_final")

export RUN_ID STATE_DIR MANIFEST GIT_COMMIT REPO_PATH QUEUE GPU_RESOURCES PRIORITY
export JOB_GPU="$job_gpu" JOB_FINAL="$job_final" SUCCESS_SENTINEL FINAL_SENTINEL
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "run_id": os.environ["RUN_ID"],
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "gpu-job-user-held",
    "git_commit": os.environ["GIT_COMMIT"],
    "repo_path": os.environ["REPO_PATH"],
    "state_dir": os.environ["STATE_DIR"],
    "release_command": f"qrls {os.environ['JOB_GPU']}",
    "jobs": [
        {
            "model": "google/gemma-4-12B-it",
            "label": "gemma4_12b",
            "hardware": "NVIDIA L40",
            "job_id": os.environ["JOB_GPU"],
            "queue": os.environ["QUEUE"],
            "resources": os.environ["GPU_RESOURCES"],
            "priority": int(os.environ["PRIORITY"]),
            "user_held": True,
            "success_sentinel": os.environ["SUCCESS_SENTINEL"],
            "outputs": [
                "results/tables/layer_sweep_gemma4_12b.csv",
                "results/tables/layer_sweep_gemma4_12b.meta.json",
            ],
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
Path(os.environ["MANIFEST"]).write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
PY

trap - EXIT
echo "[submitted-held] 12b=$job_gpu finalizer=$job_final"
echo "[manifest] $MANIFEST"
echo "[next] synchronize the manifest, then release the GPU job: qrls $job_gpu"
