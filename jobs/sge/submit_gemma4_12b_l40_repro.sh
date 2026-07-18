#!/bin/bash
# Submit one held Gemma 4 12B Stage 3 reproducibility audit to an exact L40.
set -euo pipefail

MODE="${1:-}"
if [[ -n "$MODE" && "$MODE" != "--dry-run" ]]; then
  echo "usage: bash jobs/sge/submit_gemma4_12b_l40_repro.sh [--dry-run]" >&2
  exit 2
fi
DRY_RUN=0
if [[ "$MODE" == "--dry-run" ]]; then DRY_RUN=1; fi

cd "$(git rev-parse --show-toplevel)"
REPO_PATH="$(pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="/work/emrecan.ulu/gemma4_stage3_l40_repro/$RUN_ID"
MANIFEST="results/logs/gemma4_stage3_retry_submission_12b_l40_repro_${RUN_ID}.json"
GIT_COMMIT="$(git rev-parse HEAD)"
QUEUE="gpu@scc192"
GPU_RESOURCES="gpu=1,h_vmem=32G,h_rt=01:00:00"
PRIORITY="0"
OUTPUT_LABEL="gemma4_12b_l40_repro"
EXPECTED_GPU_KIND="L40"
SUCCESS_SENTINEL="$STATE_DIR/12b_l40_repro.success"
FINAL_SENTINEL="$STATE_DIR/final.success"
OUTPUT_CSV="results/tables/layer_sweep_${OUTPUT_LABEL}.csv"
OUTPUT_META="results/tables/layer_sweep_${OUTPUT_LABEL}.meta.json"

if ((DRY_RUN)); then
  echo "[dry-run] held model=12b hardware=NVIDIA_L40 queue=$QUEUE resources=$GPU_RESOURCES output_label=$OUTPUT_LABEL"
  echo "[dry-run] finalizer queue=scc hold=12b_l40_repro gpu=0"
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
conda run -n wc-tl-g4 python -m pip check

source_args=(
  --model google/gemma-4-12B-it
  --label gemma4_12b
  --vectors-subdir concept_vectors_gemma4_12b
  --expected-layers 48
  --expected-d-model 3840
)
output_args=(
  --model google/gemma-4-12B-it
  --label "$OUTPUT_LABEL"
  --vectors-subdir concept_vectors_gemma4_12b
  --expected-layers 48
  --expected-d-model 3840
)
conda run -n wc-tl-g4 python -m src.validate_gemma4_stage --stage 1 "${source_args[@]}"
conda run -n wc-tl-g4 python -m src.validate_gemma4_stage --stage 2 "${source_args[@]}"
conda run -n wc-tl-g4 python -m src.validate_gemma4_stage --stage 3 "${output_args[@]}" --require-absent

available=$(qstat -F gpu -q "$QUEUE" | awk -F= '/qc:gpu=/{gsub(/[[:space:]]/, "", $2); print $2; exit}')
if [[ ! "$available" =~ ^[0-9]+$ ]] || ((available < 1)); then
  echo "Refusing submission: $QUEUE has no parseable available GPU." >&2
  exit 6
fi
echo "[preflight] $QUEUE reports $available available exact-L40 GPUs"

vars="SUCCESS_SENTINEL=$SUCCESS_SENTINEL,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT,OUTPUT_LABEL=$OUTPUT_LABEL,EXPECTED_GPU_KIND=$EXPECTED_GPU_KIND"
final_vars="SUCCESS_SENTINEL=$SUCCESS_SENTINEL,FINAL_SENTINEL=$FINAL_SENTINEL,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT,OUTPUT_LABEL=$OUTPUT_LABEL"

qsub -w v -h -p "$PRIORITY" -q "$QUEUE" -l "$GPU_RESOURCES" -pe smp 2 \
  -N g4_12_l40r -v "$vars" jobs/sge/gemma4_12b_stage3_retry.sh >/dev/null
qsub -w v -p "$PRIORITY" -q scc -l h_vmem=4G,h_rt=00:15:00 -pe smp 1 \
  -N g4_12_l40f -v "$final_vars" jobs/sge/gemma4_12b_stage3_finalize.sh >/dev/null

mkdir -p "$STATE_DIR" results/logs
submitted_jobs=()
cleanup_partial_submission() {
  status=$?
  if ((status != 0 && ${#submitted_jobs[@]})); then
    qdel "${submitted_jobs[@]}" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup_partial_submission EXIT

job_gpu=$(qsub -terse -h -p "$PRIORITY" -q "$QUEUE" -l "$GPU_RESOURCES" -pe smp 2 \
  -N g4_12_l40r \
  -o "results/logs/gemma4_12b_l40_repro_${RUN_ID}.out" \
  -e "results/logs/gemma4_12b_l40_repro_${RUN_ID}.err" \
  -v "$vars" jobs/sge/gemma4_12b_stage3_retry.sh)
submitted_jobs+=("$job_gpu")
job_final=$(qsub -terse -p "$PRIORITY" -hold_jid "$job_gpu" \
  -q scc -l h_vmem=4G,h_rt=00:15:00 -pe smp 1 \
  -N g4_12_l40f \
  -o "results/logs/gemma4_12b_l40_repro_${RUN_ID}_final.out" \
  -e "results/logs/gemma4_12b_l40_repro_${RUN_ID}_final.err" \
  -v "$final_vars" jobs/sge/gemma4_12b_stage3_finalize.sh)
submitted_jobs+=("$job_final")

export RUN_ID STATE_DIR MANIFEST GIT_COMMIT REPO_PATH QUEUE GPU_RESOURCES PRIORITY
export OUTPUT_LABEL EXPECTED_GPU_KIND JOB_GPU="$job_gpu" JOB_FINAL="$job_final"
export SUCCESS_SENTINEL FINAL_SENTINEL
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "run_id": os.environ["RUN_ID"],
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "gpu-job-user-held",
    "purpose": "L40 versus L40S Stage 3 reproducibility audit",
    "git_commit": os.environ["GIT_COMMIT"],
    "repo_path": os.environ["REPO_PATH"],
    "state_dir": os.environ["STATE_DIR"],
    "release_command": f"qrls {os.environ['JOB_GPU']}",
    "jobs": [
        {
            "model": "google/gemma-4-12B-it",
            "label": os.environ["OUTPUT_LABEL"],
            "expected_gpu_name": f"NVIDIA {os.environ['EXPECTED_GPU_KIND']}",
            "job_id": os.environ["JOB_GPU"],
            "queue": os.environ["QUEUE"],
            "resources": os.environ["GPU_RESOURCES"],
            "priority": int(os.environ["PRIORITY"]),
            "user_held": True,
            "success_sentinel": os.environ["SUCCESS_SENTINEL"],
            "outputs": [
                f"results/tables/layer_sweep_{os.environ['OUTPUT_LABEL']}.csv",
                f"results/tables/layer_sweep_{os.environ['OUTPUT_LABEL']}.meta.json",
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
Path(os.environ["MANIFEST"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

trap - EXIT
echo "[submitted-held] 12b_l40_repro=$job_gpu finalizer=$job_final"
echo "[manifest] $MANIFEST"
echo "[next] synchronize the manifest, then release: qrls $job_gpu"
