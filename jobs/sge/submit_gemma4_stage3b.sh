#!/bin/bash
# Submit three independent held Gemma 4 Stage 3B GPU jobs and one CPU validator.
set -euo pipefail

MODE="${1:-}"
if [[ -n "$MODE" && "$MODE" != "--dry-run" ]]; then
  echo "usage: bash jobs/sge/submit_gemma4_stage3b.sh [--dry-run]" >&2
  exit 2
fi
DRY_RUN=0
[[ "$MODE" == "--dry-run" ]] && DRY_RUN=1

cd "$(git rev-parse --show-toplevel)"
REPO_PATH="$(pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="/work/emrecan.ulu/gemma4_stage3b/$RUN_ID"
MANIFEST="results/logs/gemma4_stage3b_submission_${RUN_ID}.json"
GIT_COMMIT="$(git rev-parse HEAD)"
RTX_QUEUE="gpu@scc214"
RTX_RESOURCES="gpu=1,rtx_6000=1,h_vmem=96G,h_rt=02:00:00"
L40_QUEUE="gpu@scc192"
L40_RESOURCES="gpu=1,h_vmem=32G,h_rt=02:00:00"
PRIORITY=0
SENTINEL_12B="$STATE_DIR/12b.success"
SENTINEL_26B="$STATE_DIR/26b.success"
SENTINEL_31B="$STATE_DIR/31b.success"
FINAL_SENTINEL="$STATE_DIR/final.success"

labels=(stage3b_gemma4_12b_l40 stage3b_gemma4_26b_a4b stage3b_gemma4_31b)
outputs=()
for label in "${labels[@]}"; do
  outputs+=(
    "results/tables/layer_sweep_${label}.csv"
    "results/tables/layer_sweep_${label}.meta.json"
    "results/logs/validate_layer_sweep_${label}.json"
  )
done

if ((DRY_RUN)); then
  echo "[dry-run] held model=12b queue=$L40_QUEUE hardware=NVIDIA_L40 resources=$L40_RESOURCES predecessor=none"
  echo "[dry-run] held model=26b queue=$RTX_QUEUE hardware=RTX_PRO_6000 resources=$RTX_RESOURCES predecessor=none"
  echo "[dry-run] held model=31b queue=$RTX_QUEUE hardware=RTX_PRO_6000 resources=$RTX_RESOURCES predecessor=none"
  echo "[dry-run] finalizer queue=scc hold=12b,26b,31b gpu=0"
  echo "[dry-run] run_id=$RUN_ID manifest=$MANIFEST"
  exit 0
fi

for path in "$MANIFEST" "$STATE_DIR" "${outputs[@]}"; do
  [[ ! -e "$path" ]] || { echo "Refusing submission: path exists: $path" >&2; exit 3; }
done
[[ -z "$(git status --porcelain)" ]] || {
  echo "Refusing submission: SCCKN worktree is not clean." >&2; exit 4;
}
conda run -n wc-tl-g4 python -m pip check

validate_source() {
  local model="$1" label="$2" vectors="$3" layers="$4" width="$5"
  for stage in 1 2; do
    conda run -n wc-tl-g4 python -m src.validate_gemma4_stage \
      --stage "$stage" --model "$model" --label "$label" --vectors-subdir "$vectors" \
      --expected-layers "$layers" --expected-d-model "$width"
  done
}
validate_source google/gemma-4-12B-it gemma4_12b concept_vectors_gemma4_12b 48 3840
validate_source google/gemma-4-26B-A4B-it gemma4_26b_a4b concept_vectors_gemma4_26b_a4b 30 2816
validate_source google/gemma-4-31B-it gemma4_31b concept_vectors_gemma4_31b 60 5376

available_rtx=$(qstat -F gpu -q "$RTX_QUEUE" | awk -F= '/qc:gpu=/{gsub(/[[:space:]]/, "", $2); print $2; exit}')
available_l40=$(qstat -F gpu -q "$L40_QUEUE" | awk -F= '/qc:gpu=/{gsub(/[[:space:]]/, "", $2); print $2; exit}')
[[ "$available_rtx" =~ ^[0-9]+$ && "$available_l40" =~ ^[0-9]+$ ]] || {
  echo "Refusing submission: could not parse GPU availability." >&2; exit 5;
}
((available_rtx >= 2 && available_l40 >= 1)) || {
  echo "Refusing submission: need 2 RTX and 1 L40; available RTX=$available_rtx L40=$available_l40" >&2; exit 6;
}

common_12="MODEL_NAME=google/gemma-4-12B-it,LABEL=stage3b_gemma4_12b_l40,SOURCE_LABEL=gemma4_12b,VECTORS_SUBDIR=concept_vectors_gemma4_12b,EXPECTED_LAYERS=48,EXPECTED_D_MODEL=3840,EXPECTED_GPU_KIND=L40,SUCCESS_SENTINEL=$SENTINEL_12B,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT"
common_26="MODEL_NAME=google/gemma-4-26B-A4B-it,LABEL=stage3b_gemma4_26b_a4b,SOURCE_LABEL=gemma4_26b_a4b,VECTORS_SUBDIR=concept_vectors_gemma4_26b_a4b,EXPECTED_LAYERS=30,EXPECTED_D_MODEL=2816,EXPECTED_GPU_KIND=RTX_PRO_6000,SUCCESS_SENTINEL=$SENTINEL_26B,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT"
common_31="MODEL_NAME=google/gemma-4-31B-it,LABEL=stage3b_gemma4_31b,SOURCE_LABEL=gemma4_31b,VECTORS_SUBDIR=concept_vectors_gemma4_31b,EXPECTED_LAYERS=60,EXPECTED_D_MODEL=5376,EXPECTED_GPU_KIND=RTX_PRO_6000,SUCCESS_SENTINEL=$SENTINEL_31B,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT"
final_vars="REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT,SENTINEL_12B=$SENTINEL_12B,SENTINEL_26B=$SENTINEL_26B,SENTINEL_31B=$SENTINEL_31B,FINAL_SENTINEL=$FINAL_SENTINEL"

for spec in \
  "$L40_QUEUE|$L40_RESOURCES|g4_s3b_12|$common_12" \
  "$RTX_QUEUE|$RTX_RESOURCES|g4_s3b_26|$common_26" \
  "$RTX_QUEUE|$RTX_RESOURCES|g4_s3b_31|$common_31"; do
  IFS='|' read -r queue resources name vars <<< "$spec"
  qsub -w v -h -p "$PRIORITY" -q "$queue" -l "$resources" -pe smp 2 \
    -N "$name" -v "$vars" jobs/sge/gemma4_stage3b.sh >/dev/null
done
qsub -w v -p "$PRIORITY" -q scc -l h_vmem=4G,h_rt=00:20:00 -pe smp 1 \
  -N g4_s3b_fin -v "$final_vars" jobs/sge/gemma4_stage3b_finalize.sh >/dev/null

mkdir -p "$STATE_DIR" results/logs
submitted=()
cleanup() {
  status=$?
  if ((status != 0 && ${#submitted[@]})); then qdel "${submitted[@]}" >/dev/null 2>&1 || true; fi
  exit "$status"
}
trap cleanup EXIT
job_12=$(qsub -terse -h -p "$PRIORITY" -q "$L40_QUEUE" -l "$L40_RESOURCES" -pe smp 2 \
  -N g4_s3b_12 -o "results/logs/gemma4_stage3b_${RUN_ID}_12b.out" \
  -e "results/logs/gemma4_stage3b_${RUN_ID}_12b.err" -v "$common_12" jobs/sge/gemma4_stage3b.sh)
submitted+=("$job_12")
job_26=$(qsub -terse -h -p "$PRIORITY" -q "$RTX_QUEUE" -l "$RTX_RESOURCES" -pe smp 2 \
  -N g4_s3b_26 -o "results/logs/gemma4_stage3b_${RUN_ID}_26b.out" \
  -e "results/logs/gemma4_stage3b_${RUN_ID}_26b.err" -v "$common_26" jobs/sge/gemma4_stage3b.sh)
submitted+=("$job_26")
job_31=$(qsub -terse -h -p "$PRIORITY" -q "$RTX_QUEUE" -l "$RTX_RESOURCES" -pe smp 2 \
  -N g4_s3b_31 -o "results/logs/gemma4_stage3b_${RUN_ID}_31b.out" \
  -e "results/logs/gemma4_stage3b_${RUN_ID}_31b.err" -v "$common_31" jobs/sge/gemma4_stage3b.sh)
submitted+=("$job_31")
job_final=$(qsub -terse -p "$PRIORITY" -hold_jid "$job_12,$job_26,$job_31" \
  -q scc -l h_vmem=4G,h_rt=00:20:00 -pe smp 1 -N g4_s3b_fin \
  -o "results/logs/gemma4_stage3b_${RUN_ID}_final.out" \
  -e "results/logs/gemma4_stage3b_${RUN_ID}_final.err" \
  -v "$final_vars" jobs/sge/gemma4_stage3b_finalize.sh)
submitted+=("$job_final")

export RUN_ID STATE_DIR MANIFEST GIT_COMMIT REPO_PATH L40_QUEUE L40_RESOURCES RTX_QUEUE RTX_RESOURCES
export JOB_12="$job_12" JOB_26="$job_26" JOB_31="$job_31" JOB_FINAL="$job_final"
export SENTINEL_12B SENTINEL_26B SENTINEL_31B FINAL_SENTINEL
python - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path

def job(model, label, job_id, queue, resources, sentinel):
    return {
        "model": model, "label": label, "job_id": job_id, "queue": queue,
        "resources": resources, "success_sentinel": sentinel,
        "outputs": [
            f"results/tables/layer_sweep_{label}.csv",
            f"results/tables/layer_sweep_{label}.meta.json",
            f"results/logs/validate_layer_sweep_{label}.json",
        ],
    }
payload = {
    "run_id": os.environ["RUN_ID"],
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "gpu-jobs-user-held", "git_commit": os.environ["GIT_COMMIT"],
    "repo_path": os.environ["REPO_PATH"], "state_dir": os.environ["STATE_DIR"],
    "release_command": f"qrls {os.environ['JOB_12']} {os.environ['JOB_26']} {os.environ['JOB_31']}",
    "jobs": [
        job("google/gemma-4-12B-it", "stage3b_gemma4_12b_l40", os.environ["JOB_12"], os.environ["L40_QUEUE"], os.environ["L40_RESOURCES"], os.environ["SENTINEL_12B"]),
        job("google/gemma-4-26B-A4B-it", "stage3b_gemma4_26b_a4b", os.environ["JOB_26"], os.environ["RTX_QUEUE"], os.environ["RTX_RESOURCES"], os.environ["SENTINEL_26B"]),
        job("google/gemma-4-31B-it", "stage3b_gemma4_31b", os.environ["JOB_31"], os.environ["RTX_QUEUE"], os.environ["RTX_RESOURCES"], os.environ["SENTINEL_31B"]),
        {"role": "finalizer", "job_id": os.environ["JOB_FINAL"], "queue": "scc", "hold_job_ids": [os.environ["JOB_12"], os.environ["JOB_26"], os.environ["JOB_31"]], "success_sentinel": os.environ["FINAL_SENTINEL"]},
    ],
}
Path(os.environ["MANIFEST"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
trap - EXIT
echo "[submitted-held] 12b=$job_12 26b=$job_26 31b=$job_31 finalizer=$job_final"
echo "[manifest] $MANIFEST"
echo "[next] synchronize the manifest, then release: qrls $job_12 $job_26 $job_31"
