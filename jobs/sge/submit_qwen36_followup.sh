#!/bin/bash
# Submit independent CPU Stage 2 and RTX Stage 3 jobs for one validated model.
set -euo pipefail

CONFIG_PATH="${1:-}"
MODE="${2:-}"
if [[ -z "$CONFIG_PATH" || ( -n "$MODE" && "$MODE" != "--dry-run" ) ]]; then
  echo "usage: bash jobs/sge/submit_qwen36_followup.sh <config.yaml> [--dry-run]" >&2
  exit 2
fi
DRY_RUN=0
if [[ "$MODE" == "--dry-run" ]]; then DRY_RUN=1; fi

cd "$(git rev-parse --show-toplevel)"
REPO_PATH="$(pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GIT_COMMIT="$(git rev-parse HEAD)"
GPU_QUEUE="gpu@scc214"
GPU_RESOURCES="gpu=1,rtx_6000=1,h_vmem=96G,h_rt=04:00:00"
CPU_RESOURCES="h_vmem=8G,h_rt=00:15:00"

if ((DRY_RUN)); then
  echo "[dry-run] config=$CONFIG_PATH independent stages=2,3 hold_jid=none"
  echo "[dry-run] stage2 queue=scc gpu=0 resources=$CPU_RESOURCES"
  echo "[dry-run] stage3 queue=$GPU_QUEUE resources=$GPU_RESOURCES"
  exit 0
fi
[[ -z "$(git status --porcelain)" ]] || { echo "SCCKN worktree is not clean." >&2; exit 3; }
module load conda  # ADJUST
PYTHON=(conda run --no-capture-output -n wc-qwen36-hf python)
label=$("${PYTHON[@]}" -c 'import sys; from src.utils.config import load_config; print(load_config(sys.argv[1]).native_hf.label)' "$CONFIG_PATH")
STATE_DIR="/work/emrecan.ulu/qwen36_full/$RUN_ID"
MANIFEST="results/logs/qwen36_full_submission_${label}_followup_${RUN_ID}.json"
for path in "$STATE_DIR" "$MANIFEST"; do [[ ! -e "$path" ]] || { echo "Existing path: $path" >&2; exit 4; }; done
"${PYTHON[@]}" -m src.validate_qwen36_stage --config "$CONFIG_PATH" --stage 1
for stage in 2 3; do
  "${PYTHON[@]}" -m src.validate_qwen36_stage \
    --config "$CONFIG_PATH" --stage "$stage" --require-absent
done
available=$(qstat -F gpu -q "$GPU_QUEUE" | awk -F= '/qc:gpu=/{gsub(/[[:space:]]/, "", $2); print $2; exit}')
[[ "$available" =~ ^[0-9]+$ ]] || { echo "Could not parse RTX availability." >&2; exit 5; }
mkdir -p "$STATE_DIR" results/logs

jobs=()
cleanup() { status=$?; if ((status != 0 && ${#jobs[@]})); then qdel "${jobs[@]}" >/dev/null 2>&1 || true; fi; exit "$status"; }
trap cleanup EXIT
for stage in 2 3; do
  sentinel="$STATE_DIR/${label}.stage${stage}.success"
  vars="STAGE=$stage,CONFIG_PATH=$CONFIG_PATH,SUCCESS_SENTINEL=$sentinel,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT"
  if [[ "$stage" == "2" ]]; then
    queue="scc"; resources="$CPU_RESOURCES"; slots=1
  else
    queue="$GPU_QUEUE"; resources="$GPU_RESOURCES"; slots=2
  fi
  qsub -w v -h -q "$queue" -l "$resources" -pe smp "$slots" -v "$vars" \
    jobs/sge/qwen36_stage.sh >/dev/null
  job=$(qsub -terse -h -q "$queue" -l "$resources" -pe smp "$slots" \
    -N "${label}_s${stage}" \
    -o "results/logs/${label}_stage${stage}_${RUN_ID}.out" \
    -e "results/logs/${label}_stage${stage}_${RUN_ID}.err" \
    -v "$vars" jobs/sge/qwen36_stage.sh)
  jobs+=("$job")
done

export CONFIG_PATH RUN_ID STATE_DIR MANIFEST GIT_COMMIT GPU_QUEUE GPU_RESOURCES CPU_RESOURCES
export AVAILABLE="$available" LABEL="$label" JOB_STAGE2="${jobs[0]}" JOB_STAGE3="${jobs[1]}"
"${PYTHON[@]}" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
from src.utils.config import load_config

cfg = load_config(os.environ["CONFIG_PATH"])
state = Path(os.environ["STATE_DIR"])
jobs = []
for stage, job_id, queue, resources in (
    (2, os.environ["JOB_STAGE2"], "scc", os.environ["CPU_RESOURCES"]),
    (3, os.environ["JOB_STAGE3"], os.environ["GPU_QUEUE"], os.environ["GPU_RESOURCES"]),
):
    jobs.append({
        "job_id": job_id, "stage": stage, "queue": queue, "resources": resources,
        "gpu_request": 0 if stage == 2 else 1, "user_held": True,
        "success_sentinel": str(state / f"{cfg.native_hf.label}.stage{stage}.success"),
    })
payload = {
    "run_id": os.environ["RUN_ID"],
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "independent-followup-jobs-user-held",
    "model": cfg.model.name, "revision": cfg.model.revision,
    "config": os.environ["CONFIG_PATH"], "git_commit": os.environ["GIT_COMMIT"],
    "availability_at_submission": int(os.environ["AVAILABLE"]),
    "hold_jid_used": False, "jobs": jobs,
    "release_command": f"qrls {os.environ['JOB_STAGE2']} {os.environ['JOB_STAGE3']}",
}
Path(os.environ["MANIFEST"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
trap - EXIT
echo "[submitted-held] model=$label stage2=${jobs[0]} stage3=${jobs[1]} availability=$available"
echo "[manifest] $MANIFEST"
echo "[next] synchronize manifest, then: qrls ${jobs[*]}"
