#!/bin/bash
# Submit one write-once, independent, user-held calibrated-steering pilot.
set -euo pipefail

usage() {
  echo "usage: bash jobs/sge/submit_calibrated_steering_pilot.sh --model {gemma3_12b|gemma4_12b|qwen36_27b} [--dry-run]" >&2
}

MODEL_KEY=""
DRY_RUN=0
while (($#)); do
  case "$1" in
    --model) MODEL_KEY="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) usage; exit 2 ;;
  esac
done
[[ "$MODEL_KEY" =~ ^(gemma3_12b|gemma4_12b|qwen36_27b)$ ]] || { usage; exit 2; }

cd "$(git rev-parse --show-toplevel)"
REPO_PATH="$(pwd)"
GIT_COMMIT="$(git rev-parse HEAD)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="/work/emrecan.ulu/calibrated_steering/$RUN_ID"
MANIFEST="results/logs/calibrated_steering_submission_${MODEL_KEY}_${RUN_ID}.json"
QUEUE=gpu@scc214
RESOURCES="gpu=1,rtx_6000=1,h_vmem=96G,h_rt=24:00:00"

case "$MODEL_KEY" in
  gemma3_12b) CONFIG_PATH=config/config.yaml; LABEL=gemma3_12b_calibrated ;;
  gemma4_12b) CONFIG_PATH=config/gemma4_12b.yaml; LABEL=gemma4_12b_calibrated ;;
  qwen36_27b) CONFIG_PATH=config/qwen36_27b.yaml; LABEL=qwen36_27b_calibrated ;;
esac

if ((DRY_RUN)); then
  echo "[dry-run] model=$MODEL_KEY config=$CONFIG_PATH label=$LABEL"
  echo "[dry-run] queue=$QUEUE resources=$RESOURCES expected_gpu=RTX_PRO_6000"
  echo "[dry-run] independent=1 user_held=1 hold_jid=none full282=disabled"
  exit 0
fi

critical_paths=(
  "$CONFIG_PATH" src/steering_calibration.py src/dense_steering.py
  src/qwen36_calibrated_steering.py src/validate_calibrated_steering.py
  jobs/sge/calibrated_steering_run.sh jobs/sge/submit_calibrated_steering_pilot.sh
)
if [[ -n "$(git status --porcelain --untracked-files=no -- "${critical_paths[@]}")" ]]; then
  echo "Refusing submission: tracked critical calibrated-steering files are not clean." >&2
  exit 3
fi
for path in "$MANIFEST" "$STATE_DIR"; do
  [[ ! -e "$path" ]] || { echo "Refusing existing path: $path" >&2; exit 4; }
done

available=$(qstat -F gpu -q "$QUEUE" | awk -F= '/qc:gpu=/{gsub(/[[:space:]]/, "", $2); total += $2} END {print total+0}')
[[ "$available" =~ ^[0-9]+$ ]] || { echo "Could not parse GPU availability." >&2; exit 5; }
mkdir -p "$STATE_DIR" results/logs
sentinel="$STATE_DIR/${MODEL_KEY}.success"
vars="MODEL_KEY=$MODEL_KEY,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT,SUCCESS_SENTINEL=$sentinel"
job=$(qsub -terse -h -q "$QUEUE" -l "$RESOURCES" -pe smp 2 \
  -N "cal_${MODEL_KEY:0:8}" \
  -o "results/logs/calibrated_steering_${MODEL_KEY}_${RUN_ID}.out" \
  -e "results/logs/calibrated_steering_${MODEL_KEY}_${RUN_ID}.err" \
  -v "$vars" jobs/sge/calibrated_steering_run.sh)

export RUN_ID MODEL_KEY CONFIG_PATH LABEL GIT_COMMIT QUEUE RESOURCES
export AVAILABLE="$available" JOB_ID_SUBMITTED="$job" MANIFEST SENTINEL="$sentinel"
python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "run_id": os.environ["RUN_ID"],
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "independent-job-user-held",
    "git_commit": os.environ["GIT_COMMIT"],
    "hold_jid_used": False,
    "full282_enabled": False,
    "availability_at_submission": int(os.environ["AVAILABLE"]),
    "job": {
        "job_id": os.environ["JOB_ID_SUBMITTED"],
        "model_key": os.environ["MODEL_KEY"],
        "config": os.environ["CONFIG_PATH"],
        "label": os.environ["LABEL"],
        "queue": os.environ["QUEUE"],
        "resources": os.environ["RESOURCES"],
        "expected_gpu_kind": "RTX_PRO_6000",
        "user_held": True,
        "success_sentinel": os.environ["SENTINEL"],
    },
    "release_command": f"qrls {os.environ['JOB_ID_SUBMITTED']}",
}
Path(os.environ["MANIFEST"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

echo "[submitted-held] job=$job availability=$available"
echo "[manifest] $MANIFEST"
echo "[next] synchronize manifest, then: qrls $job"
