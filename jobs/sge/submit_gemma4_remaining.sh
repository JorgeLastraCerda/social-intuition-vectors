#!/bin/bash
# Submit one independent pinned Gemma 4 run held. No hold_jid is ever used.
set -euo pipefail

usage() {
  echo "usage: bash jobs/sge/submit_gemma4_remaining.sh --model {12b|26b_a4b|31b} --run RUN [--full282] [--dry-run]" >&2
}

MODEL=""
RUN_NAME=""
FULL282=0
DRY_RUN=0
while (($#)); do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --run) RUN_NAME="${2:-}"; shift 2 ;;
    --full282) FULL282=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) usage; exit 2 ;;
  esac
done
[[ "$MODEL" =~ ^(12b|26b_a4b|31b)$ ]] || { usage; exit 2; }
[[ "$RUN_NAME" =~ ^(smoke|neutral|pca|dense_raw|dense_denoised|audit|hiring_local|hiring_broad|hiring_denoised|posthoc|full282_gate)$ ]] || { usage; exit 2; }
if ((FULL282)) && [[ ! "$RUN_NAME" =~ ^hiring_(local|broad|denoised)$ ]]; then
  echo "--full282 is valid only for hiring steering runs." >&2
  exit 2
fi

cd "$(git rev-parse --show-toplevel)"
REPO_PATH="$(pwd)"
CONFIG_PATH="config/gemma4_${MODEL}.yaml"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="/work/emrecan.ulu/gemma4_remaining/$RUN_ID"
suffix=""
if ((FULL282)); then suffix="_full282"; fi
MANIFEST="results/logs/gemma4_remaining_submission_${MODEL}_${RUN_NAME}${suffix}_${RUN_ID}.json"
GIT_COMMIT="$(git rev-parse HEAD)"
GPU_RUNS='smoke|neutral|dense_raw|dense_denoised|audit|hiring_local|hiring_broad|hiring_denoised'

EXPECTED_GPU_KIND=CPU
QUEUE=scc
RESOURCES="h_vmem=32G,h_rt=02:00:00"
PE=4
if [[ "$RUN_NAME" =~ ^($GPU_RUNS)$ ]]; then
  PE=2
  if [[ "$MODEL" == "12b" ]]; then
    EXPECTED_GPU_KIND=L40
    QUEUE=gpu@scc192
    RESOURCES="gpu=1,h_vmem=32G,h_rt=12:00:00"
  else
    EXPECTED_GPU_KIND=RTX_PRO_6000
    QUEUE=gpu@scc214
    RESOURCES="gpu=1,rtx_6000=1,h_vmem=96G,h_rt=12:00:00"
  fi
fi
if [[ "$RUN_NAME" == "smoke" ]]; then
  RESOURCES=${RESOURCES/h_rt=12:00:00/h_rt=01:00:00}
fi

if ((DRY_RUN)); then
  echo "[dry-run] model=$MODEL config=$CONFIG_PATH run=$RUN_NAME full282=$FULL282"
  echo "[dry-run] queue=$QUEUE resources=$RESOURCES expected_gpu=$EXPECTED_GPU_KIND"
  echo "[dry-run] independent=1 user_held=1 hold_jid=none availability=submit-even-when-zero"
  exit 0
fi
critical_paths=(
  "$CONFIG_PATH"
  src
  smoke_tests/gemma4_transformerlens/smoke_test_bridge.py
  jobs/sge/gemma4_remaining_run.sh
  jobs/sge/submit_gemma4_remaining.sh
)
if [[ -n "$(git status --porcelain --untracked-files=no -- "${critical_paths[@]}")" ]]; then
  echo "Refusing submission: tracked critical source/config files are not clean." >&2
  exit 3
fi
for path in "$MANIFEST" "$STATE_DIR"; do
  [[ ! -e "$path" ]] || { echo "Refusing existing path: $path" >&2; exit 4; }
done

module load conda  # ADJUST
PYTHON=(conda run --no-capture-output -n wc-tl-g4 python)
"${PYTHON[@]}" -m pip check
validate_args=(--config "$CONFIG_PATH" --run "$RUN_NAME" --require-absent)
if ((FULL282)); then validate_args+=(--full282); fi
"${PYTHON[@]}" -m src.validate_gemma4_remaining "${validate_args[@]}"

available="n/a"
if [[ "$EXPECTED_GPU_KIND" != "CPU" ]]; then
  available=$(qstat -F gpu -q "$QUEUE" | awk -F= '/qc:gpu=/{gsub(/[[:space:]]/, "", $2); total += $2} END {print total+0}')
  [[ "$available" =~ ^[0-9]+$ ]] || { echo "Could not parse GPU availability." >&2; exit 5; }
fi

mkdir -p "$STATE_DIR" results/logs
sentinel="$STATE_DIR/${MODEL}_${RUN_NAME}${suffix}.success"
vars="RUN_NAME=$RUN_NAME,CONFIG_PATH=$CONFIG_PATH,FULL282=$FULL282,EXPECTED_GPU_KIND=$EXPECTED_GPU_KIND,SUCCESS_SENTINEL=$sentinel,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT"
qsub -w v -h -q "$QUEUE" -l "$RESOURCES" -pe smp "$PE" -v "$vars" \
  jobs/sge/gemma4_remaining_run.sh >/dev/null
job=$(qsub -terse -h -q "$QUEUE" -l "$RESOURCES" -pe smp "$PE" \
  -N "g4_${MODEL}_${RUN_NAME:0:7}" \
  -o "results/logs/gemma4_${MODEL}_${RUN_NAME}${suffix}_${RUN_ID}.out" \
  -e "results/logs/gemma4_${MODEL}_${RUN_NAME}${suffix}_${RUN_ID}.err" \
  -v "$vars" jobs/sge/gemma4_remaining_run.sh)

export RUN_ID MODEL RUN_NAME FULL282 STATE_DIR MANIFEST GIT_COMMIT CONFIG_PATH
export QUEUE RESOURCES EXPECTED_GPU_KIND AVAILABLE="$available" JOB_ID_SUBMITTED="$job" SENTINEL="$sentinel"
"${PYTHON[@]}" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
from src.utils.config import load_config

cfg = load_config(os.environ["CONFIG_PATH"])
payload = {
    "run_id": os.environ["RUN_ID"],
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "independent-job-user-held",
    "git_commit": os.environ["GIT_COMMIT"],
    "hold_jid_used": False,
    "availability_at_submission": os.environ["AVAILABLE"],
    "job": {
        "job_id": os.environ["JOB_ID_SUBMITTED"],
        "run": os.environ["RUN_NAME"],
        "full282": os.environ["FULL282"] == "1",
        "model": cfg.model.name,
        "revision": cfg.model.revision,
        "config": os.environ["CONFIG_PATH"],
        "queue": os.environ["QUEUE"],
        "resources": os.environ["RESOURCES"],
        "expected_gpu_kind": os.environ["EXPECTED_GPU_KIND"],
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
