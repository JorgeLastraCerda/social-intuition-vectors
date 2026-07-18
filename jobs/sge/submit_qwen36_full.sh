#!/bin/bash
# Submit the two independent Qwen3.6 Stage 1 jobs held; no scheduler dependency.
set -euo pipefail

MODE="${1:-}"
if [[ -n "$MODE" && "$MODE" != "--dry-run" ]]; then
  echo "usage: bash jobs/sge/submit_qwen36_full.sh [--dry-run]" >&2
  exit 2
fi
DRY_RUN=0
if [[ "$MODE" == "--dry-run" ]]; then DRY_RUN=1; fi

cd "$(git rev-parse --show-toplevel)"
REPO_PATH="$(pwd)"
CONFIGS=(config/qwen36_27b.yaml config/qwen36_35b_a3b.yaml)
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="/work/emrecan.ulu/qwen36_full/$RUN_ID"
MANIFEST="results/logs/qwen36_full_submission_stage1_${RUN_ID}.json"
GIT_COMMIT="$(git rev-parse HEAD)"
QUEUE="gpu@scc214"
RESOURCES="gpu=1,rtx_6000=1,h_vmem=96G,h_rt=04:00:00"

if ((DRY_RUN)); then
  echo "[dry-run] independent held Stage 1 jobs configs=${CONFIGS[*]}"
  echo "[dry-run] queue=$QUEUE resources=$RESOURCES availability=submit-even-when-zero"
  echo "[dry-run] hold_jid=none predecessor_sentinel=none"
  exit 0
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing submission: SCCKN worktree is not clean." >&2
  exit 3
fi
for path in "$MANIFEST" "$STATE_DIR"; do
  [[ ! -e "$path" ]] || { echo "Refusing existing path: $path" >&2; exit 4; }
done

module load conda  # ADJUST
PYTHON=(conda run --no-capture-output -n wc-qwen36-hf python)
"${PYTHON[@]}" -m pip check
for config in "${CONFIGS[@]}"; do
  "${PYTHON[@]}" -m src.qwen36_pipeline --config "$config" --stage 1 --dry-run >/dev/null
  "${PYTHON[@]}" -m src.validate_qwen36_stage \
    --config "$config" --stage 1 --require-absent
done
available=$(qstat -F gpu -q "$QUEUE" | awk -F= '/qc:gpu=/{gsub(/[[:space:]]/, "", $2); print $2; exit}')
[[ "$available" =~ ^[0-9]+$ ]] || { echo "Could not parse RTX availability." >&2; exit 5; }
echo "[preflight] $QUEUE available=$available; both submissions will proceed"

mkdir -p "$STATE_DIR" results/logs
jobs=()
cleanup() {
  status=$?
  if ((status != 0 && ${#jobs[@]})); then qdel "${jobs[@]}" >/dev/null 2>&1 || true; fi
  exit "$status"
}
trap cleanup EXIT

for config in "${CONFIGS[@]}"; do
  label=$("${PYTHON[@]}" -c 'import sys; from src.utils.config import load_config; print(load_config(sys.argv[1]).native_hf.label)' "$config")
  sentinel="$STATE_DIR/${label}.stage1.success"
  vars="STAGE=1,CONFIG_PATH=$config,SUCCESS_SENTINEL=$sentinel,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT"
  qsub -w v -h -q "$QUEUE" -l "$RESOURCES" -pe smp 2 -v "$vars" \
    jobs/sge/qwen36_stage.sh >/dev/null
  job=$(qsub -terse -h -q "$QUEUE" -l "$RESOURCES" -pe smp 2 \
    -N "${label}_s1" \
    -o "results/logs/${label}_stage1_${RUN_ID}.out" \
    -e "results/logs/${label}_stage1_${RUN_ID}.err" \
    -v "$vars" jobs/sge/qwen36_stage.sh)
  jobs+=("$job")
done

export RUN_ID STATE_DIR MANIFEST GIT_COMMIT REPO_PATH QUEUE RESOURCES AVAILABLE="$available"
export JOB_27B="${jobs[0]}" JOB_35B="${jobs[1]}"
"${PYTHON[@]}" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
from src.utils.config import load_config

configs = ["config/qwen36_27b.yaml", "config/qwen36_35b_a3b.yaml"]
job_ids = [os.environ["JOB_27B"], os.environ["JOB_35B"]]
state = Path(os.environ["STATE_DIR"])
payload = {
    "run_id": os.environ["RUN_ID"],
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "independent-stage1-jobs-user-held",
    "git_commit": os.environ["GIT_COMMIT"],
    "availability_at_submission": int(os.environ["AVAILABLE"]),
    "hold_jid_used": False,
    "jobs": [],
}
for config, job_id in zip(configs, job_ids):
    cfg = load_config(config)
    payload["jobs"].append({
        "job_id": job_id, "stage": 1, "model": cfg.model.name,
        "revision": cfg.model.revision, "config": config,
        "queue": os.environ["QUEUE"], "resources": os.environ["RESOURCES"],
        "user_held": True,
        "success_sentinel": str(state / f"{cfg.native_hf.label}.stage1.success"),
    })
payload["release_command"] = "qrls " + " ".join(job_ids)
Path(os.environ["MANIFEST"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

trap - EXIT
echo "[submitted-held] jobs=${jobs[*]} availability=$available"
echo "[manifest] $MANIFEST"
echo "[next] synchronize manifest, then: qrls ${jobs[*]}"
