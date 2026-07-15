#!/bin/bash
# Submit one held, first-fit Gemma 4 12B parity job to the SCCKN L40 pool.
set -euo pipefail

MODE="${1:-}"
if [[ -n "$MODE" && "$MODE" != "--dry-run" ]]; then
  echo "usage: bash jobs/sge/submit_gemma4_12b_multigpu_parity.sh [--dry-run]" >&2
  exit 2
fi

cd "$(git rev-parse --show-toplevel)"
REPO_PATH="$(pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="/work/emrecan.ulu/gemma4_parity/$RUN_ID"
MANIFEST="results/logs/gemma4_parity_submission_${RUN_ID}.json"
OUTPUT="results/logs/gemma4_parity_12b_${RUN_ID}.json"
QUEUE="gpu@scc192,gpu@scc213"
GIT_COMMIT="$(git rev-parse HEAD)"

for path in "$MANIFEST" "$OUTPUT" "$STATE_DIR"; do
  if [[ -e "$path" ]]; then
    echo "Refusing submission: path already exists: $path" >&2
    exit 3
  fi
done

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing submission: isolated SCCKN checkout is not clean." >&2
  exit 4
fi

conda run -n wc-tl-g4 python -m pip check
conda run -n wc-tl-g4 python -c \
  'from importlib.metadata import version; import torch, transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} transformer-lens={version(\"transformer-lens\")}")'

if [[ "$MODE" == "--dry-run" ]]; then
  echo "run_id=$RUN_ID queue=$QUEUE gpu=2 held=yes repo=$REPO_PATH commit=$GIT_COMMIT"
  exit 0
fi

vars="RUN_ID=$RUN_ID,STATE_DIR=$STATE_DIR,REPO_PATH=$REPO_PATH,GIT_COMMIT=$GIT_COMMIT"
job_id=$(qsub -terse -h \
  -q "$QUEUE" -l "gpu=2,h_vmem=32G,h_rt=01:00:00" -pe smp 2 \
  -N "g4p_12b_${RUN_ID:9:4}" \
  -o "results/logs/gemma4_parity_${RUN_ID}.out" \
  -e "results/logs/gemma4_parity_${RUN_ID}.err" \
  -v "$vars" jobs/sge/gemma4_12b_multigpu_parity.sh)

export RUN_ID STATE_DIR MANIFEST OUTPUT QUEUE GIT_COMMIT REPO_PATH JOB_ID="$job_id"
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "run_id": os.environ["RUN_ID"],
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "submitted-user-held",
    "job_id": os.environ["JOB_ID"],
    "model_source": "data/processed/concept_vectors_gemma4_12b/meta.json",
    "queue": os.environ["QUEUE"],
    "gpu_request": 2,
    "host_policy": "single-first-fit-job-on-one-L40-host",
    "git_commit": os.environ["GIT_COMMIT"],
    "repo_path": os.environ["REPO_PATH"],
    "state_dir": os.environ["STATE_DIR"],
    "result_json": os.environ["OUTPUT"],
}
Path(os.environ["MANIFEST"]).write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
PY

echo "[submitted-held] job=$job_id manifest=$MANIFEST"
echo "[next] commit/push the manifest and submission step log, pull this checkout, then: qrls $job_id"
