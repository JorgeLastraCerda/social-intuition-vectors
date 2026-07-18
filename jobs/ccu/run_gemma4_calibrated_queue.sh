#!/bin/bash
# Serial, fail-closed Gemma 4 CCU queue: 12B -> 26B-A4B -> 31B.
set -euo pipefail

REPO_PATH=${REPO_PATH:-/home/jovyan/work/normalcy-axis}
STATE_ROOT=${STATE_ROOT:-/home/jovyan/work/normalcy-gemma4-state}
STATE_PATH="$STATE_ROOT/queue-state.json"
mkdir -p "$STATE_ROOT/logs"
cd "$REPO_PATH"

update_state() {
  local model=$1 status=$2
  /home/jovyan/.venvs/normalcy-gemma4-cu124/bin/python - "$STATE_PATH" "$model" "$status" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1]); model = sys.argv[2]; status = sys.argv[3]
state = json.loads(path.read_text()) if path.exists() else {
    "order": ["12b", "26b_a4b", "31b"],
    "models": {key: "pending" for key in ["12b", "26b_a4b", "31b"]},
}
state["models"][model] = status
state["updated_at"] = datetime.now(timezone.utc).isoformat()
fd, temporary = tempfile.mkstemp(prefix=".queue-state.", dir=path.parent)
with os.fdopen(fd, "w") as handle:
    json.dump(state, handle, indent=2, sort_keys=True); handle.write("\n")
os.replace(temporary, path)
PY
}

for model in 12b 26b_a4b 31b; do
  update_state "$model" running
  log="$STATE_ROOT/logs/${model}.log"
  if bash jobs/ccu/run_gemma4_calibrated.sh "$model" 2>&1 | tee -a "$log"; then
    update_state "$model" completed
  else
    update_state "$model" failed
    echo "[queue] stopped after technical failure in $model" >&2
    exit 1
  fi
done
echo "[queue] all Gemma 4 calibrated runs completed"
