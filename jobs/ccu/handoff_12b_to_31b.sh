#!/bin/bash
# Wait for CCU 12B success, retire its stopped coordinator, then run only 31B.
set -euo pipefail

REPO_PATH=${REPO_PATH:-/home/jovyan/work/normalcy-axis}
STATE_ROOT=${STATE_ROOT:-/home/jovyan/work/normalcy-gemma4-state}
COORDINATOR_PID=${COORDINATOR_PID:-1197}
SENTINEL="$STATE_ROOT/sentinels/gemma4_12b_calibrated_ccu_h100.success"
STATE_PATH="$STATE_ROOT/queue-state.json"
PYTHON=/home/jovyan/.venvs/normalcy-gemma4-cu124/bin/python

update_models() {
  "$PYTHON" - "$STATE_PATH" "$@" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
updates = dict(argument.split("=", 1) for argument in sys.argv[2:])
state = json.loads(path.read_text(encoding="utf-8"))
state["models"].update(updates)
state["updated_at"] = datetime.now(timezone.utc).isoformat()
fd, temporary = tempfile.mkstemp(prefix=".queue-state.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(state, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
}

while [[ ! -f "$SENTINEL" ]]; do sleep 15; done

# SIGTERM remains pending while the coordinator is stopped. SIGCONT delivers it
# before the shell can advance from the completed 12B pipeline to 26B.
kill -TERM "$COORDINATOR_PID" 2>/dev/null || true
kill -CONT "$COORDINATOR_PID" 2>/dev/null || true
sleep 2
update_models 12b=completed 26b_a4b=delegated_scckn 31b=running

cd "$REPO_PATH"
if bash jobs/ccu/run_gemma4_calibrated.sh 31b \
  >> "$STATE_ROOT/logs/31b.log" 2>&1; then
  update_models 31b=completed
else
  update_models 31b=failed
  exit 1
fi
