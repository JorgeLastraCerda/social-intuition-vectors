#!/bin/bash
# Keep CCU active while serially closing larger Gemma 4 remaining tests.
set -euo pipefail

REPO_PATH=${REPO_PATH:-/home/jovyan/work/normalcy-axis}
VENV_PATH=${VENV_PATH:-/home/jovyan/.venvs/normalcy-gemma4-cu124}
STATE_ROOT=${STATE_ROOT:-/home/jovyan/work/normalcy-gemma4-remaining}
RUNNER_PATH=${RUNNER_PATH:-$STATE_ROOT/run_gemma4_remaining.sh}
ACTIVE_31B_PID=${ACTIVE_31B_PID:-5555}
PYTHON="$VENV_PATH/bin/python"
export PYTHONPATH="$REPO_PATH"

wait_for_active_31b() {
  local sentinel="$STATE_ROOT/sentinels/gemma4_31b_denoised_local.success"
  while [[ ! -f "$sentinel" ]]; do
    if ! kill -0 "$ACTIVE_31B_PID" 2>/dev/null; then
      echo "31B denoised prerequisite exited without sentinel." >&2
      exit 20
    fi
    sleep 10
  done
}

run_gate_31b() {
  cd "$REPO_PATH"
  "$PYTHON" -m src.validate_gemma4_remaining \
    --config config/gemma4_31b.yaml --run full282_gate --require-absent
  "$PYTHON" -m src.summarize_hiring_steering gate \
    --config config/gemma4_31b.yaml --model-label gemma4_31b
  "$PYTHON" -m src.validate_gemma4_remaining \
    --config config/gemma4_31b.yaml --run full282_gate
}

gate_decision() {
  "$PYTHON" -c 'import json, sys; print(str(json.load(open(sys.argv[1]))["run_full_282"]).lower())' "$1"
}

run_expansion() {
  local model=$1
  local task
  for task in local_full282 broad_full282 denoised_full282; do
    echo "[queue] start model=$model task=$task $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    bash "$RUNNER_PATH" "$model" "$task" \
      > "$STATE_ROOT/logs/${model}_${task}.log" 2>&1
    echo "[queue] complete model=$model task=$task $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  done
}

wait_for_active_31b
run_gate_31b > "$STATE_ROOT/logs/31b_full282_gate.log" 2>&1

gate26="$REPO_PATH/results/logs/hiring_full282_gate_gemma4_26b_a4b.json"
gate31="$REPO_PATH/results/logs/hiring_full282_gate_gemma4_31b.json"
[[ "$(gate_decision "$gate26")" == true ]] && run_expansion 26b_a4b
[[ "$(gate_decision "$gate31")" == true ]] && run_expansion 31b
echo "[queue] all required larger-model remaining tests complete"
