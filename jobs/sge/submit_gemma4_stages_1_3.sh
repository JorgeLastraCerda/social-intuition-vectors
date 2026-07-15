#!/bin/bash
# Submit the nine Gemma 4 Stage 1--3 jobs in one fully serial stage-major chain.
set -euo pipefail

MODE="${1:-}"
if [[ -n "$MODE" && "$MODE" != "--dry-run" ]]; then
  echo "usage: bash jobs/sge/submit_gemma4_stages_1_3.sh [--dry-run]" >&2
  exit 2
fi
DRY_RUN=0
if [[ "$MODE" == "--dry-run" ]]; then DRY_RUN=1; fi

cd "$(git rev-parse --show-toplevel)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
STATE_DIR="/work/emrecan.ulu/gemma4_stage_state/$RUN_ID"
MANIFEST="results/logs/gemma4_stages_1_3_submission_${RUN_ID}.json"

models=(
  "12b|google/gemma-4-12B-it|gemma4_12b|concept_vectors_gemma4_12b|48|3840|gpu@scc192,gpu@scc213,gpu@scc214|32G"
  "26b|google/gemma-4-26B-A4B-it|gemma4_26b_a4b|concept_vectors_gemma4_26b_a4b|30|2816|gpu@scc214|96G"
  "31b|google/gemma-4-31B-it|gemma4_31b|concept_vectors_gemma4_31b|60|5376|gpu@scc214|96G"
)

check_collision() {
  local path="$1"
  if [[ -e "$path" ]]; then
    echo "Refusing submission: output already exists: $path" >&2
    exit 3
  fi
}

for spec in "${models[@]}"; do
  IFS='|' read -r short model label vectors layers d_model queue h_vmem <<<"$spec"
  check_collision "data/processed/$vectors"
  check_collision "results/tables/probe_metrics_${label}.csv"
  check_collision "results/logs/validate_probes_${label}.json"
  check_collision "results/tables/layer_sweep_${label}.csv"
  check_collision "results/tables/layer_sweep_${label}.meta.json"
done
check_collision "$MANIFEST"

if ((DRY_RUN == 0)); then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing submission: SCCKN worktree is not clean." >&2
    exit 4
  fi
  if pgrep -u "$USER" -f 'pip(3)? install|conda (create|install|update)' >/dev/null; then
    echo "Refusing submission: a pip/conda mutation is still running for $USER." >&2
    exit 5
  fi
  conda run -n wc-tl-g4 python -m pip check
  conda run -n wc-tl-g4 python -c \
    'from importlib.metadata import version; import torch, transformers; tl=version("transformer-lens"); print(f"torch={torch.__version__} transformers={transformers.__version__} transformer-lens={tl}")'
  git config user.name >/dev/null
  git config user.email >/dev/null
  mkdir -p "$STATE_DIR" results/logs
fi

previous_job=""
previous_sentinel=""
ordinal=0
job_rows=""
first_job=""

for stage in 1 2 3; do
  for spec in "${models[@]}"; do
    IFS='|' read -r short model label vectors layers d_model queue h_vmem <<<"$spec"
    ordinal=$((ordinal + 1))
    printf -v ordinal_padded '%02d' "$ordinal"
    success_sentinel="$STATE_DIR/${ordinal_padded}_s${stage}_${short}.success"
    log_base="results/logs/gemma4_${RUN_ID}_s${stage}_${short}"
    h_rt="12:00:00"
    if [[ "$stage" == "1" ]]; then h_rt="08:00:00"; fi
    if [[ "$stage" == "2" ]]; then h_rt="01:00:00"; fi
    vars="STAGE=$stage,MODEL_NAME=$model,LABEL=$label,VECTORS_SUBDIR=$vectors,EXPECTED_LAYERS=$layers,EXPECTED_D_MODEL=$d_model,PREDECESSOR_SENTINEL=$previous_sentinel,SUCCESS_SENTINEL=$success_sentinel"

    if ((DRY_RUN)); then
      job="DRY$(printf '%03d' "$ordinal")"
      echo "$ordinal: stage=$stage model=$short hold=${previous_job:-none} queue=$queue h_vmem=$h_vmem h_rt=$h_rt -> $job"
    else
      hold_args=()
      if [[ -n "$previous_job" ]]; then hold_args=(-hold_jid "$previous_job"); fi
      initial_hold=()
      if [[ "$ordinal" == "1" ]]; then initial_hold=(-h); fi
      job=$(qsub -terse "${initial_hold[@]}" "${hold_args[@]}" \
        -q "$queue" -l "h_vmem=$h_vmem,h_rt=$h_rt" \
        -N "g4_s${stage}_${short}" -o "${log_base}.out" -e "${log_base}.err" \
        -v "$vars" jobs/sge/gemma4_stage_1_3.sh)
      echo "$ordinal: stage=$stage model=$short hold=${previous_job:-user-hold} -> $job"
    fi

    if [[ -z "$first_job" ]]; then first_job="$job"; fi
    job_rows+="$ordinal|$stage|$short|$model|$label|$vectors|$job|$previous_job|$queue|$h_vmem|$h_rt|$success_sentinel"$'\n'
    previous_job="$job"
    previous_sentinel="$success_sentinel"
  done
done

if ((DRY_RUN)); then
  echo "[dry-run] run_id=$RUN_ID first_job=$first_job jobs=$ordinal"
  exit 0
fi

export RUN_ID STATE_DIR MANIFEST FIRST_JOB="$first_job" JOB_ROWS="$job_rows"
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

jobs = []
for line in os.environ["JOB_ROWS"].splitlines():
    if not line:
        continue
    fields = line.split("|")
    jobs.append(
        dict(
            ordinal=int(fields[0]),
            stage=int(fields[1]),
            model_short=fields[2],
            model=fields[3],
            label=fields[4],
            vectors_subdir=fields[5],
            job_id=fields[6],
            hold_job_id=fields[7] or None,
            queue=fields[8],
            h_vmem=fields[9],
            h_rt=fields[10],
            success_sentinel=fields[11],
        )
    )
payload = {
    "run_id": os.environ["RUN_ID"],
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "ordering": "fully-serial-stage-major",
    "status": "submitted-first-job-user-held",
    "first_job_id": os.environ["FIRST_JOB"],
    "state_dir": os.environ["STATE_DIR"],
    "jobs": jobs,
}
Path(os.environ["MANIFEST"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

echo "[submitted] manifest=$MANIFEST first_job=$first_job jobs=$ordinal"
echo "[next] sync the manifest and submission step log, then run: qrls $first_job"
