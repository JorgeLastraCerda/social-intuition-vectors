#!/bin/bash
# Submit gated Gemma 4 chains. By default submits smoke jobs only; after reviewing
# their JSON outputs, rerun with --full to submit each model's dependent pipeline.
set -euo pipefail

MODE="${1:---smoke}"
if [[ "$MODE" != "--smoke" && "$MODE" != "--full" ]]; then
  echo "usage: bash jobs/sge/submit_gemma4.sh [--smoke|--full]" >&2
  exit 2
fi

submit_model() {
  local model="$1" label="$2" vectors="$3" layers="$4" d_model="$5" initial_hold="${6:-}"
  local vars="MODEL_NAME=$model,LABEL=$label,VECTORS_SUBDIR=$vectors,EXPECTED_LAYERS=$layers,EXPECTED_D_MODEL=$d_model"
  local hold_args=()
  if [[ -n "$initial_hold" ]]; then hold_args=(-hold_jid "$initial_hold"); fi
  if [[ "$MODE" == "--smoke" ]]; then
    LAST_SMOKE_JOB=$(qsub -terse "${hold_args[@]}" -N "g4_smoke_${label}" \
      -o "results/logs/smoke_${label}.out" \
      -e "results/logs/smoke_${label}.err" -v "$vars" jobs/sge/gemma4_smoke.sh)
    echo "$label: smoke=$LAST_SMOKE_JOB"
    return
  fi

  local core neutral dense hiring
  core=$(qsub -terse "${hold_args[@]}" -N "g4_core_${label}" \
    -o "results/logs/core_${label}.out" \
    -e "results/logs/core_${label}.err" -v "$vars" jobs/sge/gemma4_core.sh)
  neutral=$(qsub -terse -hold_jid "$core" -N "g4_pca_${label}" \
    -o "results/logs/pca_${label}.out" -e "results/logs/pca_${label}.err" \
    -v "$vars" jobs/sge/gemma4_neutral.sh)
  dense=$(qsub -terse -hold_jid "$neutral" -N "g4_dense_${label}" \
    -o "results/logs/dense_${label}.out" -e "results/logs/dense_${label}.err" \
    -v "$vars" jobs/sge/gemma4_dense.sh)
  hiring=$(qsub -terse -hold_jid "$dense" -N "g4_hire_${label}" \
    -o "results/logs/hiring_${label}.out" -e "results/logs/hiring_${label}.err" \
    -v "$vars" jobs/sge/gemma4_hiring.sh)
  echo "$label: core=$core neutral=$neutral dense=$dense hiring=$hiring"
  LAST_HIRING_JOB="$hiring"
}

# Dense first, then MoE, matching the smoke and production gate order.
submit_model "google/gemma-4-31B-it" "gemma4_31b" "concept_vectors_gemma4_31b" 60 5376
if [[ "$MODE" == "--full" ]]; then
  submit_model "google/gemma-4-26B-A4B-it" "gemma4_26b_a4b" \
    "concept_vectors_gemma4_26b_a4b" 30 2816 "$LAST_HIRING_JOB"
else
  submit_model "google/gemma-4-26B-A4B-it" "gemma4_26b_a4b" \
    "concept_vectors_gemma4_26b_a4b" 30 2816 "$LAST_SMOKE_JOB"
fi
