#!/bin/bash
# Submit gated Gemma 4 chains. By default submits smoke jobs only; after reviewing
# their JSON outputs, rerun with --full to submit each model's dependent pipeline.
set -euo pipefail

MODE="${1:---smoke}"
if [[ "$MODE" != "--smoke" && "$MODE" != "--smoke-31b-12b" && "$MODE" != "--full" ]]; then
  echo "usage: bash jobs/sge/submit_gemma4.sh [--smoke|--smoke-31b-12b|--full]" >&2
  exit 2
fi

preflight() {
  if pgrep -u "$USER" -f 'pip(3)? install|conda (create|install|update)' >/dev/null; then
    echo "Refusing submission: a pip/conda mutation is still running for $USER." >&2
    exit 1
  fi
  conda run -n wc-tl-g4 python -m pip check
  conda run -n wc-tl-g4 python -c \
    'from importlib.metadata import version; import torch, torchvision, transformers; tl=version("transformer-lens"); print(f"torch={torch.__version__} torchvision={torchvision.__version__} transformers={transformers.__version__} transformer-lens={tl}")'
  git config user.name >/dev/null
  git config user.email >/dev/null
}

submit_smoke() {
  local model="$1" label="$2" vectors="$3" layers="$4" d_model="$5"
  local queue="$6" h_vmem="$7"
  local vars="MODEL_NAME=$model,LABEL=$label,VECTORS_SUBDIR=$vectors,EXPECTED_LAYERS=$layers,EXPECTED_D_MODEL=$d_model"
  local job
  job=$(qsub -terse -q "$queue" -l "h_vmem=$h_vmem" \
    -N "g4_smoke_${label}" -o "results/logs/smoke_${label}.out" \
    -e "results/logs/smoke_${label}.err" -v "$vars" jobs/sge/gemma4_smoke.sh)
  echo "$label: smoke=$job"
}

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

if [[ "$MODE" == "--smoke-31b-12b" ]]; then
  preflight
  submit_smoke "google/gemma-4-31B-it" "gemma4_31b_retry1" \
    "concept_vectors_gemma4_31b" 60 5376 "gpu@scc214" "96G"
  submit_smoke "google/gemma-4-12B-it" "gemma4_12b" \
    "concept_vectors_gemma4_12b" 48 3840 \
    "gpu@scc192,gpu@scc213,gpu@scc214" "32G"
  exit 0
fi

# Dense first, then MoE, matching the smoke and production gate order.
submit_model "google/gemma-4-31B-it" "gemma4_31b" "concept_vectors_gemma4_31b" 60 5376
if [[ "$MODE" == "--full" ]]; then
  submit_model "google/gemma-4-26B-A4B-it" "gemma4_26b_a4b" \
    "concept_vectors_gemma4_26b_a4b" 30 2816 "$LAST_HIRING_JOB"
else
  submit_model "google/gemma-4-26B-A4B-it" "gemma4_26b_a4b" \
    "concept_vectors_gemma4_26b_a4b" 30 2816 "$LAST_SMOKE_JOB"
fi
