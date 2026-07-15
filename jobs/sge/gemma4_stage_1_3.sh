#!/bin/bash
# Execute exactly one Gemma 4 replication stage. Submit through the stage runner.
# ADJUST: module and scratch paths if SCCKN changes its runtime environment.
#$ -q gpu@scc214
#$ -l h_rt=12:00:00
#$ -l h_vmem=96G
#$ -pe smp 2
#$ -l gpu=1
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${STAGE:?STAGE is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${LABEL:?LABEL is required}"
: "${VECTORS_SUBDIR:?VECTORS_SUBDIR is required}"
: "${EXPECTED_LAYERS:?EXPECTED_LAYERS is required}"
: "${EXPECTED_D_MODEL:?EXPECTED_D_MODEL is required}"
: "${SUCCESS_SENTINEL:?SUCCESS_SENTINEL is required}"

module load conda  # ADJUST
conda activate wc-tl-g4
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis
cd /work/emrecan.ulu/normalcy-axis
git pull --ff-only

if [[ -n "${PREDECESSOR_SENTINEL:-}" && ! -f "$PREDECESSOR_SENTINEL" ]]; then
  echo "Refusing stage $STAGE: predecessor sentinel is missing: $PREDECESSOR_SENTINEL" >&2
  exit 20
fi

common_args=(
  --stage "$STAGE"
  --model "$MODEL_NAME"
  --label "$LABEL"
  --vectors-subdir "$VECTORS_SUBDIR"
  --expected-layers "$EXPECTED_LAYERS"
  --expected-d-model "$EXPECTED_D_MODEL"
)
python -m src.validate_gemma4_stage "${common_args[@]}" --require-absent

case "$STAGE" in
  1)
    python src/extract_vectors.py --model "$MODEL_NAME" --out-subdir "$VECTORS_SUBDIR"
    ;;
  2)
    python src/validate_probes.py --vectors-subdir "$VECTORS_SUBDIR" --label "$LABEL"
    ;;
  3)
    python src/layer_sweep.py --model "$MODEL_NAME" --label "$LABEL"
    ;;
  *)
    echo "Unsupported STAGE=$STAGE" >&2
    exit 2
    ;;
esac

python -m src.validate_gemma4_stage "${common_args[@]}"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis
mkdir -p "$(dirname "$SUCCESS_SENTINEL")"
touch "$SUCCESS_SENTINEL"
echo "[success] $LABEL stage $STAGE -> $SUCCESS_SENTINEL"
