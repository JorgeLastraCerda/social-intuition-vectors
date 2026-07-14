#!/bin/bash
# Gemma 4 broad/local/denoised hiring, audit, mediation and R4.
# ADJUST: SCCKN resources if smoke measurements require it.
#$ -q gpu@scc214
#$ -l h_rt=12:00:00
#$ -l h_vmem=96G
#$ -pe smp 2
#$ -l gpu=1
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${LABEL:?LABEL is required}"
: "${VECTORS_SUBDIR:?VECTORS_SUBDIR is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${EXPECTED_LAYERS:?EXPECTED_LAYERS is required}"
: "${EXPECTED_D_MODEL:?EXPECTED_D_MODEL is required}"
module load conda  # ADJUST
conda activate wc-tl-g4
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis
cd /work/emrecan.ulu/normalcy-axis
git pull --ff-only

python -m src.hiring_steering \
  --vectors-subdir "$VECTORS_SUBDIR" --label "${LABEL}_broad" --n-names 60 \
  --strengths=-0.5,-0.25,0,0.25,0.5 --vector-kind raw --prompt-format native-chat
python -m src.hiring_steering \
  --vectors-subdir "$VECTORS_SUBDIR" --label "${LABEL}_local" --n-names 60 \
  --strengths=-0.1,-0.05,0,0.05,0.1 --vector-kind raw --prompt-format native-chat
python -m src.hiring_steering \
  --vectors-subdir "$VECTORS_SUBDIR" --label "${LABEL}_denoised_local" --n-names 60 \
  --strengths=-0.1,-0.05,0,0.05,0.1 --vector-kind denoised --prompt-format native-chat
python -m src.hiring_audit \
  --vectors-subdir "$VECTORS_SUBDIR" --label "$LABEL" --prompt-format native-chat
python -m src.hiring_disparity --label "$LABEL"
python -m src.hiring_r4 --label "$LABEL"
python -m src.validate_gemma4_run \
  --model "$MODEL_NAME" --label "$LABEL" --vectors-subdir "$VECTORS_SUBDIR" \
  --expected-layers "$EXPECTED_LAYERS" --expected-d-model "$EXPECTED_D_MODEL"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis
