#!/bin/bash
# Gemma 4 raw dense steering in the existing local regime.
# ADJUST: SCCKN resources if smoke measurements require it.
#$ -q gpu@scc214
#$ -l h_rt=03:00:00
#$ -l h_vmem=96G
#$ -pe smp 2
#$ -l gpu=1
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${LABEL:?LABEL is required}"
: "${VECTORS_SUBDIR:?VECTORS_SUBDIR is required}"
module load conda  # ADJUST
conda activate wc-tl-g4
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis
cd /work/emrecan.ulu/normalcy-axis
git pull --ff-only

python -m src.dense_steering \
  --vectors-subdir "$VECTORS_SUBDIR" --label "$LABEL" \
  --strengths=-0.1,-0.05,0,0.05,0.1 --prompt-format native-chat
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis
