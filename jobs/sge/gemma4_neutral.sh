#!/bin/bash
# Gemma 4 neutral activation extraction and PCA denoising.
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
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${VECTORS_SUBDIR:?VECTORS_SUBDIR is required}"
module load conda  # ADJUST
conda activate wc-tl-g4
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis
cd /work/emrecan.ulu/normalcy-axis
git pull --ff-only

python src/extract_neutral.py --model "$MODEL_NAME" --vectors-subdir "$VECTORS_SUBDIR"
python src/denoise_vectors.py --vectors-subdir "$VECTORS_SUBDIR"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis
