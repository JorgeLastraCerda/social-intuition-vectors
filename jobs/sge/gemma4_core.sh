#!/bin/bash
# Gemma 4 extraction, validation and layer sweep. Submit after smoke passes.
# ADJUST: SCCKN resources if smoke measurements require it.
#$ -q gpu@scc214
#$ -l h_rt=08:00:00
#$ -l h_vmem=96G
#$ -pe smp 2
#$ -l gpu=1
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${LABEL:?LABEL is required}"
: "${VECTORS_SUBDIR:?VECTORS_SUBDIR is required}"
module load conda  # ADJUST
conda activate wc-tl-g4
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis
cd /work/emrecan.ulu/normalcy-axis
git pull --ff-only

python src/extract_vectors.py --model "$MODEL_NAME" --out-subdir "$VECTORS_SUBDIR"
python src/validate_probes.py --vectors-subdir "$VECTORS_SUBDIR" --label "$LABEL"
python src/layer_sweep.py --model "$MODEL_NAME" --label "$LABEL"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis
