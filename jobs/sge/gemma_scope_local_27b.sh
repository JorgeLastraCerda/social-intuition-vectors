#!/bin/bash
# SGE job: local-regime Gemma Scope steering for Gemma-3-27B-it.

#$ -N wc_scope27_loc
#$ -q gpu@scc214
#$ -l h_rt=04:00:00
#$ -l h_vmem=96G
#$ -pe smp 2
#$ -l gpu=1
#$ -o results/logs/gemma_scope_local_job_27b.out
#$ -e results/logs/gemma_scope_local_job_27b.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail

module load conda
conda activate wc-tl
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis

cd /work/emrecan.ulu/normalcy-axis
python -m src.gemma_scope_causality \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_gemma3_27b \
    --scope-subdir gemma_scope_gemma3_27b \
    --label gemma3_27b_local \
    --sae-release gemma-scope-2-27b-it-res \
    --sae-id layer_40_width_65k_l0_medium \
    --strengths=-0.1,-0.05,0,0.05,0.1 \
    --skip-ablation

echo "[job] Gemma-3-27B local-regime steering complete."
