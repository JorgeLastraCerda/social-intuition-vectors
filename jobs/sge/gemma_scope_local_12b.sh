#!/bin/bash
# SGE job: local-regime Gemma Scope steering for Gemma-3-12B-it.

#$ -N wc_scope12_loc
#$ -q gpu@scc192,gpu@scc213
#$ -l h_rt=04:00:00
#$ -l h_vmem=64G
#$ -pe smp 2
#$ -l gpu=1
#$ -o results/logs/gemma_scope_local_job_12b.out
#$ -e results/logs/gemma_scope_local_job_12b.err
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
    --vectors-subdir concept_vectors \
    --scope-subdir gemma_scope_gemma3_12b \
    --label gemma3_12b_local \
    --sae-release gemma-scope-2-12b-it-res \
    --sae-id layer_31_width_65k_l0_medium \
    --strengths=-0.1,-0.05,0,0.05,0.1 \
    --skip-ablation

echo "[job] Gemma-3-12B local-regime steering complete."
