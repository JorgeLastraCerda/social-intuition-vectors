#!/bin/bash
# SGE job: Gemma Scope 2 analysis + concept causality for Gemma-3-12B-it.
# Uses committed layer-31 activations; it does not rerun activation extraction.

#$ -N wc_scope_12b
#$ -q gpu@scc213
#$ -l h_rt=12:00:00
#$ -l h_vmem=64G
#$ -pe smp 2
#$ -l gpu=1
#$ -o results/logs/gemma_scope_job_12b.out
#$ -e results/logs/gemma_scope_job_12b.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail

module load conda
conda activate wc-tl

export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis

cd /work/emrecan.ulu/normalcy-axis
mkdir -p data/processed/gemma_scope_gemma3_12b results/logs results/tables

python -c "import sae_lens; print('[env] sae_lens', sae_lens.__version__)"

python -m src.gemma_scope_analysis \
    --config config/config.yaml \
    --vectors-subdir concept_vectors \
    --output-subdir gemma_scope_gemma3_12b \
    --label gemma3_12b \
    --sae-release gemma-scope-2-12b-it-res \
    --sae-ids layer_31_width_16k_l0_medium,layer_31_width_65k_l0_medium,layer_31_width_262k_l0_medium

python -m src.gemma_scope_causality \
    --config config/config.yaml \
    --vectors-subdir concept_vectors \
    --scope-subdir gemma_scope_gemma3_12b \
    --label gemma3_12b \
    --sae-release gemma-scope-2-12b-it-res \
    --sae-id layer_31_width_65k_l0_medium

echo "[job] Gemma-3-12B Gemma Scope analysis complete."
