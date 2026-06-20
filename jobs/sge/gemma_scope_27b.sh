#!/bin/bash
# SGE job: Gemma Scope 2 analysis + concept causality for Gemma-3-27B-it.
# Uses committed layer-40 activations; it does not rerun activation extraction.

#$ -N wc_scope_27b
#$ -q gpu@scc214
#$ -l h_rt=12:00:00
#$ -l h_vmem=96G
#$ -pe smp 2
#$ -l gpu=1
#$ -o results/logs/gemma_scope_job_27b.out
#$ -e results/logs/gemma_scope_job_27b.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail

module load conda
conda activate wc-tl

export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis

cd /work/emrecan.ulu/normalcy-axis
mkdir -p data/processed/gemma_scope_gemma3_27b results/logs results/tables

python -c "import sae_lens; print('[env] sae_lens', sae_lens.__version__)"

python -m src.gemma_scope_analysis \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_gemma3_27b \
    --output-subdir gemma_scope_gemma3_27b \
    --label gemma3_27b \
    --sae-release gemma-scope-2-27b-it-res \
    --sae-ids layer_40_width_16k_l0_medium,layer_40_width_65k_l0_medium,layer_40_width_262k_l0_medium \
    --neuronpedia-base 'gemma-3-27b-it/40-gemmascope-2-res-{width}'

python -m src.gemma_scope_causality \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_gemma3_27b \
    --scope-subdir gemma_scope_gemma3_27b \
    --label gemma3_27b \
    --sae-release gemma-scope-2-27b-it-res \
    --sae-id layer_40_width_65k_l0_medium

echo "[job] Gemma-3-27B Gemma Scope analysis complete."
