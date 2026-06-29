#!/bin/bash
# SGE job: Dense (SAE-free) concept steering — Llama-3.1-8B-Instruct.
#
# Runs src/dense_steering.py with the raw_dense direction only (no SAE).
# Submit AFTER verifying the Gemma-12B regression gate job passes.
#
# Model: meta-llama/Llama-3.1-8B-Instruct (~16 GB bf16) — fits any >=16 GB GPU.
# Outputs: results/tables/steering_dense_raw_llama31_8b.csv
#          results/tables/steering_dense_llama31_8b.csv
#          results/logs/steering_dense_llama31_8b.json
#
# ADJUST: queue/GPU flag syntax — confirm with Stefan.
# ADJUST: h_vmem semantics on SCCKN (64G confirmed working for 8B extraction).

#$ -N wc_dense_ll8b
#$ -q gpu@scc192,gpu@scc213   # ADJUST: any >=16 GB GPU node
#$ -l h_rt=02:00:00
#$ -l h_vmem=64G              # ADJUST
#$ -pe smp 2                  # ADJUST
#$ -l gpu=1
#$ -o results/logs/steering_dense_llama31_8b.out
#$ -e results/logs/steering_dense_llama31_8b.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail

module load conda              # ADJUST
conda activate wc-tl

export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis

cd /work/emrecan.ulu/normalcy-axis
git pull

echo "[job] Dense steering — Llama-3.1-8B-Instruct"
python -m src.dense_steering \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_llama31_8b \
    --label llama31_8b \
    --strengths=-0.1,-0.05,0,0.05,0.1

echo "[job] Syncing outputs to git"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
    || echo "[job] WARNING: push failed — run jobs/sync_outputs.sh from login node"

echo "[job] Llama-3.1-8B dense steering complete."
