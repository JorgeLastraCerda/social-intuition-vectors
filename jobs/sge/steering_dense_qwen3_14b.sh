#!/bin/bash
# SGE job: Dense (SAE-free) concept steering — Qwen3-14B.
#
# Runs src/dense_steering.py with the raw_dense direction only (no SAE).
# Submit AFTER verifying the Gemma-12B regression gate job passes.
#
# Model: Qwen/Qwen3-14B (~28 GB bf16) — requires a >=32 GB GPU node.
# Note: config/config.yaml has model name "Qwen/Qwen3-14B" (base variant).
# The script reads the model name from meta.json, so it stays consistent
# with the extraction run.
# Outputs: results/tables/steering_dense_raw_qwen3_14b.csv
#          results/tables/steering_dense_qwen3_14b.csv
#          results/logs/steering_dense_qwen3_14b.json
#
# ADJUST: queue/GPU — 28 GB bf16 needs >=32 GB VRAM (L40 48 GB on scc213 is fine).
# ADJUST: h_vmem semantics on SCCKN.

#$ -N wc_dense_qw14b
#$ -q gpu@scc192,gpu@scc213   # ADJUST: >=32 GB node required
#$ -l h_rt=02:00:00
#$ -l h_vmem=64G              # ADJUST
#$ -pe smp 2                  # ADJUST
#$ -l gpu=1
#$ -o results/logs/steering_dense_qwen3_14b.out
#$ -e results/logs/steering_dense_qwen3_14b.err
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

echo "[job] Dense steering — Qwen3-14B"
python -m src.dense_steering \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_qwen3_14b \
    --label qwen3_14b \
    --strengths=-0.1,-0.05,0,0.05,0.1

echo "[job] Syncing outputs to git"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
    || echo "[job] WARNING: push failed — run jobs/sync_outputs.sh from login node"

echo "[job] Qwen3-14B dense steering complete."
