#!/bin/bash
# SGE job: Dense (SAE-free) concept steering — Gemma-3-27B-it.
#
# Runs src/dense_steering.py with the raw_dense direction only (no SAE).
# Mirrors the local-regime Gemma Scope run but without SAE dependency.
#
# Model: google/gemma-3-27b-it (~54 GB bf16) — requires scc214 (RTX 6000 96 GB).
# Outputs: results/tables/steering_dense_raw_gemma3_27b.csv
#          results/tables/steering_dense_gemma3_27b.csv
#          results/logs/steering_dense_gemma3_27b.json
#
# ADJUST: confirm scc214 availability and h_vmem semantics with Stefan.

#$ -N wc_dense_g27b
#$ -q gpu@scc214              # ADJUST: 96 GB node required for 27B
#$ -l h_rt=02:00:00
#$ -l h_vmem=96G              # ADJUST
#$ -pe smp 2                  # ADJUST
#$ -l gpu=1
#$ -o results/logs/steering_dense_gemma3_27b.out
#$ -e results/logs/steering_dense_gemma3_27b.err
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

echo "[job] Dense steering — Gemma-3-27B-it"
python -m src.dense_steering \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_gemma3_27b \
    --label gemma3_27b \
    --strengths=-0.1,-0.05,0,0.05,0.1

echo "[job] Syncing outputs to git"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
    || echo "[job] WARNING: push failed — run jobs/sync_outputs.sh from login node"

echo "[job] Gemma-3-27B dense steering complete."
