#!/bin/bash
# SGE job: Dense (SAE-free) concept steering — Gemma-3-12B-it.
#
# Runs src/dense_steering.py with the raw_dense direction only (no SAE).
# This is the REGRESSION-GATE job: raw_dense rows in the output CSV must
# match gemma_scope_causality_gemma3_12b_local.csv (warmth +0.1 → 3.88125,
# competence +0.1 → 2.00625). Run this first; verify before submitting
# Llama/Qwen/27B jobs.
#
# Model: google/gemma-3-12b-it (~24 GB bf16) — fits L40 48 GB nodes.
# Outputs: results/tables/steering_dense_raw_gemma3_12b.csv
#          results/tables/steering_dense_gemma3_12b.csv
#          results/logs/steering_dense_gemma3_12b.json
#
# ADJUST: queue/GPU flag syntax — confirm with Stefan if scc192/scc213 available.
# ADJUST: h_vmem semantics (per-slot vs total) on SCCKN.

#$ -N wc_dense_g12b
#$ -q gpu@scc192,gpu@scc213   # ADJUST
#$ -l h_rt=02:00:00
#$ -l h_vmem=64G              # ADJUST: 64G confirmed working for 12B smoke
#$ -pe smp 2                  # ADJUST
#$ -l gpu=1
#$ -o results/logs/steering_dense_gemma3_12b.out
#$ -e results/logs/steering_dense_gemma3_12b.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail

module load conda              # ADJUST: module name if different
conda activate wc-tl

export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis

cd /work/emrecan.ulu/normalcy-axis
git pull

echo "[job] Dense steering — Gemma-3-12B-it (regression gate)"
python -m src.dense_steering \
    --config config/config.yaml \
    --vectors-subdir concept_vectors \
    --label gemma3_12b \
    --strengths=-0.1,-0.05,0,0.05,0.1

echo "[job] Syncing outputs to git"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
    || echo "[job] WARNING: push failed — run jobs/sync_outputs.sh from login node"

echo "[job] Gemma-3-12B dense steering complete."
