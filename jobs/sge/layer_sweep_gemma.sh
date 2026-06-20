#!/bin/bash
# SGE job: Layer sweep for Gemma-3-12B-it (all residual layers, 200 concept stories).
# Produces results/tables/layer_sweep_google_gemma-3-12b-it.csv (or use --label).
#
# Model: google/gemma-3-12b-it (~24 GB bf16) — fits scc192/213/214.
# h_rt: single forward pass per story x 200 stories x ~48 layers — ~1 h estimate.
#
# ADJUST: verify queue names / GPU flag syntax with Stefan if needed.

#$ -N wc_sweep_gemma
#$ -q gpu@scc192,gpu@scc213,gpu@scc214
#$ -l h_rt=02:00:00
#$ -l h_vmem=64G
#$ -pe smp 2
#$ -l gpu=1
#$ -o results/logs/layer_sweep_gemma.out
#$ -e results/logs/layer_sweep_gemma.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail

module load conda                         # ADJUST: module name if different
conda activate wc-tl

export HF_HOME=/work/emrecan.ulu/hf_cache

cd /work/emrecan.ulu/normalcy-axis
git pull

export PYTHONPATH=/work/emrecan.ulu/normalcy-axis

mkdir -p results/logs results/tables

echo "[job] Layer sweep — Gemma-3-12B-it (all layers, topic-holdout CV)"
python src/layer_sweep.py \
    --config config/config.yaml \
    --label gemma3_12b

echo "[job] Sync outputs to git (additive)"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
  || echo "[job] WARNING: push failed (credentials?) — run jobs/sync_outputs.sh from the login node"

echo "[job] Done."
