#!/bin/bash
# SGE job: Layer sweep for Llama-3.1-8B-Instruct (all residual layers, 200 concept stories).
# Produces results/tables/layer_sweep_llama31_8b.csv
#
# Model: meta-llama/Llama-3.1-8B-Instruct (~16 GB bf16) — fits any >=16 GB GPU node.
# h_rt: 32 layers x 200 stories single-pass — ~1 h estimate.
#
# ADJUST: verify queue names / GPU flag syntax with Stefan if needed.

#$ -N wc_sweep_ll31_8b
#$ -q gpu@scc192,gpu@scc213,gpu@scc214
#$ -l h_rt=02:00:00
#$ -l h_vmem=64G                          # 32G OOM'd on extract; 64G confirmed working
#$ -pe smp 2
#$ -l gpu=1
#$ -o results/logs/layer_sweep_llama31_8b.out
#$ -e results/logs/layer_sweep_llama31_8b.err
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

echo "[job] Layer sweep — Llama-3.1-8B-Instruct (all layers, topic-holdout CV)"
python src/layer_sweep.py \
    --config config/config.yaml \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --label llama31_8b

echo "[job] Sync outputs to git (additive)"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
  || echo "[job] WARNING: push failed (credentials?) — run jobs/sync_outputs.sh from the login node"

echo "[job] Done."
