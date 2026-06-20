#!/bin/bash
# SGE job: Layer sweep for Qwen3-14B (all residual layers, 200 concept stories).
# Produces results/tables/layer_sweep_qwen3_14b.csv
#
# Model: Qwen/Qwen3-14B (~28 GB bf16) — needs >=48 GB GPU; scc192/213/214.
# h_rt: 40 layers x 200 stories single-pass — ~1.5 h estimate.
#
# ADJUST: verify queue names / GPU flag syntax with Stefan if needed.

#$ -N wc_sweep_qw3_14b
#$ -q gpu@scc192,gpu@scc213,gpu@scc214
#$ -l h_rt=02:30:00
#$ -l h_vmem=64G                          # 32G OOM'd on extract; 64G confirmed working
#$ -pe smp 2
#$ -l gpu=1
#$ -o results/logs/layer_sweep_qwen3_14b.out
#$ -e results/logs/layer_sweep_qwen3_14b.err
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

echo "[job] Layer sweep — Qwen3-14B (all layers, topic-holdout CV)"
python src/layer_sweep.py \
    --config config/config.yaml \
    --model Qwen/Qwen3-14B \
    --label qwen3_14b

echo "[job] Sync outputs to git (additive)"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
  || echo "[job] WARNING: push failed (credentials?) — run jobs/sync_outputs.sh from the login node"

echo "[job] Done."
