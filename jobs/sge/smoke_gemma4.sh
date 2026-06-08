#!/bin/bash
# SGE job: Gemma 4 12B-IT smoke test (nnsight)
# Submit from repo root: qsub jobs/sge/smoke_gemma4.sh
#
# Node pinned to scc213 (L40, 48 GB VRAM) — same node as Gemma 3 job for fair comparison.
# Gemma 4 12B-IT (google/gemma-4-12B-it) is Apache 2.0; no HF license gate.
# ADJUST: module names before submitting. Alternative nodes: scc192 (L40), scc214 (RTX 6000).
# Model VRAM estimates (bf16):
#   gemma-4-12B-it  ~24 GB  -> L40 (48 GB) has ample headroom
#   gemma-4-26b     ~52 GB  -> use scc214 (RTX 6000, 96 GB)

#$ -N smoke_gemma4
#$ -q gpu@scc213                   # L40 48 GB; ADJUST: gpu@scc192 or gpu@scc214 if busy
#$ -l h_rt=02:00:00
#$ -l h_vmem=64G                   # ADJUST: RAM per slot
#$ -pe smp 4                       # ADJUST: CPU cores
#$ -l gpu=1
#$ -o results/logs/smoke_gemma4.out
#$ -e results/logs/smoke_gemma4.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

module load conda                  # ADJUST: module name if different
conda activate wc-nn               # nnsight environment (NOT wc-tl)

export HF_HOME=/work/emrecan.ulu/hf_cache

cd /work/emrecan.ulu/normalcy-axis
git pull

mkdir -p results/logs

python smoke_tests/gemma4_nnsight/smoke_test_probe.py \
    --model google/gemma-4-12B-it \
    --seed 20260527
