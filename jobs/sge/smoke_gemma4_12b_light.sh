#!/bin/bash
# SGE job: Gemma 4 12B-IT smoke test — LIGHT variant (nnsight)
# Submit from repo root: qsub jobs/sge/smoke_gemma4_12b_light.sh
#
# This is a resource-trimmed duplicate of smoke_gemma4.sh designed to schedule
# faster in the SCCKN queue.  The test logic and assertions are identical.
#
# Why it schedules sooner than smoke_gemma4.sh:
#   1. Queue spans three big-VRAM nodes instead of one pin.
#   2. h_vmem, smp slots, and h_rt are reduced to the minimum needed for 12B.
#
# ADJUST: verify node list has >= 48 GB VRAM.
# ADJUST: confirm the exact GPU resource flag syntax with Stefan.
# ADJUST: h_vmem semantics (per-slot vs total) on SCCKN; 32G is a starting point.
# ADJUST: module name if different from "conda".
# Model VRAM estimates (bf16):
#   gemma-4-12B-it  ~24 GB  -> needs a >=48 GB GPU (L40 or RTX 6000 96 GB)
# Gemma 4 12B-IT (google/gemma-4-12B-it) is Apache 2.0; no HF license gate.

#$ -N smk_g4_12l
#$ -q gpu@scc192,gpu@scc213,gpu@scc214   # ADJUST: three big-VRAM nodes
#$ -l h_rt=01:00:00
#$ -l h_vmem=32G                          # ADJUST: RAM per slot
#$ -pe smp 2                              # ADJUST: CPU cores
#$ -l gpu=1
#$ -o results/logs/smoke_gemma4_12b_light.out
#$ -e results/logs/smoke_gemma4_12b_light.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

module load conda                         # ADJUST: module name if different
conda activate wc-nn                      # nnsight environment (NOT wc-tl)

export HF_HOME=/work/emrecan.ulu/hf_cache

cd /work/emrecan.ulu/normalcy-axis
git pull

mkdir -p results/logs

python smoke_tests/gemma4_nnsight/smoke_test_probe.py \
    --model google/gemma-4-12B-it \
    --seed 20260527
