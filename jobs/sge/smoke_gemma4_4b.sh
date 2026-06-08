#!/bin/bash
# SGE job: Gemma 4 E4B-IT smoke test (nnsight)
# Submit from repo root: qsub jobs/sge/smoke_gemma4_4b.sh
#
# Uses google/gemma-4-E4B-it (bf16 full precision, ~8 GB VRAM) via nnsight.
# This is the exploratory Gemma 4 counterpart to smoke_gemma3_4b.sh.
# Gemma 4 is Apache 2.0; no HF license gate.
#
# NOTE on Gemma 4 PLE (Per-Layer Embeddings):
#   Gemma 4 adds position-dependent learned offsets at each layer.
#   This does not prevent extraction or steering, but the "clean residual-stream
#   superposition" assumption is slightly weaker than for Gemma 3.
#   Treat this result as exploratory; the committed model is Gemma 3.
#
# Why this job schedules fastest:
#   - ~8 GB VRAM -> fits on ANY GPU node; no pin needed.
#   - Minimal h_vmem, smp, and h_rt -> small footprint -> backfill scheduling.
#
# ADJUST: confirm exact GPU queue name and resource flag with Stefan.
# ADJUST: h_vmem semantics (per-slot vs total) on SCCKN; 16G is a starting point.
# ADJUST: module name if different from "conda".

#$ -N smk_g4_4b
#$ -q gpu                                 # ADJUST: any GPU queue (no node pin needed)
#$ -l h_rt=00:30:00
#$ -l h_vmem=16G                          # ADJUST: RAM per slot
#$ -pe smp 2                              # ADJUST: CPU cores
#$ -l gpu=1
#$ -o results/logs/smoke_gemma4_4b.out
#$ -e results/logs/smoke_gemma4_4b.err
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
    --model google/gemma-4-E4B-it \
    --seed 20260527
