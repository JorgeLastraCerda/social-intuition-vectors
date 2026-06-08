#!/bin/bash
# SGE job: Gemma 3 12B-IT smoke test (TransformerLens + GemmaScope 2 SAE)
# Submit from repo root: qsub jobs/sge/smoke_gemma3.sh
#
# Node pinned to scc213 (L40, 48 GB VRAM) so 12B fits comfortably in bf16 (~24 GB).
# ADJUST: module names before submitting. Alternative high-VRAM nodes: scc192 (L40 48GB),
#         scc214 (RTX 6000 96GB). Change gpu@scc213 accordingly if scc213 is busy.
# Model VRAM estimates (bf16):
#   gemma-3-12b-it  ~24 GB  -> L40 (48 GB) has ample headroom
#   gemma-3-27b-it  ~54 GB  -> use scc214 (RTX 6000, 96 GB) instead

#$ -N smoke_gemma3
#$ -q gpu@scc213                   # L40 48 GB; ADJUST: gpu@scc192 or gpu@scc214 if busy
#$ -l h_rt=02:00:00
#$ -l h_vmem=64G                   # ADJUST: RAM per slot
#$ -pe smp 4                       # ADJUST: CPU cores
#$ -l gpu=1
#$ -o results/logs/smoke_gemma3.out
#$ -e results/logs/smoke_gemma3.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

module load conda                  # ADJUST: module name if different
conda activate wc-tl

export HF_HOME=/work/emrecan.ulu/hf_cache

cd /work/emrecan.ulu/normalcy-axis
git pull

mkdir -p results/logs

# Step 1: probe + extract warmth vector
python smoke_tests/gemma3_transformerlens/smoke_test_probe.py \
    --model google/gemma-3-12b-it \
    --start-token 1 \
    --seed 20260527

# Step 2: SAE warmth-vs-tone decomposition (runs only if Step 1 passed)
# Gemma 3 12B has 48 layers. probe_layer_frac=0.66 -> round((48-1)*0.66) = 31.
# GemmaScope 2 12B has SAEs at 25/50/65/85% depth; 31/48 = 65% -> exact match.
# sae-id: verify via SAE.from_pretrained("gemma-scope-2-12b-it-res") catalog if needed.
python smoke_tests/gemma3_transformerlens/sae_decompose.py \
    --model google/gemma-3-12b-it \
    --sae-release gemma-scope-2-12b-it-res \
    --sae-id layer_31_width_16k_l0_medium \
    --seed 20260527
