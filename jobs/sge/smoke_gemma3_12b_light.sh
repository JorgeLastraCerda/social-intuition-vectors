#!/bin/bash
# SGE job: Gemma 3 12B-IT smoke test — LIGHT variant (TransformerLens + GemmaScope 2 SAE)
# Submit from repo root: qsub jobs/sge/smoke_gemma3_12b_light.sh
#
# This is a resource-trimmed duplicate of smoke_gemma3.sh designed to schedule
# faster in the SCCKN queue.  The test logic and assertions are identical.
#
# Why it schedules sooner than smoke_gemma3.sh:
#   1. The queue spans three big-VRAM nodes instead of being pinned to one
#      (scc192, scc213, scc214 all have >= 48 GB VRAM; 12B bf16 needs ~24 GB).
#   2. h_vmem, smp slots, and h_rt are reduced to the minimum needed for 12B.
#
# ADJUST: verify that scc192 and scc214 have >= 48 GB VRAM on your cluster;
#         add or remove nodes in the -q line as appropriate.
# ADJUST: confirm the exact GPU resource flag syntax with Stefan.
# ADJUST: h_vmem semantics (per-slot vs total) on SCCKN; 32G is a starting point.
# ADJUST: module name if different from "conda".
# Model VRAM estimates (bf16):
#   gemma-3-12b-it  ~24 GB  -> needs a >=48 GB GPU (L40 or RTX 6000 96 GB)

#$ -N smk_g3_12l
#$ -q gpu@scc192,gpu@scc213,gpu@scc214   # ADJUST: three big-VRAM nodes; remove any that lack >=48 GB
#$ -l h_rt=01:00:00
#$ -l h_vmem=64G                          # ADJUST: RAM per slot (12B bf16 ~24GB VRAM + CPU overhead)
#$ -pe smp 2                              # ADJUST: CPU cores
#$ -l gpu=1
#$ -o results/logs/smoke_gemma3_12b_light.out
#$ -e results/logs/smoke_gemma3_12b_light.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

module load conda                         # ADJUST: module name if different
conda activate wc-tl

export HF_HOME=/work/emrecan.ulu/hf_cache

cd /work/emrecan.ulu/normalcy-axis
git pull

mkdir -p results/logs

# Step 1: probe + extract warmth vector
# --out-dir isolates saved vectors from other concurrent Gemma 3 runs (4B vs 12B)
python smoke_tests/gemma3_transformerlens/smoke_test_probe.py \
    --model google/gemma-3-12b-it \
    --start-token 1 \
    --seed 20260527 \
    --out-dir smoke_tests/gemma3_transformerlens/results/g3_12b

# Step 2: SAE warmth-vs-tone decomposition (runs only if Step 1 passed)
# Gemma 3 12B has 48 layers. probe_layer_frac=0.66 -> round((48-1)*0.66) = 31.
# GemmaScope 2 12B has SAEs at 25/50/65/85% depth; 31/48 = 65% -> exact match.
# ADJUST: verify sae-id via SAE.from_pretrained("gemma-scope-2-12b-it-res") catalog if needed.
python smoke_tests/gemma3_transformerlens/sae_decompose.py \
    --model google/gemma-3-12b-it \
    --sae-release gemma-scope-2-12b-it-res \
    --sae-id layer_31_width_16k_l0_medium \
    --seed 20260527 \
    --out-dir smoke_tests/gemma3_transformerlens/results/g3_12b
