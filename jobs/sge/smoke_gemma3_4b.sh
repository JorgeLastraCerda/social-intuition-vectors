#!/bin/bash
# SGE job: Gemma 3 4B-IT smoke test (TransformerLens + GemmaScope 2 SAE)
# Submit from repo root: qsub jobs/sge/smoke_gemma3_4b.sh
#
# Uses google/gemma-3-4b-it (bf16 full precision, ~8 GB VRAM) with the same
# probe + SAE pipeline as the 12B jobs.  Method is fully intact:
#   - TransformerLens supports Gemma 3 4B natively.
#   - GemmaScope 2 provides residual SAEs for 4B: google/gemma-scope-2-4b-it-res
#     (SAEs at 25/50/65/85% depth, same checkpoint granularity as 12B).
#   - probe_layer_frac=0.66 -> round((34-1)*0.66) = round(21.78) = layer 22 (65% depth).
#
# Why this job schedules fastest:
#   - ~8 GB VRAM -> fits on ANY GPU node; no big-VRAM pin needed.
#   - Minimal h_vmem, smp, and h_rt -> small footprint -> backfill scheduling.
#
# ADJUST: confirm exact GPU queue name and resource flag with Stefan.
# ADJUST: h_vmem semantics (per-slot vs total) on SCCKN; 16G is a starting point.
# ADJUST: module name if different from "conda".
# ADJUST (IMPORTANT): verify the SAE sae-id layer index before relying on it.
#   Gemma 3 4B has 34 layers; 65% depth checkpoint is approximately layer 21-22.
#   Confirm exact layer and l0 tag by running:
#     python -c "from sae_lens import SAE; print(list(SAE.from_pretrained('gemma-scope-2-4b-it-res')))"
#   Replace layer_22_width_16k_l0_medium below with the confirmed id.

#$ -N smk_g3_4b
#$ -q gpu                                 # ADJUST: any GPU queue (no node pin needed)
#$ -l h_rt=00:30:00
#$ -l h_vmem=16G                          # ADJUST: RAM per slot
#$ -pe smp 2                              # ADJUST: CPU cores
#$ -l gpu=1
#$ -o results/logs/smoke_gemma3_4b.out
#$ -e results/logs/smoke_gemma3_4b.err
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
# --out-dir isolates this run's vectors from concurrent 12B Gemma 3 jobs
python smoke_tests/gemma3_transformerlens/smoke_test_probe.py \
    --model google/gemma-3-4b-it \
    --start-token 1 \
    --seed 20260527 \
    --out-dir smoke_tests/gemma3_transformerlens/results/g3_4b

# Step 2: SAE warmth-vs-tone decomposition
# ADJUST: confirm layer index + l0 tag from the gemma-scope-2-4b-it-res catalog.
#   Gemma 3 4B: 34 layers, probe_layer_frac=0.66 -> round(33*0.66)=22.
#   GemmaScope 2 4B has checkpoints at 25/50/65/85% depth.
#   65% of 34 layers ≈ layer 22 -> sae-id should be layer_22_width_16k_l0_medium
#   (verify; the l0 suffix may differ from the 12B release).
python smoke_tests/gemma3_transformerlens/sae_decompose.py \
    --model google/gemma-3-4b-it \
    --sae-release gemma-scope-2-4b-it-res \
    --sae-id layer_22_width_16k_l0_medium \
    --seed 20260527 \
    --out-dir smoke_tests/gemma3_transformerlens/results/g3_4b
