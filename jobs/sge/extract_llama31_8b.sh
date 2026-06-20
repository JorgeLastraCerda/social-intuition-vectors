#!/bin/bash
# SGE job: Extract warmth/competence vectors for Llama-3.1-8B-Instruct (200 concept stories)
# and validate probes.  Mirrors jobs/sge/extract_vectors.sh (Gemma baseline).
#
# Model: meta-llama/Llama-3.1-8B-Instruct  (~16 GB bf16) — fits any >=16 GB GPU.
# Outputs go to data/processed/concept_vectors_llama31_8b/ to avoid clobbering
# the Gemma vectors in data/processed/concept_vectors/.
# start_token=50 (from config) — parity with the Gemma 200-story run.
#
# ADJUST: verify queue names / GPU resource flag syntax with Stefan.
# ADJUST: h_vmem semantics (per-slot vs total) on SCCKN.
#         h_vmem=64G confirmed working for the smoke run (32G OOM'd).
# Model VRAM estimate (bf16): Llama-3.1-8B ~16 GB -> fits any GPU node with >=16 GB.

#$ -N wc_extract_ll31_8b
#$ -q gpu@scc192,gpu@scc213,gpu@scc214   # ADJUST: any big-VRAM node; or use "-q gpu"
#$ -l h_rt=01:30:00
#$ -l h_vmem=64G                          # ADJUST: 32G OOM'd; 64G confirmed working
#$ -pe smp 2                              # ADJUST: CPU cores
#$ -l gpu=1
#$ -o results/logs/extract_llama31_8b.out
#$ -e results/logs/extract_llama31_8b.err
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

mkdir -p results/logs results/tables "results/figures/llama31_8b" \
         data/processed/concept_vectors_llama31_8b

echo "[job] Step 1: Extract residual-stream activations — Llama-3.1-8B-Instruct"
python src/extract_vectors.py \
    --config config/config.yaml \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --out-subdir concept_vectors_llama31_8b

echo "[job] Step 2: Validate probes (CPU-only — vectors already on disk)"
python src/validate_probes.py \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_llama31_8b \
    --label llama31_8b

echo "[job] Step 3: Sync outputs to git (additive — never force-pushes)"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
  || echo "[job] WARNING: push failed (credentials?) — run jobs/sync_outputs.sh from the login node"

echo "[job] Done."
