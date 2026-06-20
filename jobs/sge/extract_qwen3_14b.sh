#!/bin/bash
# SGE job: Extract warmth/competence vectors for Qwen3-14B (200 concept stories)
# and validate probes.  Mirrors jobs/sge/extract_vectors.sh (Gemma baseline).
#
# Model: Qwen/Qwen3-14B  (~28 GB bf16) — needs >=48 GB GPU (L40 or RTX 6000).
# Outputs go to data/processed/concept_vectors_qwen3_14b/ to avoid clobbering
# the Gemma vectors in data/processed/concept_vectors/.
# start_token=50 (from config) — parity with the Gemma 200-story run.
#
# ADJUST: verify queue names / GPU resource flag syntax with Stefan.
# ADJUST: h_vmem semantics (per-slot vs total) on SCCKN; 64G per slot is the
#         estimate that worked for the smoke test.
# Model VRAM estimate (bf16): Qwen3-14B ~28 GB -> needs >=48 GB GPU.

#$ -N wc_extract_qw3_14b
#$ -q gpu@scc192,gpu@scc213,gpu@scc214   # ADJUST: three big-VRAM nodes
#$ -l h_rt=01:30:00
#$ -l h_vmem=64G                          # ADJUST: RAM per slot
#$ -pe smp 2                              # ADJUST: CPU cores
#$ -l gpu=1
#$ -o results/logs/extract_qwen3_14b.out
#$ -e results/logs/extract_qwen3_14b.err
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

mkdir -p results/logs results/tables \
         data/processed/concept_vectors_qwen3_14b

echo "[job] Step 1: Extract residual-stream activations — Qwen3-14B"
python src/extract_vectors.py \
    --config config/config.yaml \
    --model Qwen/Qwen3-14B \
    --out-subdir concept_vectors_qwen3_14b

echo "[job] Step 2: Validate probes (CPU-only — vectors already on disk)"
python src/validate_probes.py \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_qwen3_14b \
    --label qwen3_14b

echo "[job] Step 3: Sync outputs to git (additive — never force-pushes)"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
  || echo "[job] WARNING: push failed (credentials?) — run jobs/sync_outputs.sh from the login node"

echo "[job] Done."
