#!/bin/bash
# SGE job: extract neutral-corpus activations for Qwen3-14B, then PCA-denoise
# its warmth/competence vectors.
#
# Mirrors jobs/sge/extract_neutral.sh (Gemma-3-12B baseline) and
# jobs/sge/gemma4_neutral.sh (parametrized Gemma-4 pattern), applied to the
# same model/vectors-subdir already used in jobs/sge/extract_qwen3_14b.sh.
#
# Closes a coverage gap found 2026-08-04: Qwen3-14B and Llama-3.1-8B-Instruct
# were the only two of the nine models with no X_neutral.npy anywhere (not on
# SCCKN, not in git); the original 2026-06-29 PCA-denoising rollout only
# covered Gemma-3-12B and Gemma-3-27B, and the later gemma4/qwen36 wave never
# looped back to these two. See paper/2026-08-04_1544_nine_model_cross_model_agreement.md
# and step_logs/STEP_LOG.md for the investigation.
#
# Model: Qwen/Qwen3-14B (~28 GB bf16) — needs a >=48 GB GPU; L40 fits. Weights
# already cached under HF_HOME on SCCKN; no download.
# Requires data/processed/concept_vectors_qwen3_14b/{warmth_vec,competence_vec,
# X_<condition>}.npy to already exist (they do — Phase 4/5 extraction ran
# 2026-06-19/20).
#
# Queued on scc213/scc214 (both busy but not admin-disabled at submission
# time; scc192 and spiderman showed SGE state "d" / disabled and were
# excluded). SGE will queue the job and start it once a slot frees up.

#$ -N wc_extract_neutral_qwen3_14b
#$ -q gpu@scc213,gpu@scc214
#$ -l h_rt=01:00:00
#$ -l h_vmem=64G
#$ -pe smp 2
#$ -l gpu=1
#$ -o results/logs/extract_neutral_qwen3_14b.out
#$ -e results/logs/extract_neutral_qwen3_14b.err
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

mkdir -p results/logs data/processed/concept_vectors_qwen3_14b

echo "[job] Step 1: extract neutral-corpus activations — Qwen3-14B (GPU)"
python src/extract_neutral.py \
    --config config/config.yaml \
    --model Qwen/Qwen3-14B \
    --vectors-subdir concept_vectors_qwen3_14b

echo "[job] Step 2: PCA denoise warmth/competence vectors (CPU)"
python src/denoise_vectors.py \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_qwen3_14b

echo "[job] Step 3: Sync outputs to git (additive — never force-pushes)"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
  || echo "[job] WARNING: push failed (credentials?) — run jobs/sync_outputs.sh from the login node"

echo "[job] Done."
