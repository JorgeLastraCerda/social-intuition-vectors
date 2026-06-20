#!/bin/bash
# SGE job: Gemma-3-27B-Instruct — full pipeline (extract → validate → sweep → sync)
# Runs the complete B3 pipeline in a single allocation to avoid queueing twice for
# the scc214 96 GB node.
#
# Model: google/gemma-3-27b-it  (~54 GB bf16) — ONLY scc214 (RTX 6000 Blackwell, 96 GB)
#   has enough VRAM.  L40 nodes (scc192, scc213) top out at 48 GB — they will OOM.
#
# Outputs:
#   data/processed/concept_vectors_gemma3_27b/   (warmth_vec, competence_vec, X_*.npy)
#   results/tables/probe_metrics_gemma3_27b.csv
#   results/logs/validate_probes_*_gemma3_27b.json
#   results/tables/layer_sweep_gemma3_27b.csv + .meta.json
#
# Walltime estimate:
#   Step 1 (extract, single layer, 200 stories): ~30 min
#   Step 3 (layer sweep, 62 layers x 200 stories single-pass): ~2.5–3 h
#   Total with overhead: 4 h ceiling is safe.
#
# ADJUST: verify queue/GPU flag syntax with Stefan.
# ADJUST: h_vmem semantics on SCCKN (per-slot vs. total); 96G is a conservative estimate.
#         from_pretrained_no_processing halves peak CPU RAM during load, but 27B bf16 is
#         still ~54 GB on GPU — keep h_vmem generous.

#$ -N wc_27b_full
#$ -q gpu@scc214                          # ADJUST: 96 GB node only — do NOT add scc192/scc213
#$ -l h_rt=04:00:00                       # ADJUST: raise if sweep takes longer than expected
#$ -l h_vmem=96G                          # ADJUST: per-slot RAM estimate for 27B bf16 + overhead
#$ -pe smp 2
#$ -l gpu=1
#$ -o results/logs/extract_gemma3_27b.out
#$ -e results/logs/extract_gemma3_27b.err
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

mkdir -p results/logs results/tables "results/figures/gemma3_27b" \
         data/processed/concept_vectors_gemma3_27b

echo "[job] ============================================================"
echo "[job] Gemma-3-27B full pipeline  (extract -> validate -> sweep)"
echo "[job] ============================================================"

echo "[job] Step 1: Extract residual-stream activations — Gemma-3-27B"
python src/extract_vectors.py \
    --config config/config.yaml \
    --model google/gemma-3-27b-it \
    --out-subdir concept_vectors_gemma3_27b

echo "[job] Step 2: Validate probes (CPU-only — vectors already on disk)"
python src/validate_probes.py \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_gemma3_27b \
    --label gemma3_27b

echo "[job] Step 3: Layer sweep — all residual layers, topic-holdout CV"
python src/layer_sweep.py \
    --config config/config.yaml \
    --model google/gemma-3-27b-it \
    --label gemma3_27b

echo "[job] Step 4: Sync outputs to git (additive — never force-pushes)"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
  || echo "[job] WARNING: push failed (credentials?) — run jobs/sync_outputs.sh from the login node"

echo "[job] Done."
