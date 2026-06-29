#!/bin/bash
# SGE job: Phase 7 hiring audit — Gemma-3-27B-it.
#
# Submit only after the Gemma-3-12B regression gate passes.
# Model: google/gemma-3-27b-it (~54 GB bf16) — requires 96G node (scc214).
#
# Outputs:
#   results/tables/hiring_steering_raw_gemma3_27b.csv
#   results/tables/hiring_audit_gemma3_27b.csv
#   results/tables/hiring_disparity_gemma3_27b.csv
#   results/logs/hiring_steering_gemma3_27b.json
#   results/logs/hiring_probe_vs_human_gemma3_27b.json
#   results/logs/hiring_mediation_gemma3_27b.json
#
# ADJUST: queue/GPU flag syntax — confirm scc214 (96G) availability.
# ADJUST: h_vmem semantics (per-slot vs total) on SCCKN.

#$ -N wc_hire_g27b
#$ -q gpu@scc214               # ADJUST — 96G node required for 27B
#$ -l h_rt=04:00:00
#$ -l h_vmem=96G              # ADJUST
#$ -pe smp 2                  # ADJUST
#$ -l gpu=1
#$ -o results/logs/hiring_gemma3_27b.out
#$ -e results/logs/hiring_gemma3_27b.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail

module load conda              # ADJUST
conda activate wc-tl

export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis

cd /work/emrecan.ulu/normalcy-axis
git pull

echo "[job] Phase 7 hiring steering — Gemma-3-27B-it"
python -m src.hiring_steering \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_gemma3_27b \
    --label gemma3_27b \
    --n-names 60

echo "[job] Phase 7 hiring audit — Gemma-3-27B-it"
python -m src.hiring_audit \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_gemma3_27b \
    --label gemma3_27b

echo "[job] Phase 7 hiring disparity + mediation — Gemma-3-27B-it (CPU)"
python -m src.hiring_disparity \
    --config config/config.yaml \
    --label gemma3_27b

echo "[job] Syncing outputs to git"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
    || echo "[job] WARNING: push failed — run jobs/sync_outputs.sh from login node"

echo "[job] Gemma-3-27B hiring audit complete."
