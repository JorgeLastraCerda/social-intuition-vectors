#!/bin/bash
# SGE job: Phase 7 hiring audit — Gemma-3-12B-it.
#
# REGRESSION GATE — run this first.
# On completion, check against legacy notebook outputs:
#   hiring_steering_raw_gemma3_12b.csv : warmth Δmargin at +0.25/+0.50 ≈ +7.125 / +8.404
#   hiring_audit_gemma3_12b.csv        : probe-vs-human rho ≈ 0.355 (warmth), 0.230 (competence)
# If within GPU float noise, submit the other three jobs.
#
# Model: google/gemma-3-12b-it (~24 GB bf16) — fits L40 48 GB nodes.
# Runs in sequence: hiring_steering → hiring_audit → hiring_disparity (CPU).
#
# Outputs:
#   results/tables/hiring_steering_raw_gemma3_12b.csv
#   results/tables/hiring_audit_gemma3_12b.csv
#   results/tables/hiring_disparity_gemma3_12b.csv
#   results/logs/hiring_steering_gemma3_12b.json
#   results/logs/hiring_probe_vs_human_gemma3_12b.json
#   results/logs/hiring_mediation_gemma3_12b.json
#
# ADJUST: queue/GPU flag syntax — confirm with Stefan if scc192/scc213 available.
# ADJUST: h_vmem semantics (per-slot vs total) on SCCKN.

#$ -N wc_hire_g12b
#$ -q gpu@scc192,gpu@scc213   # ADJUST
#$ -l h_rt=03:00:00
#$ -l h_vmem=64G              # ADJUST
#$ -pe smp 2                  # ADJUST
#$ -l gpu=1
#$ -o results/logs/hiring_gemma3_12b.out
#$ -e results/logs/hiring_gemma3_12b.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail

module load conda              # ADJUST: module name if different
conda activate wc-tl

export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis

cd /work/emrecan.ulu/normalcy-axis
git pull

echo "[job] Phase 7 hiring steering — Gemma-3-12B-it (regression gate)"
python -m src.hiring_steering \
    --config config/config.yaml \
    --vectors-subdir concept_vectors \
    --label gemma3_12b \
    --n-names 60

echo "[job] Phase 7 hiring audit — Gemma-3-12B-it"
python -m src.hiring_audit \
    --config config/config.yaml \
    --vectors-subdir concept_vectors \
    --label gemma3_12b

echo "[job] Phase 7 hiring disparity + mediation — Gemma-3-12B-it (CPU)"
python -m src.hiring_disparity \
    --config config/config.yaml \
    --label gemma3_12b

echo "[job] Syncing outputs to git"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
    || echo "[job] WARNING: push failed — run jobs/sync_outputs.sh from login node"

echo "[job] Gemma-3-12B hiring audit complete."
