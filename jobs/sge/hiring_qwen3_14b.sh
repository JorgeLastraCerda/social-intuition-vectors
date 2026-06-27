#!/bin/bash
# SGE job: Phase 7 hiring audit — Qwen3-14B.
#
# Submit only after the Gemma-3-12B regression gate passes.
# Model: Qwen/Qwen3-14B (~28 GB bf16) — fits L40 48 GB nodes.
#
# Outputs:
#   results/tables/hiring_steering_raw_qwen3_14b.csv
#   results/tables/hiring_audit_qwen3_14b.csv
#   results/tables/hiring_disparity_qwen3_14b.csv
#   results/logs/hiring_steering_qwen3_14b.json
#   results/logs/hiring_probe_vs_human_qwen3_14b.json
#   results/logs/hiring_mediation_qwen3_14b.json
#
# ADJUST: queue/GPU flag syntax — confirm with Stefan if scc192/scc213 available.
# ADJUST: h_vmem semantics (per-slot vs total) on SCCKN.

#$ -N wc_hire_qw14b
#$ -q gpu@scc192,gpu@scc213   # ADJUST
#$ -l h_rt=03:00:00
#$ -l h_vmem=64G              # ADJUST
#$ -pe smp 2                  # ADJUST
#$ -l gpu=1
#$ -o results/logs/hiring_qwen3_14b.out
#$ -e results/logs/hiring_qwen3_14b.err
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

echo "[job] Phase 7 hiring steering — Qwen3-14B"
python -m src.hiring_steering \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_qwen3_14b \
    --label qwen3_14b \
    --n-names 60

echo "[job] Phase 7 hiring audit — Qwen3-14B"
python -m src.hiring_audit \
    --config config/config.yaml \
    --vectors-subdir concept_vectors_qwen3_14b \
    --label qwen3_14b

echo "[job] Phase 7 hiring disparity + mediation — Qwen3-14B (CPU)"
python -m src.hiring_disparity \
    --config config/config.yaml \
    --label qwen3_14b

echo "[job] Syncing outputs to git"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis \
    || echo "[job] WARNING: push failed — run jobs/sync_outputs.sh from login node"

echo "[job] Qwen3-14B hiring audit complete."
