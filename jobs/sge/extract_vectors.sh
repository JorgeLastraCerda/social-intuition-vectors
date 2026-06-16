#!/bin/bash
# SGE job: Extract warmth/competence vectors then validate probes
# 200 concept stories (data/stimuli/concept_stories.jsonl) -> Gemma-3-12B-it
# Submit from cluster repo root: qsub jobs/sge/extract_vectors.sh
#
# Queue optimisation — 3-node fan-out:
#   Pinning to three big-VRAM nodes instead of one means SGE schedules the job
#   on whichever becomes free first, cutting median queue wait significantly.
#   scc192 (L40 48 GB, 4 GPUs), scc213 (L40 48 GB, 8 GPUs), scc214 (RTX 6000, 96 GB).
#   Gemma-3-12B bf16 needs ~24 GB VRAM — all three nodes can run it.
#
# ADJUST: verify queue names / GPU resource flag syntax with Stefan if needed.
# ADJUST: h_vmem semantics — 64G per-slot, 2 slots = 128G total is a safe estimate.
# Model VRAM estimate (bf16): gemma-3-12b-it ~24 GB -> needs >= 48 GB GPU.

#$ -N wc_extract_concept
#$ -q gpu@scc192,gpu@scc213,gpu@scc214
#$ -l h_rt=01:00:00
#$ -l h_vmem=64G
#$ -pe smp 2
#$ -l gpu=1
#$ -o results/logs/wc_extract_concept.out
#$ -e results/logs/wc_extract_concept.err
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

mkdir -p results/logs results/tables results/figures data/processed/concept_vectors

echo "[job] Step 1: Extract residual-stream activations and build vectors"
python src/extract_vectors.py --config config/config.yaml

echo "[job] Step 2: Validate probes (CPU-only — vectors already on disk)"
python src/validate_probes.py --config config/config.yaml

echo "[job] Done."
