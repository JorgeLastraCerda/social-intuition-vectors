#!/bin/bash
# SGE job: extract neutral-corpus activations, then PCA-denoise the vectors.
#
# PREREQUISITE (run once on the LOGIN node, which has internet):
#   pip install datasets
#   PYTHONPATH=. python scripts/build_neutral_corpus.py --config config/config.yaml
# That writes data/stimuli/neutral_corpus.jsonl. THEN submit this GPU job:
#   qsub jobs/sge/extract_neutral.sh
#
# Requires the Phase-4 outputs to already exist in data/processed/concept_vectors/
# (warmth_vec.npy, competence_vec.npy, X_<condition>.npy from extract_vectors.sh).
#
# ADJUST: verify queue names / GPU flag syntax with Stefan if needed.

#$ -N wc_extract_neutral
#$ -q gpu@scc192,gpu@scc213,gpu@scc214
#$ -l h_rt=01:00:00
#$ -l h_vmem=64G
#$ -pe smp 2
#$ -l gpu=1
#$ -o results/logs/wc_extract_neutral.out
#$ -e results/logs/wc_extract_neutral.err
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

mkdir -p results/logs data/processed/concept_vectors

echo "[job] Step 1: extract neutral-corpus activations (GPU)"
python src/extract_neutral.py --config config/config.yaml

echo "[job] Step 2: PCA denoise warmth/competence vectors (CPU)"
python src/denoise_vectors.py --config config/config.yaml

echo "[job] Done."
