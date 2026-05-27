#!/bin/bash
#$ -N wc_extract_vectors
#$ -q gpu                 # ADJUST
#$ -pe smp 8              # ADJUST
#$ -l h_vmem=16G          # ADJUST
#$ -l h_rt=04:00:00       # ADJUST
#$ -o results/logs/$JOB_NAME.$JOB_ID.out
#$ -e results/logs/$JOB_NAME.$JOB_ID.err
#$ -cwd

set -euo pipefail

module load conda         # ADJUST
source activate python-3.13  # ADJUST
export HF_HOME=/path/to/scratch/hf_cache  # ADJUST

python src/extract_vectors.py --config config/config.yaml
