#!/bin/bash
#$ -N wc_download_sources
#$ -q scc                 # ADJUST
#$ -pe smp 1              # ADJUST
#$ -l h_vmem=4G           # ADJUST
#$ -l h_rt=01:00:00       # ADJUST
#$ -o results/logs/$JOB_NAME.$JOB_ID.out
#$ -e results/logs/$JOB_NAME.$JOB_ID.err
#$ -cwd

set -euo pipefail

module load conda         # ADJUST
source activate python-3.13  # ADJUST

mkdir -p papers data/raw results/logs

curl -L -o literature/emotion_concepts_anthropic_2026.pdf https://arxiv.org/pdf/2604.07729
curl -L -o literature/warmth_competence_callback_plos_2024.pdf "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0304723&type=printable"

if [ ! -d data/raw/SocialPerceptions-Predict-Callback ]; then
  git clone https://github.com/carinahausladen/SocialPerceptions-Predict-Callback.git data/raw/SocialPerceptions-Predict-Callback
fi

file literature/*.pdf
ls -lh literature/
