#!/bin/bash
# Generic Gemma 4 TransformerLens Bridge smoke gate. Submit via submit_gemma4.sh.
# ADJUST: queue, module, memory and GPU resource syntax for SCCKN if needed.
#$ -q gpu@scc214
#$ -l h_rt=01:00:00
#$ -l h_vmem=96G
#$ -pe smp 2
#$ -l gpu=1
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

set -euo pipefail
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${LABEL:?LABEL is required}"
: "${EXPECTED_LAYERS:?EXPECTED_LAYERS is required}"
: "${EXPECTED_D_MODEL:?EXPECTED_D_MODEL is required}"

module load conda  # ADJUST
conda activate wc-tl-g4
export HF_HOME=/work/emrecan.ulu/hf_cache
export PYTHONPATH=/work/emrecan.ulu/normalcy-axis
cd /work/emrecan.ulu/normalcy-axis
git pull --ff-only
mkdir -p results/logs

python smoke_tests/gemma4_transformerlens/smoke_test_bridge.py \
  --model "$MODEL_NAME" \
  --expected-layers "$EXPECTED_LAYERS" \
  --expected-d-model "$EXPECTED_D_MODEL" \
  --output "results/logs/smoke_${LABEL}.json"
bash jobs/sync_outputs.sh /work/emrecan.ulu/normalcy-axis
