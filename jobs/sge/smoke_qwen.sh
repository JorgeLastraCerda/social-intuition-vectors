#!/bin/bash
# SGE job: Qwen2.5-1.5B-Instruct smoke test (TransformerLens)
# Submit from repo root: qsub jobs/sge/smoke_qwen.sh
#
# ADJUST: queue, GPU resource flag, h_vmem, module names before submitting.

#$ -N smoke_qwen
#$ -q gpu                          # ADJUST: GPU queue name on SCCKN
#$ -l h_rt=01:00:00
#$ -l h_vmem=16G                   # ADJUST: RAM per slot
#$ -pe smp 4                       # ADJUST: CPU cores
#$ -l gpu=1                        # ADJUST: exact GPU resource flag (ask Stefan)
#$ -o results/logs/smoke_qwen.out
#$ -e results/logs/smoke_qwen.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

module load conda                  # ADJUST: module name if different
conda activate wc-tl

export HF_HOME=/work/emrecan.ulu/hf_cache

cd /work/emrecan.ulu/normalcy-axis
git pull

mkdir -p results/logs

python smoke_tests/qwen_transformerlens/smoke_test_probe.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --start-token 1 \
    --seed 20260527
