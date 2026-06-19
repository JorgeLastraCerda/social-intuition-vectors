#!/bin/bash
# SGE job: Qwen3-14B smoke test — TransformerLens, 50+50 warm/cold probe
# Submit from repo root: qsub jobs/sge/smoke_qwen3_14b.sh
#
# Model: Qwen/Qwen3-14B  (~28 GB bf16)  — needs a >=48 GB GPU node.
# Uses the family-neutral smoke_tests/transformerlens_probe.py.
# Pass criterion: probe_cv_mean > 0.80  (same as Gemma 3 12B baseline).
# No SAE step: no GemmaScope equivalent exists for Qwen3.
#
# ADJUST: verify that scc192/scc213/scc214 have >=48 GB VRAM on your cluster;
#         add or remove nodes in the -q line as appropriate.
# ADJUST: confirm the exact GPU resource flag syntax with Stefan.
# ADJUST: h_vmem semantics (per-slot vs total) on SCCKN; 64G is a starting point.
# ADJUST: module name if different from "conda".
# Model VRAM estimates (bf16):
#   Qwen3-14B  ~28 GB  -> needs a >=48 GB GPU (L40 or RTX 6000 96 GB)

#$ -N smk_qw3_14b
#$ -q gpu@scc192,gpu@scc213,gpu@scc214   # ADJUST: three big-VRAM nodes
#$ -l h_rt=01:00:00
#$ -l h_vmem=64G                          # ADJUST: RAM per slot
#$ -pe smp 2                              # ADJUST: CPU cores
#$ -l gpu=1
#$ -o results/logs/smoke_qwen3_14b.out
#$ -e results/logs/smoke_qwen3_14b.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

module load conda                         # ADJUST: module name if different
conda activate wc-tl

export HF_HOME=/work/emrecan.ulu/hf_cache

cd /work/emrecan.ulu/normalcy-axis
git pull

mkdir -p results/logs smoke_tests/results/qwen3_14b

python smoke_tests/transformerlens_probe.py \
    --model Qwen/Qwen3-14B \
    --label qwen3_14b \
    --start-token 1 \
    --seed 20260527 \
    --out-dir smoke_tests/results/qwen3_14b
