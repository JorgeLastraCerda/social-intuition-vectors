#!/bin/bash
# SGE job: Llama-3.1-8B-Instruct smoke test — TransformerLens, 50+50 warm/cold probe
# Submit from repo root: qsub jobs/sge/smoke_llama31_8b.sh
#
# Model: meta-llama/Llama-3.1-8B-Instruct  (~16 GB bf16)  — fits any L40 node.
# Uses the family-neutral smoke_tests/transformerlens_probe.py.
# Pass criterion: probe_cv_mean > 0.80  (same as Gemma 3 12B baseline).
# No SAE step (though Llama Scope SAEs exist; add sae_decompose later if needed).
#
# ADJUST: verify that scc192/scc213/scc214 have >=48 GB VRAM on your cluster;
#         Llama 3.1 8B only needs ~16 GB, so any GPU node works — keep node pool
#         for faster scheduling.
# ADJUST: confirm the exact GPU resource flag syntax with Stefan.
# ADJUST: h_vmem semantics (per-slot vs total) on SCCKN; 32G is a starting point.
# ADJUST: module name if different from "conda".
# Model VRAM estimates (bf16):
#   Llama-3.1-8B-Instruct  ~16 GB  -> fits any GPU node with >=16 GB VRAM

#$ -N smk_ll31_8b
#$ -q gpu@scc192,gpu@scc213,gpu@scc214   # ADJUST: any big-VRAM node; or use "-q gpu" for any GPU
#$ -l h_rt=01:00:00
#$ -l h_vmem=64G                          # ADJUST: RAM per slot (32G OOM'd on scc214; 64G confirmed working)
#$ -pe smp 2                              # ADJUST: CPU cores
#$ -l gpu=1
#$ -o results/logs/smoke_llama31_8b.out
#$ -e results/logs/smoke_llama31_8b.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

module load conda                         # ADJUST: module name if different
conda activate wc-tl

export HF_HOME=/work/emrecan.ulu/hf_cache

cd /work/emrecan.ulu/normalcy-axis
git pull

mkdir -p results/logs smoke_tests/results/llama31_8b

python smoke_tests/transformerlens_probe.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --label llama31_8b \
    --start-token 1 \
    --seed 20260527 \
    --out-dir smoke_tests/results/llama31_8b
