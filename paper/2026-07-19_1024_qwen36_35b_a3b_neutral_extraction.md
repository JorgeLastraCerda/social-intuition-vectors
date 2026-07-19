# Qwen3.6 35B-A3B Neutral Activation Extraction

- **Produced:** 2026-07-19 10:24 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-35B-A3B`
- **Scope:** Probe-layer residual activations for 1,500 neutral passages
- **Status:** Complete and validated on CCU H100

## Artifacts

- **Scripts:** `src/qwen36_hiring.py`, `src/validate_qwen36_hiring.py`, `src/steering_checkpoint.py`, `jobs/ccu/run_qwen36_hiring.sh`
- **Inputs:** `config/qwen36_35b_a3b.yaml`, `data/stimuli/neutral_corpus.jsonl`, `data/processed/concept_vectors_qwen36_35b_a3b/warmth_vec.npy`, `data/processed/concept_vectors_qwen36_35b_a3b/competence_vec.npy`
- **Outputs:** `data/processed/concept_vectors_qwen36_35b_a3b/X_neutral.npy`, `data/processed/concept_vectors_qwen36_35b_a3b/neutral_meta.json`

## Result

The native-HF extractor completed all 1,500 immutable checkpoint shards and published a finite 1,500 by 2,048 activation matrix from layer 26. The run took 449.2 seconds on one NVIDIA H100 80GB HBM3 and peaked at 65.55 GiB allocated VRAM.

The pinned revision, passive-hook parity, native hidden-state parity, single-device placement, text-only execution, and no-TransformerLens gates all passed. The local matrix hash exactly matches the remote output (`b45f88c01c704dddc1c6942da64c0a7bc9001e1d6aab6dfedc86d96c7402ff4c`).
