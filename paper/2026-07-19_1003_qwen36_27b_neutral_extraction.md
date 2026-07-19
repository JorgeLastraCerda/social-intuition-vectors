# Qwen3.6 27B Neutral Activation Extraction

- **Produced:** 2026-07-19 10:03 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-27B`
- **Scope:** Probe-layer residual activations for 1,500 neutral passages
- **Status:** Complete and validated on CCU H100

## Artifacts

- **Scripts:** `src/qwen36_hiring.py`, `src/validate_qwen36_hiring.py`, `src/steering_checkpoint.py`, `jobs/ccu/run_qwen36_hiring.sh`
- **Inputs:** `config/qwen36_27b.yaml`, `data/stimuli/neutral_corpus.jsonl`, `data/processed/concept_vectors_qwen36_27b/warmth_vec.npy`, `data/processed/concept_vectors_qwen36_27b/competence_vec.npy`
- **Outputs:** `data/processed/concept_vectors_qwen36_27b/X_neutral.npy`, `data/processed/concept_vectors_qwen36_27b/neutral_meta.json`

## Result

The native-HF extractor completed all 1,500 immutable checkpoint shards and published a finite 1,500 by 5,120 probe-layer activation matrix. It used raw neutral passages with an explicit beginning-of-sequence token at layer 42. The run took 353.3 seconds on one NVIDIA H100 80GB HBM3 and peaked at 51.38 GiB allocated VRAM.

The resolved model revision matched the requested pin, passive hooks did not change logits, hook captures matched native hidden states exactly, the vision path received zero calls, and TransformerLens was neither installed nor imported. The local copy was reconstructed from a compressed, base64-safe CCU transfer and its SHA-256 hash exactly matched the remote matrix (`cc362c2c929807cad5fecb996072477230a68f072556c13a93f11fe3f1ce66b9`).
