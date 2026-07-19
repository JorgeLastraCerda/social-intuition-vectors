# Qwen3.6 35B-A3B Neutral-PCA Denoising

- **Produced:** 2026-07-19 10:24 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-35B-A3B`
- **Scope:** Remove high-variance neutral residual directions from warmth and competence vectors
- **Status:** Complete; denoised hiring intervention active

## Artifacts

- **Scripts:** `src/denoise_vectors.py`
- **Inputs:** `config/qwen36_35b_a3b.yaml`, `data/processed/concept_vectors_qwen36_35b_a3b/X_neutral.npy`, `data/processed/concept_vectors_qwen36_35b_a3b/warmth_vec.npy`, `data/processed/concept_vectors_qwen36_35b_a3b/competence_vec.npy`, `data/processed/concept_vectors_qwen36_35b_a3b/X_high_warmth.npy`, `data/processed/concept_vectors_qwen36_35b_a3b/X_low_warmth.npy`, `data/processed/concept_vectors_qwen36_35b_a3b/X_high_competence.npy`, `data/processed/concept_vectors_qwen36_35b_a3b/X_low_competence.npy`
- **Outputs:** `data/processed/concept_vectors_qwen36_35b_a3b/concept_vectors_denoised.npz`, `data/processed/concept_vectors_qwen36_35b_a3b/denoise_summary.json`

## Result

Seventeen neutral principal components explained 50.28% of the neutral activation variance and were removed from both concept vectors. Warmth-competence cosine similarity decreased from 0.619 to 0.595.

Concept-story separation increased from Cohen's d=6.25 to 7.34 for warmth and from d=7.28 to 8.07 for competence. Warmth-on-competence leakage also increased substantially, from d=5.49 to 6.72. Neutral-PCA removal therefore does not disentangle the axes and is retained only as a causal robustness transformation.
