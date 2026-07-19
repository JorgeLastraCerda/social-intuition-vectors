# Qwen3.6 27B Neutral-PCA Denoising

- **Produced:** 2026-07-19 10:03 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-27B`
- **Scope:** Remove high-variance neutral residual directions from warmth and competence vectors
- **Status:** Complete; denoised hiring intervention pending

## Artifacts

- **Scripts:** `src/denoise_vectors.py`
- **Inputs:** `config/qwen36_27b.yaml`, `data/processed/concept_vectors_qwen36_27b/X_neutral.npy`, `data/processed/concept_vectors_qwen36_27b/warmth_vec.npy`, `data/processed/concept_vectors_qwen36_27b/competence_vec.npy`, `data/processed/concept_vectors_qwen36_27b/X_high_warmth.npy`, `data/processed/concept_vectors_qwen36_27b/X_low_warmth.npy`, `data/processed/concept_vectors_qwen36_27b/X_high_competence.npy`, `data/processed/concept_vectors_qwen36_27b/X_low_competence.npy`
- **Outputs:** `data/processed/concept_vectors_qwen36_27b/concept_vectors_denoised.npz`, `data/processed/concept_vectors_qwen36_27b/denoise_summary.json`

## Result

Twenty-seven neutral principal components explained 50.23% of the neutral activation variance and were projected out of both concept vectors. Warmth-competence cosine similarity decreased modestly from 0.580 to 0.560.

On the original concept-story activations, warmth separation increased from Cohen's d=7.90 to 8.11 and competence separation increased from d=8.90 to 9.86. The warmth vector's separation of competence conditions also increased from d=5.44 to 5.90, so neutral-PCA removal did not eliminate cross-axis leakage. A denoised-local hiring run is required to determine whether the causal callback response remains stable after this transformation.
