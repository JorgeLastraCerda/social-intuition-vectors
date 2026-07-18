# Gemma 4 26B-A4B Denoised Local Full-282 Hiring Steering

- **Produced:** 2026-07-19 00:22 Europe/Berlin
- **Model:** `google/gemma-4-26B-A4B-it`
- **Scope:** Neutral-PCA-denoised local steering across all 282 rated names
- **Status:** Complete and validated

## Artifacts

- **Scripts:** `src/hiring_steering.py`, `src/summarize_hiring_steering.py`, `src/validate_gemma4_remaining.py`, `jobs/ccu/run_gemma4_remaining.sh`, `jobs/ccu/run_gemma4_remaining_queue.sh`
- **Inputs:** `config/gemma4_26b_a4b.yaml`, `data/processed/concept_vectors_gemma4_26b_a4b/concept_vectors_denoised.npz`
- **Outputs:** `results/tables/hiring_steering_raw_gemma4_26b_a4b_denoised_local_full282.csv`, `results/tables/hiring_steering_gemma4_26b_a4b_denoised_local_full282.csv`, `results/logs/hiring_steering_gemma4_26b_a4b_denoised_local_full282.json`, `results/logs/hiring_steering_summary_gemma4_26b_a4b_denoised_local_full282.json`

## Result

All 2,820 rows passed validation. Denoised warmth was monotone with slope 2.305, R2 = 0.890, and +0.10 endpoint +0.075 (95% CI [0.050, 0.100]). Denoised competence remained non-monotone with near-zero R2 and endpoint -0.461 (95% CI [-0.491, -0.433]). Denoising therefore did not remove the axis-specific instability.
