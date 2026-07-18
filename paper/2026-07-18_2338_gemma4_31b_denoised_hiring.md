# Gemma 4 31B Denoised Local Hiring Steering

- **Produced:** 2026-07-18 23:38 Europe/Berlin
- **Model:** `google/gemma-4-31B-it`
- **Scope:** Missing 60-name neutral-PCA-denoised local hiring-steering prerequisite
- **Status:** Complete and validated on CCU H100

## Artifacts

- **Scripts:** `src/hiring_steering.py`, `src/summarize_hiring_steering.py`, `src/validate_gemma4_remaining.py`, `jobs/ccu/run_gemma4_remaining.sh`
- **Inputs:** `config/gemma4_31b.yaml`, `data/processed/concept_vectors_gemma4_31b/concept_vectors_denoised.npz`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`
- **Outputs:** `results/tables/hiring_steering_raw_gemma4_31b_denoised_local.csv`, `results/tables/hiring_steering_gemma4_31b_denoised_local.csv`, `results/logs/hiring_steering_gemma4_31b_denoised_local.json`, `results/logs/hiring_steering_summary_gemma4_31b_denoised_local.json`

## Result

The run produced and validated 600 raw rows across 60 names. Neither denoised direction was monotone over the local grid. Warmth had slope 4.326 but a +0.10 endpoint of -0.233 (95% CI [-0.279, -0.183]); competence had slope 2.516, R2 = 0.220, and endpoint -0.524 (95% CI [-0.562, -0.481]). Both endpoint signs opposed their fitted slopes.

This result completes the missing 31B denoised prerequisite and contributes six of the sixteen reasons that fired the full-282 expansion gate.
