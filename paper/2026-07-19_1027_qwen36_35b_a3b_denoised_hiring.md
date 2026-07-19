# Qwen3.6 35B-A3B Denoised Local Hiring Steering

- **Produced:** 2026-07-19 10:27 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-35B-A3B`
- **Scope:** Neutral-PCA-denoised warmth and competence steering across 60 held names
- **Status:** Complete and validated on CCU H100

## Artifacts

- **Scripts:** `src/qwen36_hiring.py`, `src/validate_qwen36_hiring.py`, `src/summarize_hiring_steering.py`, `src/steering_checkpoint.py`, `jobs/ccu/run_qwen36_hiring.sh`
- **Inputs:** `config/qwen36_35b_a3b.yaml`, `data/processed/concept_vectors_qwen36_35b_a3b/concept_vectors_denoised.npz`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/ratings/names/df_all.csv`
- **Outputs:** `results/tables/hiring_steering_raw_qwen36_35b_a3b_denoised_local.csv`, `results/tables/hiring_steering_qwen36_35b_a3b_denoised_local.csv`, `results/logs/hiring_steering_qwen36_35b_a3b_denoised_local.json`, `results/logs/hiring_steering_summary_qwen36_35b_a3b_denoised_local.json`

## Result

The denoised-vector run completed all 660 atomic work units and produced 600 validated rows. Warmth remained monotone, with a +0.10 callback-margin effect of +1.004 (95% bootstrap CI [0.944, 1.065]), slope 8.758, and R-squared 0.990. Competence was also monotone, with a +0.10 effect of +0.438 (95% CI [0.390, 0.483]), slope 4.817, and R-squared 0.994.

Compared with raw local steering, the denoised warmth endpoint increased by about 3.9% and competence by about 1.0%. Neutral-PCA removal therefore preserved the local causal response and did not repair or directly test the separate broad-range competence reversal.
