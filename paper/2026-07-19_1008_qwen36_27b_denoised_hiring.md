# Qwen3.6 27B Denoised Local Hiring Steering

- **Produced:** 2026-07-19 10:08 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-27B`
- **Scope:** Neutral-PCA-denoised warmth and competence steering across 60 held names
- **Status:** Complete and validated on CCU H100

## Artifacts

- **Scripts:** `src/qwen36_hiring.py`, `src/validate_qwen36_hiring.py`, `src/summarize_hiring_steering.py`, `src/steering_checkpoint.py`, `jobs/ccu/run_qwen36_hiring.sh`
- **Inputs:** `config/qwen36_27b.yaml`, `data/processed/concept_vectors_qwen36_27b/concept_vectors_denoised.npz`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/ratings/names/df_all.csv`
- **Outputs:** `results/tables/hiring_steering_raw_qwen36_27b_denoised_local.csv`, `results/tables/hiring_steering_qwen36_27b_denoised_local.csv`, `results/logs/hiring_steering_qwen36_27b_denoised_local.json`, `results/logs/hiring_steering_summary_qwen36_27b_denoised_local.json`

## Result

The denoised-vector run completed all 660 atomic work units and produced 600 validated rows. Warmth remained monotone over local strengths, with a +0.10 callback-margin effect of +1.140 (95% bootstrap CI [1.113, 1.165]), slope 12.025, and R-squared 0.999. Competence also remained monotone, with a +0.10 effect of +0.408 (95% CI [0.381, 0.438]), slope 6.654, and R-squared 0.950.

Relative to raw local steering, PCA denoising attenuated the positive endpoint by about 4.7% for warmth and 23.4% for competence without changing either sign or monotonic ordering. The callback-margin quantization warning remains active because all baseline values lie on the 0.125 grid.
