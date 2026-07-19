# Qwen3.6 35B-A3B Broad Hiring Steering

- **Produced:** 2026-07-19 10:14 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-35B-A3B`
- **Scope:** Raw warmth and competence steering across 60 held names at broad strengths
- **Status:** Complete and validated on CCU H100; robustness failure triggers conditional expansion

## Artifacts

- **Scripts:** `src/qwen36_hiring.py`, `src/validate_qwen36_hiring.py`, `src/summarize_hiring_steering.py`, `src/steering_checkpoint.py`, `jobs/ccu/run_qwen36_hiring.sh`
- **Inputs:** `config/qwen36_35b_a3b.yaml`, `data/processed/concept_vectors_qwen36_35b_a3b/`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/ratings/names/df_all.csv`
- **Outputs:** `results/tables/hiring_steering_raw_qwen36_35b_a3b_broad.csv`, `results/tables/hiring_steering_qwen36_35b_a3b_broad.csv`, `results/logs/hiring_steering_qwen36_35b_a3b_broad.json`, `results/logs/hiring_steering_summary_qwen36_35b_a3b_broad.json`

## Result

The broad-strength run completed all 660 atomic work units and produced 600 validated rows. Warmth ended positive at +0.50, with a mean callback-margin change of +1.233 (95% bootstrap CI [1.188, 1.281]), but its five-point response was not monotone. The fitted warmth slope was 3.728 with R-squared 0.909.

Competence failed several broad-range robustness checks. Its +0.50 endpoint reversed to -1.094 (95% CI [-1.131, -1.054]) even though the fitted slope was positive (+1.449). The curve was non-monotone, the endpoint sign disagreed with the slope, and R-squared was only 0.296.

These failures are intervention-range behavior, not a hook, model-load, or validation error. They are sufficient to trigger the predeclared 282-name expansion after the denoised-local regime completes.
