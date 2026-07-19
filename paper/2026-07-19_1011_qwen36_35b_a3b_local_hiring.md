# Qwen3.6 35B-A3B Local Hiring Steering

- **Produced:** 2026-07-19 10:11 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-35B-A3B`
- **Scope:** Raw warmth and competence steering across 60 held names at local strengths
- **Status:** Complete and validated on CCU H100

## Artifacts

- **Scripts:** `src/qwen36_hiring.py`, `src/validate_qwen36_hiring.py`, `src/summarize_hiring_steering.py`, `src/steering_checkpoint.py`, `jobs/ccu/run_qwen36_hiring.sh`
- **Inputs:** `config/qwen36_35b_a3b.yaml`, `data/processed/concept_vectors_qwen36_35b_a3b/`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/ratings/names/df_all.csv`
- **Outputs:** `results/tables/hiring_steering_raw_qwen36_35b_a3b_local.csv`, `results/tables/hiring_steering_qwen36_35b_a3b_local.csv`, `results/logs/hiring_steering_qwen36_35b_a3b_local.json`, `results/logs/hiring_steering_summary_qwen36_35b_a3b_local.json`

## Result

The native-HF run completed all 660 atomic work units and produced 600 validated rows. Warmth was monotone over local strengths and reached a +0.10 mean callback-margin change of +0.967 (95% bootstrap CI [0.900, 1.033]); its fitted slope was 8.525 with R-squared 0.991. Competence was also monotone, with a +0.10 effect of +0.433 (95% CI [0.394, 0.477]), slope 4.621, and R-squared 0.998.

Both causal responses agree in sign with the observational audit and satisfy the predeclared local-regime shape checks. Baseline callback margins had only seven unique values and all fell on the 0.125 grid, so the quantization warning remains active.
