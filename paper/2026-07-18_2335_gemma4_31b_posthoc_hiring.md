# Gemma 4 31B Post-hoc Hiring Analyses

- **Produced:** 2026-07-18 23:35 Europe/Berlin
- **Model:** `google/gemma-4-31B-it`
- **Scope:** Demographic disparity, bootstrap mediation, group-level R4, and name-level R4 analyses
- **Status:** Complete and validated; R4 paragraph corrected 2026-08-06 (flake_leasure/kline bug fix, see `step_logs/STEP_LOG.md`)

## Artifacts

- **Scripts:** `src/hiring_disparity.py`, `src/hiring_r4.py`, `src/validate_gemma4_remaining.py`
- **Inputs:** `config/gemma4_31b.yaml`, `results/tables/hiring_audit_gemma4_31b.csv`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`
- **Outputs:** `results/tables/hiring_disparity_gemma4_31b.csv`, `results/logs/hiring_mediation_gemma4_31b.json`, `results/tables/hiring_group_r4_gemma4_31b.csv`, `results/tables/hiring_name_level_gemma4_31b.csv`, `results/logs/hiring_r4_gemma4_31b.json`

## Result

The disparity and mediation join matched 269 names. Mean model callback margins were 25.808 for Black names and 25.577 for White names, and 25.788 for female names and 25.408 for male names.

The race-through-competence indirect path was the only interval excluding zero, at -0.2300 (95% CI [-0.4835, -0.0486]). The warmth race path and both gender paths were not significant at the 95% interval level.

The R4 join matched 246 name-study observations across 186 distinct names. Model competence correlated with callback margin at r = 0.367 (p = 2.9e-9); model warmth and human callback did not. The multivariable standardized coefficients were 1.098 for competence, -0.846 for warmth, and 0.136 for human callback, with R2 = 0.300. The opposing probe coefficients warrant interpretation alongside their correlation and collinearity structure.
