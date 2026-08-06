# Gemma 4 12B Post-hoc Hiring Analyses

- **Produced:** 2026-07-18 23:14 Europe/Berlin
- **Model:** `google/gemma-4-12B-it`
- **Scope:** Demographic disparity, bootstrap mediation, group-level R4, and name-level R4 analyses
- **Status:** Complete and validated; R4 paragraph corrected 2026-08-06 (flake_leasure/kline bug fix, see `step_logs/STEP_LOG.md`)

## Artifacts

- **Scripts:** `src/hiring_disparity.py`, `src/hiring_r4.py`, `src/validate_gemma4_remaining.py`
- **Inputs:** `config/gemma4_12b.yaml`, `results/tables/hiring_audit_gemma4_12b.csv`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`
- **Outputs:** `results/tables/hiring_disparity_gemma4_12b.csv`, `results/logs/hiring_mediation_gemma4_12b.json`, `results/tables/hiring_group_r4_gemma4_12b.csv`, `results/tables/hiring_name_level_gemma4_12b.csv`, `results/logs/hiring_r4_gemma4_12b.json`

## Result

The disparity and mediation join matched 269 of 282 audited names to published callback data. Mean model callback margins were 17.194 for Black names and 17.172 for White names, and 17.208 for female names and 17.175 for male names. These are descriptive within-model differences and should not be interpreted as human callback effects.

Only one of four predeclared mediation intervals excluded zero. The indirect path from Black-name coding through the competence probe to model callback margin was -0.1498 (95% bootstrap CI [-0.2950, -0.0190]). The warmth race path and both gender paths were not significant at the 95% interval level.

The exact-study R4 join matched 246 name-study observations across 186 distinct names. Name-level callback margin correlated positively with model competence, r = 0.247 (p = 8.9e-5), but not with model warmth, r = -0.056 (p = 0.385), or human callback, r = 0.066 (p = 0.305). The multivariable standardized OLS coefficients were 0.052 for model competence, -0.031 for model warmth, and 0.012 for human callback, with R2 = 0.088.
