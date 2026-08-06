# Gemma 4 26B-A4B Post-hoc Hiring Analyses

- **Produced:** 2026-07-18 23:35 Europe/Berlin
- **Model:** `google/gemma-4-26B-A4B-it`
- **Scope:** Demographic disparity, bootstrap mediation, group-level R4, and name-level R4 analyses
- **Status:** Complete and validated; R4 paragraph corrected 2026-08-06 (flake_leasure/kline bug fix, see `step_logs/STEP_LOG.md`)

## Artifacts

- **Scripts:** `src/hiring_disparity.py`, `src/hiring_r4.py`, `src/validate_gemma4_remaining.py`
- **Inputs:** `config/gemma4_26b_a4b.yaml`, `results/tables/hiring_audit_gemma4_26b_a4b.csv`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`
- **Outputs:** `results/tables/hiring_disparity_gemma4_26b_a4b.csv`, `results/logs/hiring_mediation_gemma4_26b_a4b.json`, `results/tables/hiring_group_r4_gemma4_26b_a4b.csv`, `results/tables/hiring_name_level_gemma4_26b_a4b.csv`, `results/logs/hiring_r4_gemma4_26b_a4b.json`

## Result

The disparity and mediation join matched 269 names. Mean model callback margins were 21.509 for Black names and 21.095 for White names, and 21.464 for female names and 20.902 for male names.

Two mediation intervals excluded zero: gender through warmth was +0.0384 (95% CI [0.0072, 0.0784]), and race through competence was +0.1308 (95% CI [0.0600, 0.2313]). The remaining two paths were not significant at the 95% interval level.

The R4 join matched 246 name-study observations across 186 distinct names. Both model warmth (r = 0.288, p = 4.3e-6) and model competence (r = 0.343, p = 3.4e-8) correlated with callback margin; human callback did not (r = -0.105, p = 0.099). The multivariable standardized coefficients were 0.074 for warmth and 0.161 for competence, with R2 = 0.144.
