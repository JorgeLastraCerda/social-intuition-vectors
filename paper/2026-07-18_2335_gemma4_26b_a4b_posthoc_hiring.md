# Gemma 4 26B-A4B Post-hoc Hiring Analyses

- **Produced:** 2026-07-18 23:35 Europe/Berlin
- **Model:** `google/gemma-4-26B-A4B-it`
- **Scope:** Demographic disparity, bootstrap mediation, group-level R4, and name-level R4 analyses
- **Status:** Complete and validated

## Artifacts

- **Scripts:** `src/hiring_disparity.py`, `src/hiring_r4.py`, `src/validate_gemma4_remaining.py`
- **Inputs:** `config/gemma4_26b_a4b.yaml`, `results/tables/hiring_audit_gemma4_26b_a4b.csv`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`
- **Outputs:** `results/tables/hiring_disparity_gemma4_26b_a4b.csv`, `results/logs/hiring_mediation_gemma4_26b_a4b.json`, `results/tables/hiring_group_r4_gemma4_26b_a4b.csv`, `results/tables/hiring_name_level_gemma4_26b_a4b.csv`, `results/logs/hiring_r4_gemma4_26b_a4b.json`

## Result

The disparity and mediation join matched 269 names. Mean model callback margins were 21.509 for Black names and 21.095 for White names, and 21.464 for female names and 20.902 for male names.

Two mediation intervals excluded zero: gender through warmth was +0.0384 (95% CI [0.0072, 0.0784]), and race through competence was +0.1308 (95% CI [0.0600, 0.2313]). The remaining two paths were not significant at the 95% interval level.

The R4 join matched 149 names. Model competence correlated with callback margin at r = 0.356 (p = 8.60e-6), while model warmth and human callback did not. The multivariable standardized competence coefficient was 0.252, and model R2 was 0.142.
