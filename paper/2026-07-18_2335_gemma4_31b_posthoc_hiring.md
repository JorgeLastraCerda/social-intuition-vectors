# Gemma 4 31B Post-hoc Hiring Analyses

- **Produced:** 2026-07-18 23:35 Europe/Berlin
- **Model:** `google/gemma-4-31B-it`
- **Scope:** Demographic disparity, bootstrap mediation, group-level R4, and name-level R4 analyses
- **Status:** Complete and validated

## Artifacts

- **Scripts:** `src/hiring_disparity.py`, `src/hiring_r4.py`, `src/validate_gemma4_remaining.py`
- **Inputs:** `config/gemma4_31b.yaml`, `results/tables/hiring_audit_gemma4_31b.csv`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`
- **Outputs:** `results/tables/hiring_disparity_gemma4_31b.csv`, `results/logs/hiring_mediation_gemma4_31b.json`, `results/tables/hiring_group_r4_gemma4_31b.csv`, `results/tables/hiring_name_level_gemma4_31b.csv`, `results/logs/hiring_r4_gemma4_31b.json`

## Result

The disparity and mediation join matched 269 names. Mean model callback margins were 25.808 for Black names and 25.577 for White names, and 25.788 for female names and 25.408 for male names.

The race-through-competence indirect path was the only interval excluding zero, at -0.2300 (95% CI [-0.4835, -0.0486]). The warmth race path and both gender paths were not significant at the 95% interval level.

The R4 join matched 149 names. Model competence correlated with callback margin at r = 0.482 (p = 4.74e-10); model warmth and human callback did not. The multivariable standardized coefficients were 1.469 for competence, -0.980 for warmth, and -0.018 for human callback, with R2 = 0.407. The opposing probe coefficients warrant interpretation alongside their correlation and collinearity structure.
