# Qwen3.6 27B Posthoc Hiring Analyses

- **Produced:** 2026-07-19 09:53 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-27B`
- **Scope:** Demographic disparity, probe mediation, group-level R4, and name-level R4
- **Status:** Complete from the validated 282-name audit; R4 paragraph corrected 2026-08-06 (flake_leasure/kline bug fix, see `step_logs/STEP_LOG.md`)

## Artifacts

- **Scripts:** `src/hiring_disparity.py`, `src/hiring_r4.py`
- **Inputs:** `config/qwen36_27b.yaml`, `results/tables/hiring_audit_qwen36_27b.csv`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`
- **Outputs:** `results/tables/hiring_disparity_qwen36_27b.csv`, `results/logs/hiring_mediation_qwen36_27b.json`, `results/tables/hiring_group_r4_qwen36_27b.csv`, `results/tables/hiring_name_level_qwen36_27b.csv`, `results/logs/hiring_r4_qwen36_27b.json`

## Result

The disparity and mediation join matched 269 of 282 audited names to published callback data. Mean model callback margins were 5.402 for Black names and 5.328 for White names, compared with human callback rates of 0.183 and 0.171. Female and male names had mean model margins of 5.384 and 5.284, while their human callback rates were 0.145 and 0.182.

Warmth did not show a statistically resolved indirect effect for race or gender. Competence did: the race indirect effect was -0.0488 (95% bootstrap CI [-0.1035, -0.0106]), and the gender indirect effect was -0.1227 (95% CI [-0.2056, -0.0608]). These mediation estimates are associational decomposition results rather than randomized causal mediation.

The exact study-and-name R4 join retained 246 name-study observations across 186 distinct names. Across eleven race-gender-study groups, model and human means had Pearson r=-0.249, with a group-level p-value of 0.461. At the name level, callback margin correlated weakly with human callback rate (r=0.043, p=0.497), positively with the model warmth projection (r=0.401, p=6.3e-11), and positively in the bivariate correlation with model competence (r=0.200, p=0.0016). The joint standardized coefficients were +0.169 for warmth and -0.090 for competence; the negative competence coefficient indicates overlap between the two probe predictors and should not be read as contradicting the bivariate association.
