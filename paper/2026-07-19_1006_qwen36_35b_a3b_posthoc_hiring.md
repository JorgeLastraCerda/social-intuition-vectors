# Qwen3.6 35B-A3B Posthoc Hiring Analyses

- **Produced:** 2026-07-19 10:06 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-35B-A3B`
- **Scope:** Demographic disparity, probe mediation, group-level R4, and name-level R4
- **Status:** Complete from the validated 282-name audit; R4 paragraph corrected 2026-08-06 (flake_leasure/kline bug fix, see `step_logs/STEP_LOG.md`)

## Artifacts

- **Scripts:** `src/hiring_disparity.py`, `src/hiring_r4.py`
- **Inputs:** `config/qwen36_35b_a3b.yaml`, `results/tables/hiring_audit_qwen36_35b_a3b.csv`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`
- **Outputs:** `results/tables/hiring_disparity_qwen36_35b_a3b.csv`, `results/logs/hiring_mediation_qwen36_35b_a3b.json`, `results/tables/hiring_group_r4_qwen36_35b_a3b.csv`, `results/tables/hiring_name_level_qwen36_35b_a3b.csv`, `results/logs/hiring_r4_qwen36_35b_a3b.json`

## Result

The disparity and mediation join matched 269 audited names. Mean model callback margins were 2.981 for Black names and 3.039 for White names; female and male names had means of 3.143 and 2.873. Published human callback rates showed the opposite gender ordering, 0.145 for female names and 0.182 for male names.

The warmth indirect effect excluded zero for race (-0.1033, 95% bootstrap CI [-0.1685, -0.0459]) and gender (+0.0639, 95% CI [0.0266, 0.1131]). The competence race interval included zero, while the gender effect was small and barely resolved (+0.0289, 95% CI [0.0009, 0.0671]). These are associational decompositions, not randomized causal mediation estimates.

The exact study-and-name R4 join retained 246 name-study observations across 186 distinct names. Across eleven race-gender-study groups, model and human means correlated at r=-0.229 (p=0.498). Name-level callback margin was weakly negatively correlated with human callback rate (r=-0.156, p=0.014) and positively correlated with the model warmth projection (r=0.442, p=3.5e-13) and model competence projection (r=0.281, p=7.9e-6). In their joint regression, warmth retained a positive standardized coefficient (+0.164), while competence was slightly negative (-0.064), reflecting shared probe variance; human callback contributed a small negative coefficient (-0.026).
