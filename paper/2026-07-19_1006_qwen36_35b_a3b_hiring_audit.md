# Qwen3.6 35B-A3B Hiring Audit

- **Produced:** 2026-07-19 10:06 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-35B-A3B`
- **Scope:** Full 282-name warmth/competence projection, human-rating alignment, and unsteered callback-margin audit
- **Status:** Complete and validated on CCU H100

## Artifacts

- **Scripts:** `src/qwen36_hiring.py`, `src/validate_qwen36_hiring.py`, `src/steering_checkpoint.py`, `jobs/ccu/run_qwen36_hiring.sh`
- **Inputs:** `config/qwen36_35b_a3b.yaml`, `data/processed/concept_vectors_qwen36_35b_a3b/`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/ratings/names/df_all.csv`
- **Outputs:** `results/tables/hiring_audit_qwen36_35b_a3b.csv`, `results/logs/hiring_probe_vs_human_qwen36_35b_a3b.json`

## Result

The native-HF audit completed and validated all 282 rated names in 193.7 seconds on one NVIDIA H100 80GB HBM3. Peak allocated VRAM was 65.46 GiB. Exact revision resolution, passive-hook parity, native hidden-state parity, single-device placement, text-only execution, and no-TransformerLens gates all passed.

Model projections aligned positively with human ratings. Warmth had Spearman rho=0.2109 (p=0.00036), while competence had rho=0.1313 (p=0.0275). Callback margin correlated with the model warmth projection at rho=0.3444 and with model competence at rho=0.1968. Corresponding correlations with human warmth and competence were rho=0.2574 and 0.1528.

Relative to the 27B audit, the 35B-A3B model showed slightly stronger warmth alignment and weaker competence alignment. These are observational name-level associations; causal interpretation depends on the separate steering sequence.
