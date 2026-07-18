# Gemma 4 12B CCU Calibrated Steering

- **Produced:** 2026-07-18 23:14 Europe/Berlin
- **Model:** `google/gemma-4-12B-it`
- **Scope:** Primary H100 replication of SD-matched additive and norm-preserving steering with target, cross-axis, and 99 random directions
- **Status:** Complete and validated

## Artifacts

- **Scripts:** `src/dense_steering.py`, `src/steering_calibration.py`, `src/validate_calibrated_steering.py`, `jobs/ccu/run_gemma4_calibrated.sh`
- **Inputs:** `config/gemma4_12b.yaml`, `data/stimuli/concept_stories.jsonl`, `data/processed/concept_vectors_gemma4_12b/`
- **Outputs:** `results/tables/steering_dense_raw_gemma4_12b_calibrated_ccu_h100.csv`, `results/tables/steering_dense_gemma4_12b_calibrated_ccu_h100.csv`, `results/tables/steering_dense_null_gemma4_12b_calibrated_ccu_h100.csv`, `results/logs/steering_dense_gemma4_12b_calibrated_ccu_h100.json`, `results/logs/smoke_gemma4_12b_calibrated_ccu_h100.json`

## Result

The primary CCU run completed all 2,022 checkpoint units and passed the structural validator with 40,440 raw rows, 2,020 summary rows, and eight null-comparison rows. It used the pinned revision `12ace6d648d72bd41519e140f1185f34d38c7e3d`, seed 20260527, layer 31, native chat formatting, and one NVIDIA H100 80GB HBM3. Peak allocated VRAM was 22.67 GiB.

At strength +0.10, additive target-axis effects were +1.843 for warmth judgments and +2.657 for competence judgments. The corresponding paired-topic target-minus-random estimates were +1.546 (95% CI [1.422, 1.670]) and +2.447 (95% CI [2.360, 2.537]). Cross-axis effects were also large, so the result supports causal steerability but not axis specificity.

Norm-preserving steering closely reproduced the additive effects. The maximum observed relative norm drift was 0.005823, below the documented 0.01 bfloat16 tolerance. The null comparisons remain descriptive under the predeclared scientific gate.
