# Gemma 4 26B-A4B Calibrated Steering

- **Produced:** 2026-07-18 23:38 Europe/Berlin
- **Model:** `google/gemma-4-26B-A4B-it`
- **Scope:** SD-matched additive and norm-preserving steering with target, cross-axis, and 99 random directions
- **Status:** Complete and validated on SCCKN RTX PRO 6000

## Artifacts

- **Scripts:** `src/dense_steering.py`, `src/steering_calibration.py`, `src/validate_calibrated_steering.py`, `jobs/sge/calibrated_steering_run.sh`
- **Inputs:** `config/gemma4_26b_a4b.yaml`, `data/stimuli/concept_stories.jsonl`, `data/processed/concept_vectors_gemma4_26b_a4b/`
- **Outputs:** `results/tables/steering_dense_raw_gemma4_26b_a4b_calibrated_scckn_rtx6000.csv`, `results/tables/steering_dense_gemma4_26b_a4b_calibrated_scckn_rtx6000.csv`, `results/tables/steering_dense_null_gemma4_26b_a4b_calibrated_scckn_rtx6000.csv`, `results/logs/steering_dense_gemma4_26b_a4b_calibrated_scckn_rtx6000.json`, `results/logs/calibrated_steering_gemma4_26b_a4b_20260718T204020Z.out`

## Result

The run passed validation with 40,440 raw rows, 2,020 summary rows, and eight null-comparison rows. It used the pinned revision, seed 20260527, layer 19, and one NVIDIA RTX PRO 6000 Blackwell Server Edition. Peak allocated VRAM was 48.48 GiB, and maximum norm-preserving drift was 0.005351 under the 0.01 bfloat16 tolerance.

Target-axis endpoint effects at +0.10 were small: +0.045 for additive warmth steering and -0.069 for additive competence steering. Paired-topic target-minus-random estimates were positive, at +0.861 (95% CI [0.733, 0.988]) for warmth and +0.559 (95% CI [0.444, 0.669]) for competence. Target endpoint absolute percentiles were only 0.04 and 0.06 among the 99 random controls, so the result demonstrates implementation-level steerability but weak direction specificity under this calibrated null.

Norm-preserving results closely followed additive results. The descriptive-only scientific gate is retained; no pass/fail claim is made from the random-control percentiles.
