# Gemma 4 31B Calibrated Steering

- **Produced:** 2026-07-19 00:30 Europe/Berlin
- **Model:** `google/gemma-4-31B-it`
- **Scope:** SD-matched additive and norm-preserving steering with target, cross-axis, and 99 random directions
- **Status:** Complete and validated on SCCKN RTX PRO 6000

## Artifacts

- **Scripts:** `src/dense_steering.py`, `src/steering_calibration.py`, `src/validate_calibrated_steering.py`, `jobs/sge/calibrated_steering_run.sh`
- **Inputs:** `config/gemma4_31b.yaml`, `data/stimuli/concept_stories.jsonl`, `data/processed/concept_vectors_gemma4_31b/`
- **Outputs:** `results/tables/steering_dense_raw_gemma4_31b_calibrated_scckn_rtx6000.csv`, `results/tables/steering_dense_gemma4_31b_calibrated_scckn_rtx6000.csv`, `results/tables/steering_dense_null_gemma4_31b_calibrated_scckn_rtx6000.csv`, `results/logs/steering_dense_gemma4_31b_calibrated_scckn_rtx6000.json`, `results/logs/calibrated_steering_gemma4_31b_20260718T205911Z_retry2.out`

## Result

The checkpoint-resumed run passed validation with 40,440 raw rows, 2,020 summary rows, and eight null-comparison rows. It used the pinned revision, seed 20260527, layer 39, and one NVIDIA RTX PRO 6000 Blackwell Server Edition. Peak allocated VRAM was 58.69 GiB, and maximum norm-preserving drift was 0.006368 under the 0.01 bfloat16 tolerance.

At +0.10 additive strength, the warmth target changed the warmth margin by +0.438, whereas the competence target changed the competence margin by -0.057. Neither target exceeded its SD-matched random controls: target-minus-random paired-topic estimates were -0.501 (95% CI [-0.661, -0.343]) for warmth and -0.775 (95% CI [-0.874, -0.669]) for competence. Their endpoint absolute percentiles were 0.15 and 0.06, respectively.

Norm-preserving estimates were similar. The result validates the implementation and intervention geometry, but it does not support direction-specific causal control under this calibrated random-direction comparison. The scientific gate remains descriptive only.
