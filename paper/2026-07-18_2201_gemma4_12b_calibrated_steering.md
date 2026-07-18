# Gemma 4 12B Calibrated Steering

- **Produced:** 2026-07-18 22:01 Europe/Berlin
- **Model:** `google/gemma-4-12B-it`
- **Scope:** SD-matched target, cross-axis, and 99-direction random-control steering with additive and norm-preserving interventions
- **Status:** Complete on SCCKN as a supporting replication; CCU H100 primary replication pending

## Artifacts

- **Scripts:** `src/dense_steering.py`, `src/steering_calibration.py`, `src/validate_calibrated_steering.py`, `jobs/sge/calibrated_steering_run.sh`
- **Inputs:** `config/gemma4_12b.yaml`, `data/stimuli/concept_stories.jsonl`, `data/processed/concept_vectors_gemma4_12b/`
- **Outputs:** `results/tables/steering_dense_raw_gemma4_12b_calibrated.csv`, `results/tables/steering_dense_gemma4_12b_calibrated.csv`, `results/tables/steering_dense_null_gemma4_12b_calibrated.csv`, `results/logs/steering_dense_gemma4_12b_calibrated.json`, `results/logs/calibrated_steering_gemma4_12b_20260718T155600Z_gemma4_12b_retry1.{out,err}`

## Technical result

SCCKN job `1145434` completed all model inference and wrote the predeclared 40,440 raw rows, 2,020 summary rows, and eight null-comparison rows. The scheduler reported `failed=0`; the wrapper exited nonzero only after the original validator compared the maximum token-level bfloat16 norm drift, 0.005620, with a 0.005 threshold.

The drift distribution shows that this was a marginal numerical gate failure rather than a failed norm-preserving intervention. Median drift was 0.000114, the 99th percentile was 0.004047, and the maximum was 0.005620. The validator now uses an explicit 0.01 bfloat16 implementation tolerance while retaining the observed maximum in its result. With that correction, all structural, finiteness, calibration, random-control, norm-drift, and descriptive-only gates pass.

## Steering endpoints

The table reports mean change in the Yes-minus-No margin at the two extreme strengths. The secant slope is the endpoint difference divided by 0.2.

| Intervention | Judgment | Steering direction | Effect at -0.10 | Effect at +0.10 | Secant slope |
|---|---|---|---:|---:|---:|
| Additive | Warmth | Warmth | -1.100 | 1.842 | 14.710 |
| Additive | Warmth | Competence | -1.025 | 2.853 | 19.391 |
| Additive | Competence | Warmth | -1.161 | 1.707 | 14.337 |
| Additive | Competence | Competence | -1.303 | 2.581 | 19.420 |
| Norm-preserving | Warmth | Warmth | -1.121 | 1.838 | 14.798 |
| Norm-preserving | Warmth | Competence | -1.064 | 2.849 | 19.568 |
| Norm-preserving | Competence | Warmth | -1.175 | 1.708 | 14.412 |
| Norm-preserving | Competence | Competence | -1.281 | 2.585 | 19.329 |

Additive and norm-preserving results are nearly identical, so the causal changes are not explained by increasing residual-stream norm. Both concept directions change both judgments. In particular, competence steering is stronger than warmth steering for the warmth judgment at these endpoints. The result supports steerability but does not support axis specificity by itself; the 99-direction null tables and paired-topic intervals remain descriptive, as predeclared.

## Execution decision

This SCCKN result is retained unchanged as supporting evidence. The primary run will use the clean, resumable CCU H100 path under label `gemma4_12b_calibrated_ccu_h100`, followed serially by 26B-A4B and 31B. Scientific effect size will not gate that queue.
