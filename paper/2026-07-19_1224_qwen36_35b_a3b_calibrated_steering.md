# Qwen3.6 35B-A3B Calibrated Steering

- **Produced:** 2026-07-19 12:24 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-35B-A3B`
- **Scope:** SD-matched additive and norm-preserving steering with target, cross-axis, and 99 random directions
- **Status:** Complete and validated on SCCKN RTX PRO 6000

## Artifacts

- **Scripts:** `src/qwen36_calibrated_steering.py`, `src/steering_calibration.py`, `src/steering_checkpoint.py`, `src/validate_calibrated_steering.py`, `jobs/sge/calibrated_steering_run.sh`
- **Inputs:** `config/qwen36_35b_a3b.yaml`, `data/stimuli/concept_stories.jsonl`, `data/processed/concept_vectors_qwen36_35b_a3b/`, `results/logs/calibrated_steering_submission_qwen36_35b_a3b_20260719T0949Z_qwen36_35b_a3b_resumable.json`
- **Outputs:** `results/tables/steering_dense_raw_qwen36_35b_a3b_calibrated_scckn_rtx6000.csv`, `results/tables/steering_dense_qwen36_35b_a3b_calibrated_scckn_rtx6000.csv`, `results/tables/steering_dense_null_qwen36_35b_a3b_calibrated_scckn_rtx6000.csv`, `results/logs/steering_dense_qwen36_35b_a3b_calibrated_scckn_rtx6000.json`, `results/logs/calibrated_steering_qwen36_35b_a3b_20260719T0949Z_qwen36_35b_a3b_resumable.out`, `results/logs/calibrated_steering_qwen36_35b_a3b_20260719T0949Z_qwen36_35b_a3b_resumable.err`

## Result

The resumable MoE run passed validation with 40,440 raw rows, 2,020 summary rows, and eight null-comparison rows. It used the pinned revision, seed 20260527, probe layer 26, and one NVIDIA RTX PRO 6000 Blackwell Server Edition. Execution took 6,328.4 seconds, peak allocated VRAM was 65.52 GiB, and maximum norm-preserving drift was 0.005176 under the 0.01 bfloat16 tolerance. Passive-hook logits and hidden states agreed exactly, no vision forwards occurred, and TransformerLens was not installed or imported.

At +0.10 additive strength, matched warmth steering changed the warmth margin by +0.822 and matched competence steering changed the competence margin by +1.450. Both target endpoints exceeded all 99 SD-matched random controls in signed and absolute percentile. Their target-minus-random paired-topic estimates were +0.816 (95% CI [+0.619, +1.031]) and +1.469 (95% CI [+1.294, +1.675]), respectively. Norm-preserving results were nearly unchanged at +0.813 for warmth and +1.425 for competence, with paired-topic intervals excluding zero.

Cross-axis responses were similarly large. Competence steering changed warmth by +1.000, and warmth steering changed competence by +1.325 under additive intervention. This model strongly distinguishes the learned subspace from SD-matched random directions, but it does not provide a clean axis-specific causal decomposition. The scientific gate remains descriptive only.

The authors acknowledge support by the local computing resources through the core facility SCCKN.
