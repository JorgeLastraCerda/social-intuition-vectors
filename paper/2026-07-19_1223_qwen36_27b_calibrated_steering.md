# Qwen3.6 27B Calibrated Steering

- **Produced:** 2026-07-19 12:23 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-27B`
- **Scope:** SD-matched additive and norm-preserving steering with target, cross-axis, and 99 random directions
- **Status:** Complete and validated on SCCKN RTX PRO 6000

## Artifacts

- **Scripts:** `src/qwen36_calibrated_steering.py`, `src/steering_calibration.py`, `src/steering_checkpoint.py`, `src/validate_calibrated_steering.py`, `jobs/sge/calibrated_steering_run.sh`
- **Inputs:** `config/qwen36_27b.yaml`, `data/stimuli/concept_stories.jsonl`, `data/processed/concept_vectors_qwen36_27b/`, `results/logs/calibrated_steering_submission_qwen36_27b_20260719T0949Z_qwen36_27b_resumable.json`
- **Outputs:** `results/tables/steering_dense_raw_qwen36_27b_calibrated_topicfix_scckn_rtx6000.csv`, `results/tables/steering_dense_qwen36_27b_calibrated_topicfix_scckn_rtx6000.csv`, `results/tables/steering_dense_null_qwen36_27b_calibrated_topicfix_scckn_rtx6000.csv`, `results/logs/steering_dense_qwen36_27b_calibrated_topicfix_scckn_rtx6000.json`, `results/logs/calibrated_steering_qwen36_27b_20260719T0949Z_qwen36_27b_resumable.out`, `results/logs/calibrated_steering_qwen36_27b_20260719T0949Z_qwen36_27b_resumable.err`

## Result

The topic-corrected resumable run passed validation with 40,440 raw rows, 2,020 summary rows, and eight null-comparison rows. It used the pinned revision, seed 20260527, probe layer 42, and one NVIDIA RTX PRO 6000 Blackwell Server Edition. Execution took 5,158.5 seconds, peak allocated VRAM was 51.26 GiB, and maximum norm-preserving drift was 0.006003 under the 0.01 bfloat16 tolerance. Passive-hook logits and hidden states agreed exactly, no vision forwards occurred, and TransformerLens was not installed or imported.

At +0.10 additive strength, matched warmth steering changed the warmth margin by +0.694 and matched competence steering changed the competence margin by +0.306. Both target endpoints exceeded all 99 SD-matched random controls in signed and absolute percentile. Their target-minus-random paired-topic estimates were +0.681 (95% CI [+0.575, +0.788]) and +0.300 (95% CI [+0.163, +0.481]), respectively. Norm-preserving estimates remained positive at +0.669 for warmth and +0.269 for competence, with paired-topic intervals excluding zero.

Cross-axis interventions were also resolved: competence steering changed warmth by +0.219, while warmth steering changed competence by +0.638. The calibrated result therefore supports causal sensitivity beyond matched random directions, but not clean warmth-versus-competence specificity. The scientific gate remains descriptive only.

This artifact supersedes the rejected `qwen36_27b_calibrated` pilot for scientific use. The earlier files and `paper/2026-07-18_2201_qwen36_27b_calibrated_incomplete.md` remain preserved as the audit trail for the non-contiguous-topic selection bug.

The authors acknowledge support by the local computing resources through the core facility SCCKN.
