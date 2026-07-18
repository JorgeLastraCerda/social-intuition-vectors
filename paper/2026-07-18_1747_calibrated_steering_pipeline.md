# Calibrated Steering-Control Pipeline

- **Produced:** 2026-07-18 17:47 CEST
- **Models:** Gemma-3-12B-it, Gemma-4-12B-it, Qwen3.6-27B pilot; nine-model rollout after technical validation
- **Scope:** SD-matched random/cross-axis controls and norm-preserving intervention robustness
- **Status:** Implementation complete and locally validated; empirical pilots pending

## Artifacts

- **Scripts:** `src/steering_calibration.py`, `src/dense_steering.py`, `src/qwen36_calibrated_steering.py`, `src/validate_calibrated_steering.py`, `jobs/sge/calibrated_steering_run.sh`, `jobs/sge/submit_calibrated_steering_pilot.sh`
- **Inputs:** `data/stimuli/concept_stories.jsonl`, `data/processed/concept_vectors/`, `data/processed/concept_vectors_gemma4_12b/`, `data/processed/concept_vectors_qwen36_27b/`

## Motivation

The legacy random-direction control used the same Euclidean intervention length as the target concept direction. Direct diagnostics showed that this produced much larger standardized projection shifts for random directions because their natural activation variance was far smaller. Across the inspected models, target shifts were approximately 0.9 to 4.4 within-direction SDs, whereas legacy random shifts were approximately 12.9 to 116.7 SDs. The old random-control p values therefore do not establish specificity under a statistically comparable perturbation.

This correction follows the broader activation-steering principle that intervention magnitude and representation geometry must be interpreted together. Activation Addition defines additive residual interventions ([Turner et al., 2023](https://arxiv.org/abs/2308.10248)); recent work further emphasizes selective, geometry-aware steering and the limits of treating Euclidean direction length as a sufficient calibration ([Selective Steering, 2026](https://aclanthology.org/2026.findings-acl.529/); [A Geometric Account of Steering, 2025](https://openreview.net/forum?id=MVYKmq3Jhx)).

## Implemented calibration

Calibration uses training topics only. For a target axis with projection standard deviation `SD_target`, the original target intervention remains

`alpha_target = strength * mean_residual_norm`.

For a cross-axis or seeded random direction with projection standard deviation `SD_direction`, the calibrated intervention is

`alpha_direction = alpha_target * SD_direction / SD_target`.

Thus all directions within a judgment-axis condition receive the same standardized displacement, `alpha / SD_direction`, while the target intervention is unchanged by construction. Each output row records `direction_sd`, `alpha_absolute`, `standardized_shift`, `control_scale`, and `intervention`.

## Robustness and descriptive analysis

The primary condition is additive steering. A second condition rescales every changed token residual back to its original norm after adding the direction. The validator checks the resulting token-level norm drift, exact row counts, finite values, 99 random directions, matched standardized shifts, write-once outputs, and descriptive-only scientific status.

The null report contains signed and absolute endpoint percentiles, signed and absolute slope percentiles, random medians, target-minus-random-median contrasts, and topic-paired bootstrap confidence intervals. Decision-boundary flips are stored per story. Full-vocabulary entropy and KL were not added to this pilot because doing so would materially increase retained per-prompt state and is not needed to diagnose the control-scale mismatch; they remain an optional later diagnostic.

No scientific pass/fail threshold is used. Technical success triggers the predeclared nine-model rollout regardless of effect direction or magnitude. The 282-name hiring expansion remains paused until the user reviews the pilot calibration report.

## Execution design

The pilot consists of three independent, user-held Grid Engine jobs on RTX PRO 6000 GPUs:

1. Gemma-3-12B-it through TransformerLens.
2. Gemma-4-12B-it through TransformerLens Bridge with native chat formatting.
3. Qwen3.6-27B through native Hugging Face forward hooks without importing TransformerLens.

Each job produces 40,440 raw rows, 2,020 steering-summary rows, and eight descriptive-null rows from two judgment axes, two intervention types, two concept directions plus 99 random directions, five strengths, and 20 held-out stories per axis. Jobs do not use scheduler dependencies, do not pull Git state at runtime, and do not synchronize outputs from compute nodes.

## Local validation

The focused calibration suite passed seven tests, and the complete project test directory passed 81 tests. Python compilation, Ruff, formatting, shell syntax, and `git diff --check` passed. All three planned pilot output sets were absent under their write-once labels before submission preparation.
