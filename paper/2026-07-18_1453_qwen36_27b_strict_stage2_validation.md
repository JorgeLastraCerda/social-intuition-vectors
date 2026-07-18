# Qwen3.6-27B Stage 2: Direction Reconstruction and Strict Transfer

- **Produced:** 2026-07-18 14:53 Europe/Berlin
- **Model:** Qwen/Qwen3.6-27B, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- **Scope:** Fold-internal direction reconstruction and strict bidirectional cross-axis topic transfer
- **Status:** Complete; canonical Stage 2 artifacts upgraded and validated

## Artifacts

- **Scripts:** `src/qwen36_pipeline.py`, `src/upgrade_qwen36_stage2.py`, `src/validate_probes.py`, `src/validate_qwen36_stage.py`
- **Inputs:** `config/qwen36_27b.yaml`, `data/stimuli/concept_stories.jsonl`, `data/processed/concept_vectors_qwen36_27b/`
- **Outputs:** `results/tables/probe_metrics_qwen36_27b.csv`, `results/logs/validate_probes_qwen36_27b.json`, `results/logs/qwen36_27b_stage2_strict_audit.json`

## Summary

The exact Stage 1 mean-difference construction generalized perfectly when the direction, projection scaling, and decision boundary were rebuilt inside every topic fold. Strict transfer remained very high in both directions, showing that most target discrimination is shared across warmth and competence rather than axis-specific.

## Results

| Test | Mean accuracy | SD | Fold accuracies |
|---|---:|---:|---|
| Warmth direction, held-out warmth topics | 1.00 | 0.00 | 1.00, 1.00, 1.00, 1.00, 1.00 |
| Competence direction, held-out competence topics | 1.00 | 0.00 | 1.00, 1.00, 1.00, 1.00, 1.00 |
| Warmth trained, competence tested | 0.97 | 0.06 | 0.85, 1.00, 1.00, 1.00, 1.00 |
| Competence trained, warmth tested | 0.98 | 0.024 | 0.95, 0.95, 1.00, 1.00, 1.00 |

The older target-calibrated cross-axis scores were 0.99 for warmth on competence and 1.00 for competence on warmth. Those values remain in the JSON under compatibility names and explicit `*_calibrated_cv` aliases. The strict test is the appropriate zero-shot transfer result because its direction, scaler, and classifier are trained only on the source axis and training topics.

## Integrity and validation

The upgrade verified the pinned model revision, seed `20260527`, and Stage 1 stimulus SHA-256. It retained every pre-existing Stage 2 scalar and fold value, then added the new fields. The dedicated Stage 2 validator passed, and the Stage 2 versus Stage 3 audit retained zero differences for warmth d, competence d, and axis cosine at tolerance `1e-6`.

The final computation used NumPy 2.5.1 and scikit-learn 1.9.0 in the repository's isolated analysis environment. A system NumPy 2.3.0 build linked to Apple Accelerate emitted erroneous dot-product overflow warnings and changed fold scores; those provisional values were rejected and are not present in the canonical artifacts.

## Interpretation

The result strengthens target-axis generalization while weakening construct selectivity. Warmth and competence directions are reproducible across unseen topics, but either source axis almost perfectly classifies the other axis without target recalibration. Steering analyses therefore need paired target and non-target outcomes, strict cross-axis controls, and matched random directions.
