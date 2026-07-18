# Qwen3.6-35B-A3B Stage 2: Direction Reconstruction and Strict Transfer

- **Produced:** 2026-07-18 14:54 Europe/Berlin
- **Model:** Qwen/Qwen3.6-35B-A3B, revision `995ad96eacd98c81ed38be0c5b274b04031597b0`
- **Scope:** Fold-internal direction reconstruction and strict bidirectional cross-axis topic transfer
- **Status:** Complete; canonical Stage 2 artifacts upgraded and validated

## Artifacts

- **Scripts:** `src/qwen36_pipeline.py`, `src/upgrade_qwen36_stage2.py`, `src/validate_probes.py`, `src/validate_qwen36_stage.py`
- **Inputs:** `config/qwen36_35b_a3b.yaml`, `data/stimuli/concept_stories.jsonl`, `data/processed/concept_vectors_qwen36_35b_a3b/`
- **Outputs:** `results/tables/probe_metrics_qwen36_35b_a3b.csv`, `results/logs/validate_probes_qwen36_35b_a3b.json`, `results/logs/qwen36_35b_a3b_stage2_strict_audit.json`

## Summary

Both Stage 1 mean-difference directions achieved perfect topic-held-out accuracy when rebuilt independently within each fold. Strict cross-axis transfer was asymmetric: warmth transferred to competence at 0.99, while competence transferred to warmth at 0.93.

## Results

| Test | Mean accuracy | SD | Fold accuracies |
|---|---:|---:|---|
| Warmth direction, held-out warmth topics | 1.00 | 0.00 | 1.00, 1.00, 1.00, 1.00, 1.00 |
| Competence direction, held-out competence topics | 1.00 | 0.00 | 1.00, 1.00, 1.00, 1.00, 1.00 |
| Warmth trained, competence tested | 0.99 | 0.02 | 0.95, 1.00, 1.00, 1.00, 1.00 |
| Competence trained, warmth tested | 0.93 | 0.04 | 0.85, 0.95, 0.95, 0.95, 0.95 |

The prior target-calibrated cross-axis scores were 1.00 for warmth on competence and 0.97 for competence on warmth. The strict competence-to-warmth result is four percentage points lower because the classifier is not allowed to recalibrate on warmth examples. It is nevertheless far above chance and remains evidence of substantial shared evaluative signal.

## Integrity and validation

The upgrade checked the pinned revision, seed `20260527`, and corpus SHA-256 against Stage 1. All legacy Stage 2 values were preserved, and only the direction-reconstruction, calibrated-alias, and strict-transfer fields were added. The strengthened Stage 2 validator passed, while the Stage 2 versus Stage 3 audit continued to show zero differences at `1e-6` for both effect sizes and axis cosine.

The accepted computation used NumPy 2.5.1 and scikit-learn 1.9.0. A local NumPy 2.3.0 build linked to Apple Accelerate produced dot-product overflow warnings and unstable transfer scores; the write audit prevented those provisional results from replacing the legacy metrics, and the strict extension was regenerated in the clean environment.

## Interpretation

The MoE checkpoint encodes both target contrasts in reproducible directions, but strict transfer shows that the axes are not independent. The asymmetry suggests somewhat more axis-specific information in the competence-to-warmth direction than in the reverse direction. This is a relative distinction, not construct separation, because 0.93 transfer remains high.
