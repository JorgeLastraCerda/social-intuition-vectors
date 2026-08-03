# Nine-Model Normalized Concept Steerability

- **Produced:** 2026-07-20 09:19 Europe/Berlin
- **Models:** Gemma-3-12B, Gemma-3-27B, Llama-3.1-8B, Qwen3-14B, Gemma-4-12B, Gemma-4-26B-A4B, Gemma-4-31B, Qwen3.6-27B, Qwen3.6-35B-A3B
- **Scope:** Additive target-direction concept steering at a common local coefficient grid
- **Status:** Complete

## Artifacts

- **Scripts:** `paper/figures/generate_figures.py`
- **Inputs:** `results/tables/steering_dense_gemma3_12b.csv`, `results/tables/steering_dense_gemma3_27b.csv`, `results/tables/steering_dense_llama31_8b.csv`, `results/tables/steering_dense_qwen3_14b.csv`, `results/tables/steering_dense_gemma4_12b_calibrated_ccu_h100.csv`, `results/tables/steering_dense_gemma4_26b_a4b_calibrated_scckn_rtx6000.csv`, `results/tables/steering_dense_gemma4_31b_calibrated_scckn_rtx6000.csv`, `results/tables/steering_dense_qwen36_27b_calibrated_topicfix_scckn_rtx6000.csv`, `results/tables/steering_dense_qwen36_35b_a3b_calibrated_scckn_rtx6000.csv`, and the five corresponding `steering_dense_raw_*` tables used to reconstruct baseline gaps
- **Outputs:** `results/tables/concept_steerability_normalized_9model.csv`, `paper/paper/Ulu_Lastra.pdf`
- **Figures:** `paper/figures/fig14_dense_steering_normalized.{png,pdf}`

## Summary

Figure 14 now compares additive target-direction steering across all nine completed checkpoints. Every curve uses the common local coefficient grid $\alpha \in \{-0.10,-0.05,0,+0.05,+0.10\}$, with coefficients expressed relative to the mean residual norm at the intervention layer. The legend reports the maximum positive coefficient used for each series.

The plotted quantity is the change in held-out Yes-versus-No concept margin divided by the same model-axis baseline high-versus-low concept gap. This scaling makes raw logit magnitudes less dominant in cross-model comparison. It does not convert the curves into a direction-specificity test.

## Positive endpoint results

| Model | Warmth | Competence |
|---|---:|---:|
| Gemma-3-12B | 0.236 | 0.141 |
| Gemma-3-27B | 0.040 | 0.009 |
| Llama-3.1-8B | 0.029 | 0.024 |
| Qwen3-14B | 0.122 | 0.104 |
| Gemma-4-12B | 0.046 | 0.069 |
| Gemma-4-26B-A4B | 0.001 | -0.002 |
| Gemma-4-31B | 0.008 | -0.001 |
| Qwen3.6-27B | 0.035 | 0.017 |
| Qwen3.6-35B-A3B | 0.043 | 0.090 |

Gemma-3-12B has the largest normalized endpoint on both axes. Qwen3-14B is second on both, while Qwen3.6-35B-A3B shows a comparatively strong competence response. Gemma-4-26B-A4B and Gemma-4-31B are nearly flat at the positive competence endpoint despite strong probe separation.

## Interpretation and limitations

The result separates representational probeability from efficient causal access through a particular mean-difference vector. A model can separate high and low stories strongly while responding weakly to the corresponding additive intervention.

Axis specificity requires separate evidence. The Qwen3.6 target directions exceed all 99 standard-deviation-matched random controls, but their cross-axis effects remain large. Gemma-4-26B-A4B and Gemma-4-31B do not provide comparable evidence for direction-specific control at the local endpoint. Figure 14 is therefore a descriptive target-sensitivity comparison and must be interpreted with the calibrated random and cross-axis controls.

## Data contract

- Exactly nine model labels, two axes, and five strengths are required, producing 90 derived rows.
- Only additive target-direction rows are included.
- Legacy baseline gaps are read from their summary tables. New calibrated runs reconstruct the same quantity from companion same-run raw baseline rows.
- The generator freezes all 18 normalized values at $\alpha=+0.10$ and fails if they drift beyond numerical tolerance.

## Verification

- The derived table contains 90 rows: nine models, two axes, and five strengths per cell.
- Every zero-strength row has an exactly zero normalized effect.
- Ruff and whitespace checks pass for the updated generator and repository diff.
- The 18-page manuscript builds successfully with no undefined references, overfull boxes, or compilation errors.
- Poppler renders of the standalone figure and manuscript page 12 show a readable nine-item legend, complete labels, and no clipping or overlap.
- All fonts in the standalone figure PDF are embedded.
