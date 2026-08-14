# Manuscript Value and Claim Consistency Corrections

- **Produced:** 2026-08-14 14:58 Europe/Berlin
- **Model:** Nine Gemma, Llama, and Qwen checkpoints
- **Scope:** Probe-depth, steering, human-alignment, ablation, validation-count, and callback-resolution manuscript corrections
- **Status:** Complete and locally verified

## Artifacts

- **Scripts:** `src/build_paper_probe_tables.py`, `tests/test_paper_table_builders.py`
- **Inputs:** `results/tables/concept_steerability_normalized_9model.csv`, `results/logs/hiring_probe_vs_human_*.json`, `results/tables/hiring_audit_*.csv`, `results/tables/gemma_scope_causality_gemma3_{12b,27b}.csv`
- **Outputs:** `results/tables/gemma_scope_ablation.tex`, `paper/paper/Ulu_Lastra.tex`, `paper/paper/Ulu_Lastra.pdf`
- **Figures:** `paper/figures/paper_figure2_layer_emergence.pdf`, `paper/figures/fig14_dense_steering_normalized.pdf`

## Corrections

The fixed probe depth of 0.66 is now described as a prespecified common reference rather than a model-specific optimum. The existing Limitations explanation of why this depth was selected remains unchanged. Gemma-4-26B-A4B and Gemma-4-31B competence endpoints are corrected to -0.002 and -0.001. Human-alignment prose now reports seven positive, significant competence correlations and two very small, nonsignificant negative correlations.

The Gemma Scope ablation interpretation now matches the original analysis and the signed table values. Removing shared features shrinks both concept gaps in Gemma-3-27B but increases both gaps in Gemma-3-12B, so shared-feature necessity is scale-specific. The appendix caption is generated from the corrected builder contract.

Methods now counts seven validation checks, consisting of five within-model checks and two cross-model analyses. The callback limitation no longer assigns a universal 0.125 grid to all checkpoints. It reports model-dependent reduced-precision resolution and updates the canonical 282-name audit summaries to Gemma-3-12B SD 0.15 with 8 unique values, Llama-3.1-8B SD 0.12 with 12, Gemma-3-27B SD 0.43 with 20, and Qwen3-14B SD 0.35 with 17.

## Verification

Twelve targeted tests pass. The manuscript builds to 34 pages with no overfull boxes, unresolved citations or references, float-too-large warnings, or `??` markers. Figures 1 through 7 remain in strict visual order on pages 3, 6, 8, 9, 10, 11, and 14. The changed Methods, Results, Limitations, and Appendix S.7 pages were rendered and inspected without clipping, overlap, or harmful reflow.
