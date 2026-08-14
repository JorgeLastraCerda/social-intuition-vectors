# Manuscript Table and Figure Revision

- **Produced:** 2026-08-13 18:16 Europe/Berlin
- **Model:** Nine Gemma, Llama, and Qwen checkpoints
- **Scope:** Jorge's 2026-08-10 to 2026-08-13 manuscript handover items
- **Status:** Corrected 2026-08-14; pooled-SD revision complete and layout review in progress

## Artifacts

- **Scripts:** `src/build_paper_probe_tables.py`, `src/build_paper_mediation_table.py`, `paper/figures/_steering_transition_flow_common.py`, `paper/figures/fig6_cross_model_agreement.py`, `paper/figures/generate_figures.py`, `paper/figures/background_emotion_vector.py`, `paper/figures/background_concept_geometry.py`, `paper/figures/paper_figure4_hiring_bidirectional_examples.py`
- **Inputs:** `results/logs/hiring_probe_vs_human_*.json`, `results/logs/validate_probes_*.json`, `results/logs/split_half_stability_*.json`, `results/logs/hiring_mediation_*.json`, `results/tables/hiring_audit_*.csv`, `results/tables/hiring_steering_*.csv`, `results/tables/hiring_steering_calibrated_*.csv`, `results/tables/hiring_disparity_*.csv`, `results/tables/cross_model_agreement_9model.csv`, `results/tables/gemma_scope_*.csv`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`
- **Outputs:** `results/tables/hiring_disparity_gaps_9model.tex`, `results/tables/hiring_disparity_marginal_9model.tex`, `results/tables/hiring_disparity_race_gender_9model.tex`, `results/tables/probe_validation_9model.tex`, `results/tables/probe_human_correlation_9model.tex`, `results/tables/concept_direction_specificity.tex`, `results/tables/concept_signal_vs_control_9model.tex`, `results/tables/concept_saturation.tex`, `results/tables/hiring_steering_slopes_9model.tex`, `results/tables/hiring_steering_transition_summary_9model.{csv,tex}`, `results/tables/mediation_9model.tex`, `paper/paper/Ulu_Lastra.{tex,pdf}`, `paper/paper/references.bib`
- **Figures:** `paper/figures/background_emotion_vector.{png,pdf}`, `paper/figures/background_concept_geometry.{png,pdf}`, `paper/figures/paper_figure1_axis_arrows.{png,pdf}`, `paper/figures/paper_figure2_layer_emergence.{png,pdf}`, `paper/figures/fig6_cross_model_agreement.{png,pdf}`, `paper/figures/fig14_dense_steering_normalized.{png,pdf}`, `paper/figures/paper_figure4_hiring_bidirectional_examples.{png,pdf}`

## Summary

The manuscript now implements the table, figure, prose, citation, and cleanup requests recorded in Jorge's handover notes. The main Results section uses compact single-column tables where the content remains legible, replaces the stale agreement heatmap with a nine-model dot plot, and summarizes demographic disparities in a new nine-row table. Detailed marginal and crossed group levels remain available in the Supplementary Materials. The internal pending-updates tracker was removed.

## Findings Preserved in the Revision

- Twelve of eighteen target-direction effects fall outside their reported random-control range. None of the six Gemma-4 model-axis rows does, so concept movement alone does not establish direction specificity for that family.
- Only seven of eighteen broad-grid hiring responses have $R^2 \geq 0.8$. Eight fitted slopes disagree in sign with the corresponding $\alpha=+0.50$ endpoint.
- Qwen3-14B competence is range-dependent: its broad-grid slope is slightly negative, while the local $-0.10$ to $+0.10$ change is positive and its $+0.10$ endpoint is only $+0.010$.
- Seven models favor Black-signaling names and two favor White-signaling names after pooled within-group standardization. Eight models favor female-signaling names, opposite to the pooled human gender reference. On the common scale, the human race reference is small at $d=+0.15$ and the human gender reference is $d=-0.47$.
- Local transition summaries now use $\alpha=+0.10$ consistently. The Llama example remains No after both warmth and competence steering; the Qwen3 example remains Yes after both interventions.

## Manuscript and Layout Decisions

Model callback gaps and the human benchmark are reported as standardized mean differences using the respective outcome's pooled within-group SD on the exact matched-name population. The mediation table stays as a full-page Supplementary Materials table. Two related-work records were integrated into the argument rather than listed without context. All active figures were regenerated through the repository scripts and shared serif styling. The 2026-08-13 build had 35 pages and no overfull boxes, undefined references, unresolved citations, or float-too-large warnings; a new layout pass is under review.

## Verification

`tests/test_paper_table_builders.py` covers gap units and row count, control-range verdicts, slope-sign disagreements, local transition values, and nonduplicated human blocks in the pivoted appendix tables. The manuscript was rebuilt with `latexmk`, checked textually with `pdftotext`, and visually inspected from rendered page images, including the Results tables and figures and the large appendix tables.
