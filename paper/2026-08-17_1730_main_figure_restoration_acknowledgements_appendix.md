# Main-Figure Restoration and Appendix Acknowledgements

- **Produced:** 2026-08-17 17:30 CEST
- **Model:** Nine Gemma, Qwen, and Llama checkpoints
- **Scope:** Restore two required explanatory figures to the main body while excluding Acknowledgements from the 8,000-word main-body count
- **Status:** Complete

## Artifacts

- **Scripts:** `src/build_paper_probe_tables.py`, `src/build_paper_mediation_table.py`, `paper/figures/generate_figures.py`, `paper/figures/paper_figure4_hiring_bidirectional_examples.py`
- **Inputs:** `paper/paper/Ulu_Lastra_long.tex`, `paper/paper/Ulu_Lastra_long.pdf`, `paper/paper/Ulu_Lastra.tex`, `paper/figures/background_emotion_vector.pdf`, `paper/figures/fig6_cross_model_agreement.pdf`, `results/tables/probe_validation_9model.tex`, `results/tables/concept_signal_vs_control_9model.tex`, `results/tables/probe_human_correlation_9model.tex`, `results/tables/hiring_steering_slopes_9model.tex`, `results/tables/hiring_disparity_gaps_9model.tex`
- **Outputs:** `paper/paper/Ulu_Lastra.tex`, `paper/paper/Ulu_Lastra.pdf`
- **Figures:** `paper/figures/background_emotion_vector.pdf`, `paper/figures/background_concept_geometry.pdf`, `paper/figures/paper_figure1_axis_arrows.pdf`, `paper/figures/paper_figure2_layer_emergence.pdf`, `paper/figures/fig6_cross_model_agreement.pdf`, `paper/figures/fig14_dense_steering_normalized.pdf`, `paper/figures/paper_figure4_hiring_bidirectional_examples.pdf`

## Summary

Acknowledgements, the SCCKN and CCU credit, and the AI-assistance disclosure
now appear after the bibliography at the beginning of Supplementary
Materials. The page-based main-body count therefore excludes them. The
emotion-vector schematic returns to the Background discussion of the
read-and-steer method, and the cross-model agreement plot follows the Results
paragraph that reports all 36 model-pair correlations.

The main package now contains seven figures and five tables. The conservative
PDF-visible count is 7,867 words on pages 1 to 12, including title material,
figure-internal labels, captions, tables, and page numbers. Plain extraction
gives 7,785 words. No compensating prose cut was required because the restored
visual content remains below the strict 8,000-word ceiling.

## Numbering and Layout

Main figures resolve as Figures 1 through 7: emotion vectors, residual-stream
concept geometry, nine-model axis geometry, layer emergence, cross-model
agreement, normalized steerability, and matched bidirectional hiring
examples. Main tables remain Tables 1 through 5. The appendix begins with
Acknowledgements and Secondary Tables; model inventory and direction
specificity resolve as Tables S.1 and S.2.

The final PDF remains 32 pages. Main content occupies pages 1 to 12,
References pages 13 to 14, and Supplementary Materials begins on page 15.
Acknowledgements is visibly located on page 15. The `_long` source and PDF
remain byte-identical to the pre-revision Git versions.

## Verification

- `pdflatex`, `bibtex`, and two final `pdflatex` passes completed without overfull boxes, unresolved citations or references, LaTeX errors, or float-size warnings.
- All 32 pages were rendered to PNG and inspected. Detailed checks of the emotion-vector page, cross-model agreement page, and appendix opening found no clipping, overlap, unreadable labels, or broken section transitions.
- `tests/test_paper_table_builders.py` passes 18 of 18 tests.
- The project suite passes 113 of 115 tests. The unchanged failures are the two pre-existing `tests/test_hiring_r4.py` fixtures: one supplies a CSV where the loader expects a raw-data directory, and one omits the required `study` column.
- `git diff --check` passes, and the active source remains LF-only.

## Writing Review

The moved text is unchanged, and both restored captions come from the
archived long manuscript. Re-reading the affected Background, Results, and
appendix transitions found no em-dash punctuation in authored main prose,
repeated adjacent paragraph opener, threefold mechanical sentence frame, or
signal-only transition.

## Decision

Treat the seven-figure, five-table version as the active submission draft.
Count only the pages before References for the supervisor's main-body limit;
Acknowledgements and the AI-assistance disclosure are supplementary material
under this layout.
