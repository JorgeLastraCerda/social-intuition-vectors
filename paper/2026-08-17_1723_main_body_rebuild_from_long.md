# Main-Body Rebuild from the Long Archive

- **Produced:** 2026-08-17 17:23 CEST
- **Model:** Nine Gemma, Qwen, and Llama checkpoints
- **Scope:** Rebuild the active manuscript from the archived long source for a strict 8,000-word main-body limit that includes text, figures, and tables but excludes references and supplementary materials
- **Status:** Superseded by `paper/2026-08-17_1730_main_figure_restoration_acknowledgements_appendix.md`

## Artifacts

- **Scripts:** `src/build_paper_probe_tables.py`, `src/build_paper_mediation_table.py`, `paper/figures/generate_figures.py`, `paper/figures/paper_figure4_hiring_bidirectional_examples.py`
- **Inputs:** `paper/paper/Ulu_Lastra_long.tex`, `paper/paper/Ulu_Lastra_long.pdf`, `results/tables/probe_validation_9model.tex`, `results/tables/concept_signal_vs_control_9model.tex`, `results/tables/probe_human_correlation_9model.tex`, `results/tables/hiring_steering_slopes_9model.tex`, `results/tables/hiring_disparity_gaps_9model.tex`, `results/tables/concept_direction_specificity.tex`
- **Outputs:** `paper/paper/Ulu_Lastra.tex`, `paper/paper/Ulu_Lastra.pdf`
- **Figures:** `paper/figures/background_concept_geometry.pdf`, `paper/figures/paper_figure1_axis_arrows.pdf`, `paper/figures/paper_figure2_layer_emergence.pdf`, `paper/figures/fig14_dense_steering_normalized.pdf`, `paper/figures/paper_figure4_hiring_bidirectional_examples.pdf`, `paper/figures/background_emotion_vector.pdf`, `paper/figures/fig6_cross_model_agreement.pdf`

## Summary

This report records the direct rebuild from `_long`, but its five-figure main
package is no longer active. The author subsequently restored the
emotion-vector and cross-model agreement figures to the main body and moved
Acknowledgements to the appendix. The resulting 7,867-word version is
documented in
`paper/2026-08-17_1730_main_figure_restoration_acknowledgements_appendix.md`.

The active manuscript was reset to `Ulu_Lastra_long.tex` before editing. No
sentence from the superseded 5,621-word draft was used as the expansion base.
The rebuilt main body contains 7,746 conservatively counted visible words on
pages 1 to 12, including the title block, figure-internal text, captions,
tables, acknowledgements, and page numbers. Plain PDF extraction gives 7,666
words. Both measures remain below 8,000, and the conservative measure falls
inside the planned 7,700 to 7,900 working band.

The main package now contains five figures and five tables. Compared with the
superseded revision, the layer-emergence figure and the nine-model
signal-versus-control table return to the main body. Methods recovers the
cross-model extraction contract, probe construction, steering calibration,
hiring intervention, and benchmark definitions. Results retains enough
model-level detail to support the heterogeneous causal and human-alignment
claims, while Discussion preserves interpretation, four core limitation
classes, future work, and the conclusion.

## Main Package

The five main figures are residual-stream concept geometry, nine-model axis
geometry, layer-wise emergence, normalized concept steerability, and matched
bidirectional hiring examples. The five main tables report probe validation,
target signal versus random control, probe-to-human alignment,
hiring-steering slopes, and model-versus-human callback disparities.

The emotion-vector precursor, cross-model agreement distribution, model
inventory, six-direction specificity comparison, extended method details,
and full limitation inventory appear in the Supplementary Materials. Main
figures and tables are numbered 1 through 5; moved items begin at S.1. The
archived `_long` source and PDF remain byte-identical to the pre-revision Git
versions.

## Verification

- A full `pdflatex`, `bibtex`, and two-pass `pdflatex` build produced a 32-page PDF. References are on pages 13 and 14; Supplementary Materials begins on page 15.
- The final log has no overfull boxes, unresolved citations or references, LaTeX errors, or float-size warnings.
- All 32 pages were rendered to PNG. The complete contact sheet and detailed checks of the balanced final main page and moved appendix visuals found no clipping, overlap, illegible table text, or broken section transitions.
- `tests/test_paper_table_builders.py` passes 18 of 18 tests.
- The full project suite passes 113 of 115 tests. The two unchanged failures are in `tests/test_hiring_r4.py`: one fixture supplies a CSV where the loader expects the raw-data directory tree, and one omits the `study` column required by `group_statistics`.
- `git diff --check` passes. Main-body numbering resolves to Figures 1 to 5 and Tables 1 to 5; moved visual and table items resolve from S.1 onward.

## Writing Review

The revised main prose was reread after the final render. It uses American
spelling, contains no em-dash punctuation in authored prose, avoids adjacent
paragraphs with the same opener frame, and contains no threefold mechanical
sentence frame or signal-only transition. The generation prompt in the
appendix remains reproduced research material.

## Decision

Treat `paper/paper/Ulu_Lastra.tex` and `paper/paper/Ulu_Lastra.pdf` as the
active submission pair. Keep the `_long` pair immutable as the recoverable
pre-limit baseline. The superseded 5,621-word report remains available only
as a record of the first reduction pass.
