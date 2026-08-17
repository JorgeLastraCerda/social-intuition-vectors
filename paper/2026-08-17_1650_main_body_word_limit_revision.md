# Main-Body Word-Limit Revision

- **Produced:** 2026-08-17 16:50 CEST
- **Model:** Nine Gemma, Qwen, and Llama checkpoints
- **Scope:** Manuscript restructuring for an 8,000-word main-body limit that includes text, figures, and tables but excludes references and supplementary materials
- **Status:** Superseded by `paper/2026-08-17_1723_main_body_rebuild_from_long.md`

## Artifacts

- **Inputs:** `paper/paper/Ulu_Lastra_long.tex`, `paper/paper/Ulu_Lastra_long.pdf`, `results/tables/probe_validation_9model.tex`, `results/tables/probe_human_correlation_9model.tex`, `results/tables/hiring_steering_slopes_9model.tex`, `results/tables/hiring_disparity_gaps_9model.tex`, `results/tables/concept_direction_specificity.tex`, `results/tables/concept_signal_vs_control_9model.tex`
- **Outputs:** `paper/paper/Ulu_Lastra.tex`, `paper/paper/Ulu_Lastra.pdf`, `paper/paper/Ulu_Lastra_long.tex`, `paper/paper/Ulu_Lastra_long.pdf`
- **Figures:** `paper/figures/background_concept_geometry.{png,pdf}`, `paper/figures/paper_figure1_axis_arrows.{png,pdf}`, `paper/figures/fig14_dense_steering_normalized.{png,pdf}`, `paper/figures/paper_figure4_hiring_bidirectional_examples.{png,pdf,py}`, `paper/figures/background_emotion_vector.{png,pdf}`, `paper/figures/paper_figure2_layer_emergence.{png,pdf}`, `paper/figures/fig6_cross_model_agreement.{png,pdf}`

## Summary

This revision is retained for provenance but is no longer the active
submission draft. The author judged its 5,621-word main body too short, so the
manuscript was rebuilt directly from the immutable `_long` archive rather
than expanded from this condensed version. The replacement is documented in
`paper/2026-08-17_1723_main_body_rebuild_from_long.md`.

The manuscript now fits the supervisor's strict interpretation of the limit. A conservative one-off count of every extractable word on the pages before References, including the title block, code link, figure text, table text, captions, acknowledgements, and page numbers, falls from 12,137 in the archived long version to 5,621 in the active version. This leaves a 2,379-word buffer below 8,000. The count deliberately over-includes front matter rather than relying on a narrower prose-only convention.

The main narrative now follows one causal chain: extraction, validation, human alignment, intervention, and hiring disparity. It retains four figures and four tables. Three secondary figures, three secondary tables, detailed extraction and steering protocols, and the full robustness boundary discussion moved to the Supplementary Materials. The main Limitations passage now contains four core constraints, while the appendix preserves the model-specific qualifications.

## Before and After

| Measure | Archived `_long` version | Active version | Change |
|---|---:|---:|---:|
| Conservative visible-word count before References | 12,137 | 5,621 | -6,516 (-53.7%) |
| Main-body pages before References | 18 | 9 | -9 |
| Main figures | 7 | 4 | -3 |
| Main tables | 7 | 4 | -3 |
| Total PDF pages | 35 | 32 | -3 |

The archived source is byte-identical to the pre-revision Git version, and the archived PDF is the corresponding 35-page baseline. These files allow sentence-level or rendered comparisons without reconstructing the earlier draft.

## Main-Text Package

The four retained figures are the residual-stream concept geometry schematic, the nine-model axis geometry, normalized concept steerability, and matched bidirectional hiring transitions. The four retained tables report probe validation, probe-to-human correlation, hiring-steering slopes, and model-versus-human demographic gaps. Together they cover the paper's causal argument without requiring a secondary result to carry the main story.

The emotion-vector schematic, layer-emergence plot, and cross-model agreement plot now appear as Figures S.1 to S.3. Model inventory, direction-specificity, and random-control details moved into the appendix table sequence. Supplementary tables render in strict physical order from S.1 through S.18.

## Verification

- `pdflatex`, `bibtex`, and two final `pdflatex` passes completed with no overfull boxes, unresolved citations or references, LaTeX errors, or float-size warnings.
- The active PDF has 32 pages. References occupy pages 10 and 11, and Supplementary Materials begin on page 12.
- All main-text and appendix pages were rendered to PNG and inspected for clipping, overlap, table legibility, caption placement, page numbering, and section transitions.
- `tests/test_paper_table_builders.py` passes 18 of 18 tests.
- The full project suite passes 113 of 115 tests. The two unchanged failures are in `tests/test_hiring_r4.py`: one fixture passes a CSV path where `load_and_join` expects the raw-data directory structure, and one fixture omits the `study` column required by `group_statistics`. Neither failure touches the manuscript revision.
- The edited prose was reread for the repository's anti-formulaic rules. No em-dash punctuation was added to authored prose, no adjacent paragraphs share an opener frame, and no signal-only transition remains. The original em-dash punctuation inside the reproduced generation prompt is retained as research material via `\textemdash{}`.

## Decision

The active manuscript should be treated as the submission version. The `_long` pair is an archival comparison copy, not an alternate source for further edits. Future additions should go to the appendix unless they replace, rather than expand, one of the eight main visual or tabular elements.
