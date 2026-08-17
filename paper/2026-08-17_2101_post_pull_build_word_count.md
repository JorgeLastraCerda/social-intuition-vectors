# Post-Pull Build and Word-Count Audit

- **Produced:** 2026-08-17 21:01 CEST
- **Model:** Nine Gemma, Qwen, and Llama checkpoints
- **Scope:** Synchronize the collaborator's latest manuscript references, rebuild the active PDF, and separate the main-body, figure, table, reference, and appendix word counts
- **Status:** Complete

## Artifacts

- **Inputs:** `paper/paper/Ulu_Lastra.tex`, `results/tables/probe_validation_9model.tex`, `results/tables/concept_signal_vs_control_9model.tex`, `results/tables/probe_human_correlation_9model.tex`, `results/tables/hiring_steering_slopes_9model.tex`, `results/tables/hiring_disparity_gaps_9model.tex`
- **Outputs:** `paper/paper/Ulu_Lastra.pdf`
- **Figures:** `paper/figures/background_emotion_vector.pdf`, `paper/figures/background_concept_geometry.pdf`, `paper/figures/paper_figure1_axis_arrows.pdf`, `paper/figures/paper_figure2_layer_emergence.pdf`, `paper/figures/fig6_cross_model_agreement.pdf`, `paper/figures/fig14_dense_steering_normalized.pdf`, `paper/figures/paper_figure4_hiring_bidirectional_examples.pdf`

## Synchronization and Build

The local `main` branch fast-forwarded to `origin/main` without conflicts. The
two incoming commits were `dba69c6` (fixed table links) and `bdf54c2` (fixed
one missing parenthesis). Together they add the Figure 1 cross-reference to
the emotion-vector discussion and replace two supplementary-section links
with direct links to Tables S.6 and S.7.

Three `pdflatex` passes rebuilt the active PDF at 32 pages. The final log has
no unresolved or multiply defined references, overfull boxes, LaTeX errors,
or rerun warnings. All pages were rendered to PNG; detailed checks of the two
changed-link pages, the main-text tables, the main-to-reference and
reference-to-supplement transitions, and the final continuation table found
no clipping, overlap, or broken navigation.

## Word Counts

Counts use Poppler's PDF-visible whitespace tokens. Main content is pages 1
through 12, References are pages 13 and 14, and Supplementary Materials are
pages 15 through 32. Figure and table regions were isolated from the PDF
coordinates, including captions, axis labels, legends, cell contents, and
numeric labels. The remaining main-page tokens form the normal-text category,
which also conservatively retains front matter, headings, equations, and page
numbers.

| Category | Visible words |
|---|---:|
| Normal main text, excluding figures and tables | 5,599 |
| Main figures 1 to 7 | 1,161 |
| Main tables 1 to 5 | 1,029 |
| Figures and tables combined | 2,190 |
| **Main-body total, including everything** | **7,789** |
| Remaining below 8,000 | 211 |
| References | 815 |
| Supplementary Materials, including Acknowledgements | 10,182 |
| References and Supplementary Materials combined | 10,997 |
| Full PDF | 18,786 |

The additive split sums exactly: 5,599 + 1,161 + 1,029 = 7,789. The project's
previously accepted conservative ledger was 7,867. The collaborator's update
adds four visible tokens, so the same deliberately overinclusive compliance
ledger is now 7,871, leaving 129 words below the strict ceiling. This second
number is retained as a safety check and is not mixed into the additive table,
whose categories all use one extraction method.

## Verification

- `tests/test_paper_table_builders.py` passes 20 of 20 tests.
- The full suite passes 115 of 117 tests. The two unchanged failures are the pre-existing `tests/test_hiring_r4.py` fixtures: one supplies a CSV where the loader expects a raw-data directory, and one omits the required `study` column.
- `paper/paper/Ulu_Lastra_long.tex` and `paper/paper/Ulu_Lastra_long.pdf` were not modified.
- Graphify was not used because the author identified its index as stale.

## Writing Review

No active-manuscript prose was authored in this audit. The incoming changes
only add or redirect cross-references, so the anti-formulaic prose check is
not applicable.
