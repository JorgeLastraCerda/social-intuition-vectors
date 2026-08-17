# Table, Reference, and Caption Audit

- **Produced:** 2026-08-17 17:45 CEST
- **Model:** Nine Gemma, Qwen, and Llama checkpoints
- **Scope:** Full active-manuscript audit of table presence, numbering, references, appendix framing, caption position, and rendered order
- **Status:** Complete

## Artifacts

- **Scripts:** `src/build_paper_probe_tables.py`, `paper/figures/_steering_transition_flow_common.py`, `tests/test_paper_table_builders.py`
- **Inputs:** `paper/paper/Ulu_Lastra.tex`, `paper/paper/Ulu_Lastra.aux`, `results/tables/concept_saturation.tex`, `results/tables/cross_model_agreement_9model.tex`, `results/tables/hiring_disparity_crossed_9model.tex`, `results/tables/hiring_disparity_marginal_raw_9model.tex`, `results/tables/hiring_steering_transition_summary_9model.tex`
- **Outputs:** `paper/paper/Ulu_Lastra.tex`, `paper/paper/Ulu_Lastra.aux`, `paper/paper/Ulu_Lastra.pdf`, `results/tables/concept_saturation.tex`, `results/tables/cross_model_agreement_9model.tex`, `results/tables/hiring_disparity_crossed_9model.tex`, `results/tables/hiring_disparity_marginal_raw_9model.tex`, `results/tables/hiring_steering_transition_summary_9model.tex`

## Findings

The active manuscript contains 22 tables: Tables 1 through 5 in the main
body and Tables S.1 through S.17 in the supplement. Table S.16 was not
missing. Its label is `tab:disparity_race_gender`, its caption and three
data grids are visible on PDF page 30, and Table S.17 follows on pages 31
and 32. All 22 table labels are unique, and every `tab:` reference resolves
to one of those labels.

Two prose references described appendix material as though it appeared in
the main body. The PCA section now says that the callback-transition census
is summarized in the main text and tabulated below as Table S.12. The raw
marginal-group section now identifies Table S.13 as the preceding appendix
table and appendix version, not a main-text table.

## Caption and Ordering Corrections

Every active table caption now appears below its table body. Six captions
were moved: S.3, S.5, S.7, S.12, S.14, and S.17. The corresponding
generators were changed with the generated artifacts so regeneration cannot
restore the old caption positions.

The first rendered pass exposed a separate float-order defect on page 25:
S.11 appeared above S.9 and S.10 even though the source order was correct.
S.11 is now fixed in place under its own wide-strength subsection. The
supplement therefore reads physically as S.1, S.2, and so on through S.17.

## Verification

- Three final `pdflatex` passes produced a 32-page PDF with no unresolved or multiply defined references, overfull boxes, LaTeX errors, or rerun warnings.
- Every page containing a table was rendered and inspected. Captions are below their table bodies, long-table continuation headers remain at the top of continuation pages, and S.16 is complete on page 30.
- `tests/test_paper_table_builders.py` passes 20 of 20 tests. Two new regression tests enforce below-table captions, exactly 22 unique active table labels, complete table references, and stable S.11 placement.
- The full suite passes 115 of 117 tests. The two unchanged failures are the pre-existing `tests/test_hiring_r4.py` fixtures: one supplies a CSV where the loader expects a raw-data directory, and one omits the required `study` column.
- `git diff --check` passes. The `_long` source and PDF archives were not modified.

## Word-Limit Effect

All prose corrections and caption moves are in the supplement. Main content
remains on pages 1 through 12, References on pages 13 and 14, and
Supplementary Materials begins on page 15. The accepted main-body counts
therefore remain 7,867 conservative visible words and 7,785 plain extracted
words.

## Writing Review

The edited manuscript passages were re-read after rendering. They contain no
em-dash punctuation, repeated adjacent paragraph opener, threefold recurring
sentence frame, or signal-only transition.
