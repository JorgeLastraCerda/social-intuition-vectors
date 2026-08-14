# Pooled Disparity Standardization and Float Round A

- **Produced:** 2026-08-14 12:36 Europe/Berlin
- **Model:** Nine Gemma, Llama, and Qwen checkpoints
- **Scope:** Comparable callback-gap contract and first-pass Results float placement
- **Status:** Complete and locally verified after author-approved Float Round B

## Artifacts

- **Scripts:** `src/hiring_disparity.py`, `src/build_paper_probe_tables.py`, `tests/test_paper_table_builders.py`
- **Inputs:** `results/tables/hiring_audit_*.csv`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`
- **Outputs:** `results/tables/hiring_disparity_gaps_9model.tex`, `paper/paper/Ulu_Lastra.tex`, `paper/paper/Ulu_Lastra.pdf`
- **Figures:** `paper/figures/paper_figure1_axis_arrows.pdf`, `paper/figures/paper_figure2_layer_emergence.pdf`, `paper/figures/fig6_cross_model_agreement.pdf`, `paper/figures/fig14_dense_steering_normalized.pdf`, `paper/figures/paper_figure4_hiring_bidirectional_examples.pdf`

## Comparable Callback-Gap Contract

The main callback-gap table now puts model and human outcomes on the same standardized mean-difference scale. For each axis and outcome, the positive-group mean minus the negative-group mean is divided by the pooled within-group standard deviation on the exact matched applicant-name population. The canonical first-name join in `src/hiring_disparity.py` supplies all nine model comparisons, and the builder rejects any cross-model population change or human-reference drift.

The matched race comparison contains 47 Black-signaling and 180 White-signaling applicant-name rows. The gender comparison contains 154 female-signaling and 115 male-signaling rows. Human reference gaps are $d=+0.152$ for race and $d=-0.474$ for gender. Seven models favor Black-signaling names and two favor White-signaling names; eight favor female-signaling names and Gemma-3-27B alone favors male-signaling names ($d=-0.457$).

## Float Round A

Round A removed the float-page option from Figure 3 and released Figure 5 from forced in-place placement. The rebuilt manuscript remains 35 pages. Figure 3 now appears on page 8 with related Results prose continuing beneath it. Table 7 appears legibly on page 11, and the full-page Figure 4 on page 12 uses the page well. However, Figure 4 is still five pages after its first citation, Figure 5 occupies only part of an otherwise blank page 13, and page 14 combines concept-steering Figure 6 with hiring Figure 7 despite their different narrative blocks. These three issues require a second placement round.

## Float Round B

After author approval, Figure 5 returned to fixed in-place placement within the validation block. The intended targeted `placeins` barrier was unavailable in the local TeX Live Basic installation, which reported `LaTeX Error: File 'placeins.sty' not found.` The final source therefore uses the dependency-free `[!t]` override only on Figure 4, allowing that large figure to take the first available page top without changing global float fractions.

The final flow places Figures 3 and 5 together on page 8, where both support the same validation narrative. Figure 4 fills page 9 and is followed immediately by the concept-steering subsection. Figure 6 appears on page 11, while the hiring-specific Figure 7 appears on page 13 with Table 7. Removing the obsolete Results-ending `\clearpage` lets Discussion begin in the second column of page 14 after the mediation paragraph, eliminating the previous mostly blank page without allowing a Results float into Discussion.

## Verification

The targeted table and agreement tests pass, 8 of 8. `latexmk` produced a 34-page PDF with no overfull boxes, undefined references, unresolved citations, or float-too-large warnings. Results pages 7 through 15 were rendered and inspected at 130 DPI after the final change. No clipping, overlap, excessive float-only whitespace, or cross-block float mixing remains.
