# Consolidated Probe-vs-Human Result Tables Added to the Manuscript

- **Produced:** 2026-07-20 19:35 Europe/Berlin
- **Model:** Nine Gemma, Qwen, and Llama checkpoints
- **Scope:** Manuscript integration of existing name-level probe-vs-human results as three tables, plus a Limitations addition on probe-layer selection
- **Status:** Complete

## Artifacts

- **Scripts:** `src/build_paper_probe_tables.py` (new); reuses `src/hiring_r4.py::load_and_join`
- **Inputs:** `results/logs/hiring_probe_vs_human_<label>.json` (9 files), `data/processed/<vectors_subdir>/meta.json` (9 files), `results/tables/hiring_disparity_<label>.csv` (9 files), `results/tables/hiring_audit_<label>.csv` (9 files), `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`, `results/tables/hiring_group_r4_<label>.csv` (5 files, used only as a regression gate)
- **Outputs:**
  `results/tables/probe_human_correlation_9model.tex` (Table 1, main text),
  `results/tables/hiring_disparity_marginal_9model.tex` (Table S.2, appendix),
  `results/tables/hiring_disparity_crossed_9model.tex` (Table S.3, appendix, re-derived)
- **Figures:** none (tables only)

## What was added

No new experiments were run. All nine models already had isolated name-level
probe-vs-human correlation results and race/gender disparity results scattered
across per-model JSON/CSV logs; only a few numbers had been folded into the
manuscript prose. This step consolidates all nine models into three
publication-ready LaTeX tables and wires them into `paper/paper/Ulu_Lastra.tex`.

**Table 1 (main text, `tab:probe_human`).** Per-model probe layer (index and
resolved fraction of depth) alongside four Spearman correlations: probe
warmth/competence projection vs. human warmth/competence ratings, and probe
projection vs. the model's own unsteered callback margin. Referenced from the
"The Model's Internal Probe Scores Track Human Social Perceptions Unevenly
Across Architectures" paragraph, with two added sentences noting that
Gemma-4-12B shows no reliable warmth alignment (ρ = +0.020, n.s.) and that
Gemma-4-26B-A4B combines a weak, inverted warmth-human correlation with the
largest callback-warmth correlation in the table (ρ = +0.510).

**Table S.2 (appendix, `tab:disparity_marginal`).** Existing marginal race
(Black/White) and gender (Female/Male) breakdown, unchanged from
`hiring_disparity_<label>.csv`, formatted for all nine models.

**Table S.3 (appendix, `tab:disparity_crossed`).** Re-derived crossed
race × gender breakdown (Black-Female / Black-Male / White-Female /
White-Male) for all nine models, including `model_warmth` and
`model_competence`, which the five pre-existing `hiring_group_r4_<label>.csv`
files did not carry. Computed by joining `hiring_audit_<label>.csv` against
the published human callback data with the exact same join used by
`src/hiring_r4.py::load_and_join` (lowercase first name plus matching study).

**Limitations addition.** A new passage states that the probe layer for every
model is fixed by the `probe_layer_frac = 0.66` heuristic rather than the
layer of maximum warmth/competence separability, and quantifies the gap using
the existing layer-sweep tables: Gemma-3-12B reaches warmth Cohen's *d* = 2.68
at its selected layer versus 6.27 at the best-separated layer; Gemma-3-27B
2.95 vs. 5.10; Gemma-4-31B 7.56 vs. 11.49. It also notes that high separability
at the selected layer does not guarantee name-level generalization
(Gemma-4-12B: *d* = 8.46 at its selected layer, yet ρ = +0.020, n.s., against
human warmth ratings), and leaves open whether combining several
high-separability layers would improve name-level alignment.

## Verification

- `src/build_paper_probe_tables.py` regression-gates Table S.3 against the
  five pre-existing `hiring_group_r4_<label>.csv` files (exact match on
  recomputed callback margin per race×gender cell); the gate passed on first
  successful run.
- Spot-checked Table 1 numbers against values already cited in the manuscript
  prose: Gemma-3-12B warmth ρ = +0.366, competence ρ = +0.239; Gemma-3-27B
  warmth ρ = +0.396, competence ρ = +0.272; all match.
- First compile attempt produced an `Overfull \hbox (315pt too wide)` for
  Table 1; traced to the table being a single-column `table` environment
  inside the twocolumn body while its `\resizebox{\textwidth}{!}` used the
  full-page (not per-column) text width. Fixed by switching Table 1 to
  `table*` (the appendix, where Tables S.2/S.3 live, is already `\onecolumn`
  from line 1101 onward, so those two needed no such fix).
- Final `latexmk -pdf -interaction=nonstopmode -halt-on-error` build produced
  21 pages (up from 18) with no `Undefined` references and no `Overfull
  \hbox` warnings. Rendered page images (via `pdftoppm`) were visually
  inspected for all three tables and the edited Limitations paragraph; no
  text collisions or clipped columns. Note: `pdftotext` extraction is broken
  for this document's fonts (returns near-empty output), so all textual
  verification here was done by rendering pages to PNG and reading them
  directly rather than by grepping extracted text.
- Anti-formulaic self-check on the added Limitations passage: no em-dashes,
  and only a single "This ..." sentence opener in the entire Limitations
  paragraph (the pre-existing content plus the new addition), so no
  three-or-more repetition of any opening frame.

## Addendum: unresolved bf16 quantisation caveat added to Limitations

A follow-up request asked for a previously unresolved "byte precision" issue
to be added to Limitations as well. This was traced to
`paper/2026-07-02_1000_bf16_quantisation_limitation.md` (Bug B1): callback
margins are computed as `logit(Yes) - logit(No)` on bf16 tensors, producing a
0.125-unit grid. A `.float()` cast on the subtraction (already reflected in
the manuscript) only removes rounding in the subtraction step; the two
operands are already bf16-quantised before they reach it, so the grid
persists regardless. A complete fix (float32 inference) was never attempted
because it roughly doubles GPU memory and is likely infeasible at 27B. The
Limitations paragraph in `paper/paper/Ulu_Lastra.tex` was extended with this
explanation, plus the previously undocumented Llama-3.1-8B margin SD (0.12,
12 unique values, comparably unreliable to Gemma-3-12B) and Qwen3-14B margin
SD (0.35, 17 unique values, usable range like Gemma-3-27B), both sourced from
the table in the 2026-07-02 report. Rebuilt and re-verified `Ulu_Lastra.pdf`
(21 pages, no undefined references, no overfull boxes; pages 10-11 visually
inspected via `pdftoppm`).

## Caveats

- Model warmth/competence values in Tables S.2/S.3 are raw, unnormalized
  projections onto the concept direction and are explicitly captioned as not
  comparable in magnitude across models; only within-model group differences
  are interpretable.
- Table S.3 group sizes are small and uneven (9 names per Black cell vs. 84
  for White-Female), noted directly in the manuscript text as descriptive
  rather than adequately powered comparisons.
- A local Python virtual environment (pandas/pyyaml/scipy/scikit-learn) was
  created under the session scratchpad to run this CPU-only script, since the
  system Python at `/opt/homebrew/bin/python3` has no data-science packages
  installed and is externally managed (PEP 668). No project files or system
  Python were modified to accommodate this.
