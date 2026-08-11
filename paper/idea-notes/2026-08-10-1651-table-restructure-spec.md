# Table Restructure Spec: Main-Body Tables

- **Timestamp:** 2026-08-10 16:51 Europe/Berlin
- **Requested by:** Jorge
- **Status:** Approved change request, not yet executed
- **Scope:** the nine tables currently in the main body of `paper/paper/Ulu_Lastra.tex`

---

## Context

A full read of every main-body table (not just its size) found three distinct problems:
one table that cannot be read without doing arithmetic across rows, one that mixes two
statistical notations in a single column while understating a null result, and three
36-row tables that consume most of the main body's table space while carrying evidence a
reader would only consult when checking a claim rather than following one.

The aim is a main body with fewer, clearer tables, no loss of evidence, and more honest
presentation of the specificity result. Nothing here requires new GPU work. Every change
is a regeneration or formatting change over data already in `results/tables/`.

---

## Two conventions that must be respected

1. **Edit the generating scripts, not the output `.tex` files.** Per the 2026-08-09 layout
   pass, float placement specifiers (`[p]`, `[tp]`) were deliberately moved into
   `src/build_paper_probe_tables.py` and `src/build_paper_mediation_table.py` so a later
   regeneration cannot silently revert the layout. Any change below follows the same rule:
   change the builder, then regenerate.
2. **Regenerate with `paper/figures/.venv`.** The root interpreter lacks `pandas` and
   `pyyaml`. Use the same environment the layout pass used.

After regeneration, confirm with `grep "begin{table"` that placement specifiers survived,
and rebuild the manuscript to check page count, `??` count, and overfull boxes.

---

## Item 1 — Replace the marginal disparity table with a gap table

**Priority: highest. This is the table a co-author could not follow.**

**Current state:** `build_table2()`, `src/build_paper_probe_tables.py` line 248,
writes `results/tables/hiring_disparity_marginal_9model.tex`, registered in `main()` at
line 1285. Produces 36 data rows across 7 columns.

**Problem.** The table holds four quantities on incompatible scales in one row: model
warmth/competence as z-scores, human warmth/competence as z-scores, human callback as a
raw rate (for example 0.183), and model callback margin as a logit (for example −0.213).
No two cells in a row are comparable.

More seriously, the comparison the paper needs is a **difference between two rows**. To
learn whether a model's race gap matches the human race gap, a reader must locate the
Black row, locate the White row, subtract twice, then compare two results that live on
different scales. Repeated for nine models. The information is present but not legible.

**Required change.** Add a new builder, `build_table_disparity_gaps()`, that emits
`results/tables/hiring_disparity_gaps_9model.tex` with one row per model:

```
                  Race gap (Black - White)        Gender gap (Female - Male)
Model             Model margin   Human callback   Model margin   Human callback
Gemma-3-12B       -0.014         +0.012           +0.093         -0.036
Gemma-3-27B       +0.544         +0.012           -0.194         -0.036
...
```

**Source data.** `results/tables/hiring_disparity_<label>.csv`, already read by
`build_table2()`. Required columns exist: `group`, `model_callback_margin`,
`human_callback`. Compute `Black − White` and `Female − Male` per model. Pure arithmetic,
no new inputs.

**Placement.** Main body, replacing the current table at the same `\input` site. Nine rows
means it no longer needs a dedicated float page; `[tp]` is appropriate, not `[p]`.

**Caption must state the finding, not just the contents.** The human gap column repeats
identically down the table because the benchmark is the same for every model. Say so, and
note that this makes the spread of model gaps around a near-constant human reference the
point of the table.

**One consequence to flag deliberately.** Once the human race gap appears in a column as
`+0.012`, any prose describing model disparities as running "opposite to the human
benchmark" becomes visibly inconsistent with the table. The human race gap in the pooled
benchmark is close to zero and very slightly favours Black-signalling names (Black 0.183,
White 0.171). Prose in Results and Discussion should be checked against this and reworded
to describe **amplification of a near-zero gap**, not reversal. The gender gap does favour
men in the human data (−0.036), so genuine directional disagreement exists there and can
be stated as such.

**The current 36-row levels table moves to the appendix** for readers who want the
underlying means. Keep it generated; only its `\input` location changes.

---

## Item 2 — Fix notation and add a verdict column to signal-vs-control

**Priority: high. This one is a transparency issue, not only a formatting one.**

**Current state:** `build_table_concept_signal_vs_control()`,
`src/build_paper_probe_tables.py` line 947, writes
`results/tables/concept_signal_vs_control_9model.tex`, registered in `main()` at line 1303.

**Problem A, notation.** The `Random control` column mixes two conventions. Four models
report a bracketed interval (`+0.10 [-0.32, +0.49]`), five report mean-plus-minus
(`+0.29 ± 2.77`). Reading down the column requires switching conventions mid-table.

**Problem B, substance.** In the Gemma-4 rows the random control's interval is wider than
the target effect:

| Model | Axis | Dense target | Random control |
|---|---|---:|---|
| Gemma-4-12B | Warmth | +1.84 | +0.29 ± 2.77 |
| Gemma-4-12B | Competence | +2.66 | +0.23 ± 2.56 |
| Gemma-4-26B-A4B | Warmth | +0.05 | −0.84 ± 2.94 |
| Gemma-4-31B | Warmth | +0.44 | +0.85 ± 3.16 |

For Gemma-4-31B warmth the random control is larger than the target. Presented as two bare
numbers, the table reads as a signal-versus-control contrast while the underlying result is
that the target is not distinguishable from random in the Gemma-4 family. A careful reader
will derive this unaided and trust the surrounding claims less for having found it
themselves.

This is consistent with the paper's other evidence: `concept_direction_specificity` already
shows the dense target is the strongest direction in only one of four model-axis rows.
Stating the limitation plainly is the intended behaviour, per the repository's
no-overselling rule and the author's explicit instruction.

**Required change.**

1. Convert every `±` to an explicit interval so one convention runs down the column.
   `+0.29 ± 2.77` becomes `+0.29 [-2.48, +3.06]`.
2. Add a final column, `Exceeds control?`, with a plain `yes` / `no` per row, defined as
   whether the dense-target value falls outside the random control's interval.
3. Move the `Control basis` column into the caption. The heterogeneity of the control basis
   (99 SD-matched directions for five checkpoints, a single direction for four) must remain
   disclosed, as it is now, but it does not need a column.
4. The caption should state the resulting count directly, for example how many of the 18
   model-axis rows exceed their control, and name the family where it fails.

**Do not** drop rows, rescale, or otherwise soften the result.

---

## Item 3 — Flag sign disagreement in the hiring steering slopes table

**Priority: medium. Cheap change, sharpens the paper's headline.**

**Current state:** `build_table_hiring_steering_slopes()`, registered in `main()` at line
1311, writes `results/tables/hiring_steering_slopes_9model.tex`. Bold currently marks
R² ≥ 0.80.

**Observation.** Several rows have a positive fitted slope but a negative endpoint at
α = +0.50. Gemma-4-31B warmth reports slope +7.83 with endpoint −2.59; Gemma-4-12B warmth
reports +8.76 with endpoint −1.26. That contradiction is the heterogeneity finding, and it
is currently visible only to a reader who compares two columns for every row.

**Required change.** Add a visual marker for rows where the fitted slope and the endpoint
disagree in sign. A dagger symbol or a short flag column both work; the existing bold is
already used for R², so use a distinct marker rather than overloading bold. Explain the
marker in the caption and give the count.

---

## Item 4 — Rewrite the specificity caption, finding first

**Priority: medium. No data change.**

**Current state:** `build_table_concept_direction_specificity()`, registered in `main()` at
line 1300.

**Problem.** The table is small and scientifically important, but its caption opens with
column definitions and only reaches the finding at the end. Six terms
(`Dense target`, `SAE decoded`, `Axis-specific`, `Shared`, `Opposing axis`, `Random`)
arrive before the reader learns why the table matters.

**Required change.** Restructure the caption so the finding comes first, definitions after.
The finding is that the mean-difference direction used everywhere else in the study is the
strongest intervention in only one of four model-axis rows, and that a decomposed or
unrelated direction matches or exceeds it in the other three.

Also ensure the two-model limitation (Gemma Scope 2 exists only for Gemma) is stated
adjacent to the table rather than several paragraphs earlier.

---

## Item 5 — Move three tables to the appendix

**Priority: medium. Manuscript-only change, no generator edits.**

Move the `\input` sites in `paper/paper/Ulu_Lastra.tex`, keeping all three tables generated:

| Table | Rows | Reason |
|---|---:|---|
| `hiring_disparity_marginal_9model` | 36 | Superseded in the body by the Item 1 gap table; retained as the levels reference |
| `hiring_disparity_race_gender_9model` | 36 | Strictly more granular than the marginal version, never the primary evidence |
| `mediation_9model` | 36 | `fig19_hiring_mediation_forest` shows the same result faster via filled versus open markers |

Also move `concept_saturation` to the appendix. It is only five rows, so this is not about
space: it is a two-model table inside a nine-model results section, and it prompts "why only
two models here?" at a point in the argument where that question is a distraction. Its
saturation claim is already made in Methods prose and covered more broadly by
`hiring_steering_slopes`.

**Keep `fig19_hiring_mediation_forest` in the main body** when its table moves.

**After moving, the three tables no longer need `[p]` dedicated float pages.** Revisit their
placement specifiers in the builders, since the layout pass assigned `[p]` on the assumption
they sat in the Results body.

---

## Item 6 — Repair cross-references and prose after the moves

**Priority: required, do last.**

- Every `\autoref` pointing at a moved table resolves to an appendix label. Verify no `??`
  in the build (`pdftotext` count, as in the layout pass).
- Results prose that introduces a moved table needs rewording from "the table below shows"
  to a pointer, and should carry the claim in prose so the argument survives without the
  table on the page.
- Appendix labels currently run `S.1` to `S.11`. Confirm no gaps after renumbering.
- Re-run the paragraph-zone containment check from the layout pass: every float in a
  Results paragraph zone should land between that paragraph's prose page and the next
  paragraph's prose page.

---

## Resulting main-body inventory

**Tables (6, plus the model table):**

| Table | Rows | Role in the argument |
|---|---:|---|
| `tab:models` | 9 | Orientation |
| `probe_validation_9model` | 10 | The directions are recoverable |
| `concept_direction_specificity` | 5 | But not cleanly specific |
| `concept_signal_vs_control_9model` | 19 | Signal versus control, honestly stated |
| `hiring_steering_slopes_9model` | 19 | Hiring steering is heterogeneous |
| `probe_human_correlation_9model` | 10 | Human alignment is heterogeneous |
| `hiring_disparity_gaps_9model` *(new)* | 9 | Model gaps against the benchmark |

**Figures (5):** layer emergence, dense steering normalized, bidirectional examples,
hiring disparity, mediation forest.

**Main-body table rows: from 179 to 81**, roughly a 55% reduction, while adding a table and
relegating no evidence. Expected saving of two to three pages.

---

## Acceptance criteria

- [ ] `hiring_disparity_gaps_9model.tex` exists, nine data rows, generated by a new builder
      registered in `main()`
- [ ] Every value in the `Random control` column uses one interval notation
- [ ] `Exceeds control?` column present; caption reports the count and names the failing family
- [ ] Sign-disagreement rows marked in the steering slopes table, marker explained in caption
- [ ] Specificity caption opens with the finding
- [ ] Four tables moved to the appendix, float specifiers revisited in the builders
- [ ] Manuscript builds with zero `??`, zero overfull boxes, zero undefined references
- [ ] Prose describing race disparities no longer claims reversal against the human benchmark
- [ ] Page count reduced relative to the 33-page baseline

---

## Note on framing

The author's explicit position, recorded here so it is not treated as an open question:
results should be shown where they did not come out well, limitations should not be
minimized, and nothing should be arranged to make a weak result look stronger than it is.
Items 2 and 4 exist to satisfy this. Tables and figures should also be precisely written and
readable on their own terms, which is the reasoning behind Items 1, 3, and 6.

---

## Addendum, 2026-08-10 22:40: table size, and a question on heading style

### Table height cannot be reduced by LaTeX settings

Raised while reviewing page count. Recording the finding so the size problem is not
answered with formatting changes that would make things worse.

Every main-body table already applies `\resizebox` and `\small`, and four additionally use
`\scriptsize`. They are at the practical limit of horizontal compression, and shrinking
type further would hurt legibility in print and on a projector.

Table height is driven by **row count, which is a property of the data**: nine models times
four demographic groups produces the 36-row tables. No `\tabcolsep`, `\arraystretch`, font,
or `\resizebox` change alters that.

The only effective lever is fewer rows, which is what Item 1 of this spec achieves by
replacing 36 level rows with 9 gap rows. Please do not respond to "the tables are too
large" by reducing font size or scaling further.

### Open question: heading capitalization and grammatical form

Manuscript `\paragraph{...}` headings currently mix two forms:

- Noun phrases, which is what the original Methods headings use: `Story Preparation.`,
  `Building the Warmth and Competence Vectors.`, `Steering the Concept Vectors.`
- Full sentences, introduced in the Discussion drafted on 2026-08-10:
  `Naming a Direction Is a Claim We Make.`, `Response to Intervention Is Range-Dependent.`

Both are set in Title Case, so capitalization is internally consistent and correct for that
convention. The inconsistency is grammatical form, not capitals.

Two options, and this is a call for the authors rather than something to fix silently:

1. Convert the sentence-form headings to noun phrases (`The Naming Problem`,
   `Range-Dependent Response`, `Steerability Against Mediation`), making every heading in
   the paper a noun phrase.
2. Keep both forms, on the basis that Methods headings label procedures while Discussion
   headings state claims.

Related and separate: Title Case is standard in APA and Chicago, while sentence case is
more common in CS and ML venues. The manuscript is consistently Title Case throughout, so
no change is required, but if a switch to sentence case is ever made it must be applied to
every heading at once.

---

# Table Triage, 2026-08-11

Based on a page-by-page read of the rendered 37-page build
(`Konstanz___Fairness_and_Collective_Decision_Making_in_AI-3.pdf`). Two findings below
were only visible in the rendered output, not in the source.

## Finding A: the human columns are constant across all nine models

In Tab. 8 the columns `Human warmth/competence (z)` and `Human callback` hold the same
four value sets for every model, because the benchmark does not vary by model:

| Group | Human W/C (z) | Human callback |
|---|---|---|
| Black | −0.30 / −0.58 | 0.183 |
| White | +0.21 / +0.15 | 0.171 |
| Female | +0.17 / −0.00 | 0.145 |
| Male | −0.16 / +0.07 | 0.181 |

Those four rows are printed nine times. Of the 36 rows, 32 are repetition. Tab. 9 has the
identical problem with its four crossed groups. This is the strongest argument yet for the
Item 1 restructure, and it also suggests an alternative worth considering: print the human
reference once as a four-row block, and give the model results their own nine-row table.

## Finding B: white space is costing more pages than table size

Page 14 carries one column of prose and is otherwise empty, roughly 65% of the page.
Page 17 holds Tab. 8 alone with about 40% white, and page 18 holds Tab. 9 alone with about
35%. These are consequences of the `\clearpage` barriers from the 2026-08-09 layout pass
interacting with short paragraphs, not of the tables themselves.

Recovering that space is likely worth more than a page and a half, which is comparable to
everything the table restructure achieves. It should be handled as its own pass, after the
table changes settle, since every table move changes where the barriers need to sit.

---

## Four-way triage

### Category 1 — Stay as they are

| Table | Rows | Why it stays untouched |
|---|---:|---|
| Tab. 1, `tab:models` | 9 | Orientation. A reader cannot follow any later result without it. |
| Tab. 2, `probe_validation_9model` | 9 | Carries the entire "the directions are real" claim in one compact table. This is the format the others should aspire to. |
| Tab. 7, `probe_human_correlation_9model` | 9 | Contains the warmth-inversion result the abstract leads with. Significance stars are doing real work. |

### Category 2 — Stay, but need a change to what they contain

| Table | Change | Spec item |
|---|---|---|
| Tab. 5, `concept_signal_vs_control_9model` | Unify `±` and bracket notation into one convention; add `Exceeds control?` column; move `Control basis` into the caption | Item 2 |
| Tab. 6, `hiring_steering_slopes_9model` | Mark rows where fitted slope and α=+0.50 endpoint disagree in sign, using a marker distinct from the existing R² bold | Item 3 |
| Tab. 4, `concept_direction_specificity` | Caption rewritten finding-first; state the two-model limitation next to the table rather than paragraphs earlier | Item 4 |
| Tab. 8, `hiring_disparity_marginal_9model` | Replaced in the body by the nine-row gap table; levels version moves to appendix | Item 1 |

### Category 3 — Stay, but only the sizing or placement is wrong

| Table | Issue | Suggested fix |
|---|---|---|
| Tab. 3, `concept_saturation` | Four data rows rendered as a full-width `table*` with `\resizebox{0.6\textwidth}`, occupying a float slot far larger than its content | Convert to a single-column `table` so it can sit inline in one column rather than claiming a double-column float |
| Tab. 4, `concept_direction_specificity` | Same: four rows in a full-width float | Same treatment, once the caption change in Category 2 is done |

Neither needs its data touched. Both are small tables currently formatted as if large.

### Category 4 — Could be removed from the paper without weakening it

| Table | Rows | Assessment |
|---|---:|---|
| Tab. 9, `hiring_disparity_race_gender_9model` | 36 | **The strongest candidate.** It occupies a full page, 32 of its 36 rows repeat the human benchmark, and Results draws no finding from it. The prose says it recovers interaction structure the marginal table collapses, then stops. Its raw counterpart already exists in the appendix as Tab. S.11. Nothing in the argument depends on it. |
| Tab. 10, `mediation_9model` | 36 | `fig19_hiring_mediation_forest` communicates the same result faster through filled versus open markers, and the Results prose already reports the two numbers that matter. Move to appendix rather than delete, since the coefficients are worth keeping available. |

Removing Tab. 9 from the body and moving Tab. 10 to the appendix would free approximately
two pages on their own.

---

## Recommended order of execution

1. Item 1 gap table, which resolves Tab. 8 and demonstrates the pattern for the rest.
2. Tab. 9 decision, delete from body or relegate. This is an author call, flagged rather
   than assumed.
3. Item 2 and Item 3, the content corrections to Tab. 5 and Tab. 6.
4. Category 3 sizing changes to Tab. 3 and Tab. 4.
5. Item 4 caption rewrite.
6. Barrier and white-space pass, last, once every float has stopped moving.

## Summary of the resulting main body

Seven tables: `tab:models`, `probe_validation`, `concept_saturation`,
`concept_direction_specificity`, `concept_signal_vs_control`, `hiring_steering_slopes`,
`probe_human_correlation`, plus the new nine-row `hiring_disparity_gaps`. Five figures
unchanged. Main-body table rows fall from roughly 179 to about 90, with the largest single
saving coming from the two crossed and marginal disparity tables.
