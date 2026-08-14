# Table Restructure Spec: Main-Body Tables

> ## ACTION LIST — start here
>
> Six changes, all in `src/build_paper_probe_tables.py` unless noted. Nothing below needs
> new GPU work; every change is arithmetic or formatting over data already in
> `results/tables/`.
>
> | # | Table | What must be done | Status |
> |---|---|---|---|
> | 1 | `hiring_disparity_marginal_9model` | New `build_table_disparity_gaps()` emitting a 9-row gap table (Black−White, Female−Male). Levels version to appendix. | **DONE 2026-08-13/14.** `tab:disparity_gaps`, 9 rows. **Deviation:** both sides are standardized mean differences (`d`), not raw margin against percentage points. See Execution status. |
> | 2 | `concept_signal_vs_control_9model` | Convert every `±` to an explicit interval; add `Exceeds control?` column; move `Control basis` into the caption. Do not soften the Gemma-4 null. | **DONE 2026-08-13/14.** One interval convention, `Exceeds control?` column (two-line head), basis in caption, 12 of 18 and the Gemma-4 null stated plainly. |
> | 3 | `hiring_steering_slopes_9model` | Mark rows where fitted slope and the α=+0.50 endpoint disagree in sign, with a marker distinct from the existing R² bold. | **DONE 2026-08-13.** Dagger marker, 8 of 18, count given in the caption; R² bold untouched. |
> | 4 | `concept_direction_specificity` | Rewrite caption finding-first; state the two-model limitation adjacent to the table. | **DONE 2026-08-13.** Caption opens with "strongest in only one of four model-axis comparisons"; Gemma Scope 2 limitation is the second sentence. |
> | 5 | `hiring_disparity_race_gender_9model` | ~~Remove from body~~ **DONE 2026-08-11 by Jorge's session.** Now in Additional Results as `appx:crossed_disparity`, referenced from Future Work. No builder change needed. | **done** |
> | 6 | `mediation_9model` | **DONE 2026-08-11.** Moved to Additional Results as `appx:mediation`. **Note:** `fig19_hiring_mediation_forest` was NOT added, because it is stale (4 models, not 9) and has broken arrow glyphs. Results now carries this result in prose only. | **done, with caveat** |
> | 7 | float placement | **DONE 2026-08-11, regenerated 2026-08-13, finished 2026-08-14.** Tables regenerated so the specifiers reached the `.tex` files; two further placement rounds followed (see the 2026-08-13 float spec and `paper/2026-08-14_1236_pooled_disparity_float_round_a.md`). Final build: 34 pages, no float-only page in the body, 0 overfull boxes. | **done** |
> | 8 | `fig6_cross_model_story_agreement` | Replace four 9x9 heatmaps with a single-column dot plot. **Script written and tested:** run `python paper/figures/fig6_cross_model_agreement.py`, then swap the include. See Item 8. | **DONE 2026-08-13.** Dot plot generated and included as a single-column `figure`; the old heatmap is no longer referenced. |
> | 9 | narrow Results tables | **Tested and verified.** `probe_validation` builder already converted; **must be regenerated to take effect.** `probe_human_correlation` and the gap table still to do. Use narrowing, NOT `\resizebox`. See Item 9. | **DONE 2026-08-13/14.** All three narrowed and regenerated; no `\resizebox` anywhere in the Results tables. |
>
> **Also, added 2026-08-11:** `paper/figures/style.py` was changed from Helvetica
> sans-serif at 11pt to serif at 9pt with `mathtext.fontset: cm`, so figure text matches
> the manuscript's Computer Modern body font. **All figures need regenerating.** This is
> why Figure 3 currently reads as belonging to a different document.
>
> **Correction to the earlier Category 3 note:** `concept_saturation` (7 columns) and
> `concept_direction_specificity` (8 columns) **cannot** be narrowed to single-column
> floats. They genuinely need the double-column width. Results page pressure comes from
> float count and barrier placement, not their width.


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


---

# Table 9 decision: empirical basis, 2026-08-11

Jorge asked whether the crossed race-by-gender table contributes anything to Results. It
was tested rather than assumed. Interaction computed as (gender gap among Black names)
minus (gender gap among White names), from `hiring_disparity_race_gender_9model`:

| Model | Gap among Black | Gap among White | Interaction |
|---|---:|---:|---:|
| Gemma-3-12B | +0.107 | +0.107 | -0.000 |
| Gemma-3-27B | -0.160 | -0.274 | +0.114 |
| Llama-3.1-8B | +0.085 | +0.036 | +0.049 |
| Gemma-4-12B | -0.096 | +0.043 | -0.139 |
| Gemma-4-26B-A4B | +0.657 | +0.602 | +0.055 |
| Gemma-4-31B | +0.101 | +0.547 | -0.446 |
| Qwen3-14B | +0.196 | +0.392 | -0.196 |
| Qwen3.6-27B | +0.058 | +0.160 | -0.102 |
| Qwen3.6-35B-A3B | +0.237 | +0.366 | -0.129 |

**There is a signal.** The gender gap is not independent of race in most checkpoints. In
Gemma-4-31B the female advantage is roughly five times larger among White names; in
Gemma-4-12B the gap reverses sign by race. The marginal table cannot show this.

**Three reasons it still should not occupy a body page.**

1. **The raw values are not comparable across models.** Model margins range from about
   -0.2 (Gemma-3-12B) to about 25.8 (Gemma-4-31B), so an interaction of -0.446 means
   something entirely different in each row. Standardizing by the within-model SDs already
   reported in Limitations changes the ranking: Llama-3.1-8B is roughly +0.49 SD and
   Qwen3-14B roughly -0.54 SD, comparable in magnitude but opposite in direction. The
   printed table does not support that reading.
2. **Cell sizes are small**: 28 Black-Female and 28 Black-Male against 117 White-Female
   and 73 White-Male.
3. **Nothing is tested.** No intervals, no significance testing, no correction for the
   nine comparisons.

**Decision.** Remove from the body, keep generated for the appendix, and record the
observation in Future Work as an untested direction rather than a result. Suggested
wording:

> The crossed race-by-gender breakdown suggests the gender gap is not independent of race
> in most checkpoints, with the two effects pointing in opposite directions across model
> families. Establishing this would require standardized effects and interval estimates on
> larger cells than the present design provides.

Doing it properly, with within-model standardization and bootstrap intervals on the
interaction term, is a genuine analysis task and is out of scope before the deadline.

---

# Item 7 — Float placement pass, 2026-08-11

**Do this last, after Items 1 to 4 and 6.** Every table move invalidates the analysis below.

## The diagnosis

Measured from the rendered build, Results pages 11 to 14:

| Page | Words | Floats | Fill |
|---|---:|---:|---|
| 11 | 356 | 3 | crowded |
| 12 | 412 | 2 | reasonable |
| 13 | 192 | 1 | a 10-row table alone, roughly 70% white |
| 14 | 213 | 0 | one short column, rest empty |

A full two-column page of this layout holds roughly 900 words. Pages 13 and 14 are running
at about a fifth of capacity.

**Two distinct causes, and they need different fixes.**

### Cause 1: every medium and small float can claim a dedicated page

Current specifiers:

| Table | Rows | Specifier |
|---|---:|---|
| `probe_validation_9model` | 10 | `[tp]` |
| `concept_saturation` | 5 | `[tp]` |
| `concept_direction_specificity` | 5 | `[tp]` |
| `concept_signal_vs_control_9model` | 19 | `[tp]` |
| `hiring_steering_slopes_9model` | 19 | `[tp]` |
| `probe_human_correlation_9model` | 10 | `[tp]` |

The `p` in `[tp]` permits LaTeX to place a float alone on a float page. That is what put a
ten-row table by itself on page 13. A table that occupies a quarter of a page should never
be eligible for that.

**Fix: change `[tp]` to `[t]` for all six.** These are `table*` environments, which LaTeX
can only place at the top of a page or on a float page, never at the bottom, so `[t]` is
the correct and only alternative. Removing `p` forces each one to share a page with prose.
`\setcounter{dbltopnumber}{3}` is already set in the preamble, so up to three can stack at
a page top, which is enough headroom to avoid long deferrals.

Reserve `[p]` for floats that genuinely fill most of a page. After Items 1 and 6, the body
may have none left.

### Cause 2: the `\clearpage` barriers now flush too early

Page 14 carries 213 words and no floats at all. That is a barrier firing after a short
paragraph, ending the page with most of it unused. The barriers were added on 2026-08-09
to stop floats drifting several pages past the prose that discusses them, which was a real
problem at the time. With three large floats leaving the body, the condition that justified
them largely disappears.

**Fix, in this order:**

1. Apply Items 1 to 4 and 6 first.
2. Change the six specifiers above from `[tp]` to `[t]`.
3. Rebuild, then **remove the barriers one at a time**, checking after each whether floats
   still land in their own paragraph's zone. The 2026-08-09 entry records that one barrier
   was already removed on measurement and saved a full page without breaking containment;
   the same is likely true of others now.
4. Re-run the containment check from that entry: every float in a paragraph zone should
   land between that paragraph's prose page and the next paragraph's prose page.

Expected recovery is one and a half to two pages, comparable to the entire table
restructure.

## Also fixed 2026-08-11, no action needed

`tab:story_prompt` was drifting forward out of its own subsection and splitting "PCA
Denoising, All Nine Models". Changed from `[h]` to `[H]`, which the `float` package
supports and which is already loaded. The table now sits with its own subsection. Use `[H]`
for any appendix table that must stay inside its section rather than float.

---

# Item 8 — Replace the four agreement heatmaps with a dot plot

**Script already written and tested. Run it, swap the include, done.**

## Why

`fig6_cross_model_story_agreement` renders four 9x9 heatmaps, 324 cells. Each matrix is
symmetric, so half of every panel is redundant, and the diagonal is 1.00 by construction.
The two "overall" panels carry no visible variation at all, because every value falls
between 0.74 and 0.99 and reads as uniformly dark. The axis labels are unreadable at the
size the figure is placed. It occupies a full page while communicating four numbers.

Pair identity was checked to see whether a matrix is justified. Within-condition agreement
is slightly higher for same-family pairs (warmth median 0.56 against 0.45, competence 0.62
against 0.51), but there are only 5 same-family pairs against 31 cross-family, which is too
thin to build four panels around.

## What to run

```
python paper/figures/fig6_cross_model_agreement.py
```

Writes `fig6_cross_model_agreement.pdf` and `.png` beside the script. Verified output:

| Row | n | min | median | max |
|---|---:|---:|---:|---:|
| Warmth overall | 36 | 0.74 | 0.90 | 0.98 |
| Warmth within-condition | 36 | 0.10 | 0.46 | 0.83 |
| Competence overall | 36 | 0.78 | 0.94 | 0.99 |
| Competence within-condition | 36 | 0.20 | 0.54 | 0.90 |

These match the values already quoted in Results, so no prose needs changing.

**Data source.** The script reads `results/tables/cross_model_agreement_9model.csv`. That
file is gitignored, so it exists on the machine that generated it but not in a fresh clone.
The script falls back to parsing the committed `.tex`, which carries the same two columns
for all 72 rows, and prints which source it used. Run it where the CSV exists if possible.

**Sizing.** The figure is 3.4 by 2.6 inches, sized for a single column. It follows the
repository convention for hand-made figures: a same-basename `.py`, `.pdf` and `.png`
triplet in `paper/figures/`, and it calls `style.apply()` so it picks up the serif change
made on 2026-08-11 automatically.

## Manuscript change

Replace the `figure*` block for `fig6_cross_model_story_agreement` with a single-column
`figure`:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig6_cross_model_agreement.pdf}
\caption{\textbf{Cross-model story-ranking agreement, all 36 pairs.} Each point is one
model pair. Overall correlations include the high-versus-low condition split;
within-condition correlations restrict the comparison to stories sharing one condition and
so reflect agreement on finer story ordering. Vertical rules mark medians. Full pairwise
values are in \autoref{tab:cross_model_agreement}.}
\label{fig:fig6_cross_model_agreement}
\end{figure}
```

Update the `\autoref` in the Vector Validation paragraph to the new label. The old
`fig6_cross_model_story_agreement` files can stay in `paper/figures/` unreferenced or be
deleted; nothing else points at them.

---

# Item 9 — Convert narrow Results tables to single-column floats

## Why

Methods reads better than Results because Methods has one float across 3,100 words and
uses inline `itemize` blocks that flow with the text column. Results has ten floats across
2,000 words, and every one is a `table*` or `figure*`. A full-width float in a twocolumn
document can only be placed at the top of a page or on a float page. It can never sit at a
column bottom and text can never flow past it. That is why Results reads as consecutive
slabs rather than prose with illustrations.

Items 1, 5 and 6 reduce float count from ten to seven. Item 7 stops small floats claiming
their own page. This item addresses the remaining cause.

## Candidates

Three tables are narrow enough that single-column placement is plausible. All three should
be tested rather than assumed, since a single column is roughly 3.4 inches.

| Table | Columns | Note |
|---|---:|---|
| `probe_validation_9model` | 5 | Cells are `W / C` pairs; may need `\footnotesize` |
| `probe_human_correlation_9model` | 6 | Tightest of the three, test first |
| new `hiring_disparity_gaps_9model` | 5 | From Item 1; design it single-column from the start |

## How

In the builder, change `\begin{table*}[tp]` to `\begin{table}[tb]` and drop the
`\resizebox{\textwidth}` wrapper, since a single-column table should size to
`\columnwidth` or simply be allowed to set naturally at `\footnotesize`. Note that `tb` is
available here: a single-column float can go at a column bottom, which is exactly the
behaviour that lets text flow around it and gives Methods its rhythm.

Rebuild and check for overfull boxes. If a table overflows the column, either abbreviate
the model names in that table only, for example `G3-12B`, or leave it as `table*` and
record that it was tested and did not fit.

**Do not force it.** A cramped single-column table is worse than a clean full-width one.
The goal is that at least one or two Results floats interleave with prose, not that all of
them do.


---

# Item 9 results, 2026-08-11: verified, with one important warning

## `\resizebox` is the wrong tool

The first attempt kept `\resizebox` and changed its argument from `\textwidth` to
`\columnwidth`. It compiled with zero overfull boxes, which is misleading: `\resizebox`
scales to whatever fits, so a five-column table squeezed into half the width rendered at
roughly half the caption's type size. Legible in a PDF viewer at 200%, not on paper.

**Narrow the content instead.** For `probe_validation` this meant three changes:

1. Column heads reduced from `Cohen's $d$ (W / C)`, `Random-null $z$ (W / C)`,
   `Split-half cosine (W / C)`, `Cross-axis accuracy (W$\to$C / C$\to$W)` to
   `$d$`, `$z$`, `$\cos$`, `acc.`, with the definitions moved into the caption where they
   already partly were.
2. Model names abbreviated through a new `SHORT_NAME` map (`Gemma-4-26B-A4B` becomes
   `G4-26B`), with the abbreviation scheme stated in the caption.
3. `\small` to `\footnotesize`, `\tabcolsep` from 4pt to 3pt, and `\resizebox` removed
   entirely.

Result: zero overfull boxes, full-size readable type, the table sitting at the top of the
right column with prose flowing below it, and a completely full page. This is the Methods
rhythm the Results section was missing.

## Status

`build_table_probe_validation()` in `src/build_paper_probe_tables.py` has been updated to
emit this form, and `SHORT_NAME` has been added near `STUDY_ORDER`. The change is
syntax-checked but **could not be regenerated in the session that made it**, because
`results/logs/split_half_stability_*.json` is gitignored and absent from a fresh clone.

**Regenerate and check the rendered page, not just the log.** Zero overfull boxes does not
prove legibility, as the first attempt shows.

`probe_human_correlation_9model` (6 columns) and the Item 1 gap table still need the same
treatment. Design the gap table single-column from the start.

---

# Item 6 result, 2026-08-11: done, but the forest plot could not be used

The mediation table now sits in Additional Results as `appx:mediation`, and the Results
sentence points there.

**`fig19_hiring_mediation_forest` was deliberately not added to the body.** It is stale: it
shows four models (Gemma-3-12B, Gemma-3-27B, Llama-3.1-8B, Qwen3-14B) with sixteen rows,
from before the nine-model expansion. Its title and legend also contain broken glyphs where
arrows and a not-equal sign should be. Placing it in a paper that claims nine models
throughout would misrepresent the work.

All nine mediation logs exist (`results/logs/hiring_mediation_*.json`), so a nine-model
version is regenerable. Two options if there is time:

1. Regenerate the forest plot for nine models. Thirty-six rows will make it tall, probably
   a full-width float of roughly the same footprint as the table it replaced, which
   defeats the purpose.
2. **Preferred:** build a compact strip plot in the same style as
   `fig6_cross_model_agreement.py`. Four rows (race-warmth, race-competence,
   gender-warmth, gender-competence), nine points each, filled where the interval excludes
   zero and open where it does not, with a rule at zero. That carries the finding, fits one
   column, and matches the figure vocabulary now used elsewhere.

Until one of those exists, the mediation result lives in Results prose, which already
states both headline numbers.

---

# Item 7, part one: DONE 2026-08-11 (manuscript side only)

Jorge asked why floats appear consecutively when the `.tex` separates them with prose.
Diagnosed and partly fixed. **Two changes were made in `paper/paper/Ulu_Lastra.tex`. The
table-side half of Item 7 still requires regeneration and remains open.**

## Why floats were clumping

Three mechanisms compounding:

1. **`[p]` means float-page-only, not a preference.** Two figures carried
   `\begin{figure*}[p]`, which *forbids* sharing a page with body text. LaTeX was obeying
   an instruction, not ignoring one.
2. **Full-width floats cannot sit at a page bottom.** `table*` and `figure*` accept only
   `t` (page top) or `p` (float page). There is no `b`. So they queue waiting for tops.
3. **Six `\clearpage` barriers flushed the queue at once.** Each barrier forces every
   pending float out before the document continues, emitting them consecutively and ending
   the preceding text page wherever it happened to be.

## What was changed

- Both `\begin{figure*}[p]` changed to `\begin{figure*}[t]`, with a source comment.
- All six `\clearpage` barriers in Results removed, with a source comment recording why and
  noting that containment was re-verified.

## Measured effect

| | Before | After |
|---|---|---|
| Pages | 38 | **36** |
| Emptiest Results page | 192 words | 237 words |
| Pages with no body text | 2 | 0 |
| Overfull boxes | 0 | 0 |
| Undefined refs / `??` | 0 | 0 |

Per-page body words on 6 to 16 went from 810, 690, 432, 516, 709, 356, 412, 192, 213, 354,
409 to 867, 804, 835, 525, 717, 428, 237, 656, 368, 842, 506.

## Containment was checked, not assumed

The 2026-08-09 entry added those barriers because floats were drifting up to nine pages
past their prose. That risk was re-tested after removal by reading float pages from the
`.aux` and matching each to the paragraph that first cites it. Every float now lands with
its owning paragraph and in narrative order: Vector Validation floats on 9 to 10, Steering
the Concept Vectors on 11 to 13, Alignment and Steering Hiring on 14, Group-Level
Disparity on 16. The original concern does not recur, most likely because moving the
crossed-disparity and mediation tables to the appendix removed two large floats from the
queue.

**If floats are added back to the body, re-check containment before assuming barriers are
still unnecessary.**

## What remains for Item 7

The six generated tables still carry `[tp]`. The `p` there still permits a dedicated float
page. Changing them to `[t]` requires editing the builders and regenerating, which could
not be done in this session. Combined with Items 1, 8 and 9 this should recover further
space.


---

# Item 7 complete, 2026-08-11: what finally fixed the clumping

Four separate causes, all needed fixing. Any one left in place reproduced the problem.

## 1. `[p]` on two figures
`[p]` forbids a float from sharing a page with body text. Changed to `[t]`.

## 2. Six `\clearpage` barriers
Each flushed the whole pending float queue at once. Removed; containment re-verified from
the `.aux`.

## 3. Float-area limits permitted a 95%-float page
The 2026-08-09 preamble set `\textfraction{0.05}`, `\dbltopfraction{0.9}` and
`\setcounter{dbltopnumber}{3}`, which together allow a page to be 95 percent float with
three full-width floats stacked at one top. Retightened to `\textfraction{0.15}`,
`\topfraction{0.7}`, `\dbltopfraction{0.6}`, `\dbltopnumber{1}`.

## 4. Source order placed floats back to back
Several floats sat adjacent in the source with no prose between them. LaTeX queues in
source order, so adjacent floats emit together regardless of placement specifiers. The
Results float order was rebuilt programmatically: each float is now emitted after the
paragraph that first cites it, at most one per paragraph boundary, so prose always
separates them. Verified: zero adjacent float pairs in source.

## 5. `[tp]` on the generated tables
The `p` still permitted dedicated float pages where several tables could share. Changed to
`[t]` in `src/build_paper_probe_tables.py` (7) and `src/build_paper_mediation_table.py`
(1). **This is the change that requires regeneration.**

## Measured result, with all five applied

| | Before | After |
|---|---|---|
| Pages | 38 | 36 |
| Pages with 2+ floats | 4 | 0 |
| Emptiest Results page | 192 words | 396 words |
| Overfull / undefined / `??` | 0 | 0 |

Page map after: p9 Table 2, p10 Figure 3, p11 Table 3, p12 Table 4, p13 Table 5, p14
Table 6, p15 Table 7, each with 561 to 838 words of body text on the same page.

**Until the tables are regenerated, the `.tex` files still carry `[tp]` and pages 11 to 12
will still stack two or three tables.** Everything else is already in the manuscript.

---

# Cross-reference and citation fixes, 2026-08-11

- **Three broken "section" links repaired.** `\autoref` on a label attached to an
  unnumbered `\subsection*` renders as the word "section", because the label picks up the
  enclosing numbered counter. Affected `appx:probe_checks`, `appx:mediation` and
  `appx:crossed_disparity`. Replaced with `\nameref`, which renders the subsection title.
  **If any further appendix subsection gets a label, use `\nameref`, not `\autoref`.**
- **Three figures had no citation in the body.** `fig:paper_figure1_axis_arrows` (Fig. 3)
  and `fig:paper_figure2_layer_emergence` (Fig. 6) were cited nowhere at all;
  `fig:hiring_bidirectional_examples` (Fig. 7) was cited only from the appendix, which is
  why it appeared in the list of figures but nowhere in the text. All three now carry a
  sentence in Results that states what the figure shows. Both older figures were kept, as
  Jorge asked, and both still support their claims: the axis-arrows panel shows the
  warmth-competence angle varying from 41.5 to 59.6 degrees across models, and the
  layer-emergence panel shows Cohen's d rising through middle layers and peaking near the
  0.66 probe depth.

---

# Execution status, 2026-08-14

Every item in the ACTION LIST is now closed. This block records where each one landed, the
two places where the executed version differs from the spec, and three constraints a later
session should not rediscover the hard way.

## Where the work lives

| Item | Landed in |
|---|---|
| 1 | `build_table_disparity_gaps()`, `src/build_paper_probe_tables.py`; `tab:disparity_gaps` in Results |
| 2 | `build_table_concept_signal_vs_control()`; `tab:concept_signal_vs_control` |
| 3 | `build_table_hiring_steering_slopes()`; dagger marker plus caption count |
| 4 | `build_table_concept_direction_specificity()`, caption only |
| 8 | `paper/figures/fig6_cross_model_agreement.py`, included as a single-column `figure` |
| 9 | `probe_validation`, `probe_human_correlation` and the gap table, all narrowed |
| 7 | `Ulu_Lastra.tex` float specifiers plus the two placement rounds of 2026-08-14 |

Reports: `paper/2026-08-13_1816_manuscript_table_figure_revision.md` and
`paper/2026-08-14_1236_pooled_disparity_float_round_a.md`. The second supersedes the first on
gap units and final layout.

## Two deliberate deviations

**The gap table reports `d` on both sides, not margin against percentage points.** The spec's
worked example put the model gap in raw callback margin next to a human gap in percentage
points. Those two columns cannot be read against each other, which is the same objection the
spec raises against the old marginal table. Both sides are now standardized mean differences
on the matched-name population: positive-group mean minus negative-group mean over the pooled
within-group SD. Human reference gaps are `d=+0.15` for race and `d=-0.474` for gender. The
builder recomputes them per model and raises if the matched population or the shared human
values drift, so the repetition down the human columns is enforced rather than assumed.

**The marginal pivot is 4+9+9 rows, not 13.** The 2026-08-13 float spec proposed one 9-row
model block with grouped columns. Nine columns of `w/c` pairs plus four margin columns did not
set cleanly at `\textwidth`, so the spec's own fallback was used: the human benchmark once,
then a representation grid and a callback-margin grid. The table is in the appendix and no
longer competes for a body page. Verified by extracting every numeric token from the old and
new `.tex` and comparing as multisets: no value was lost, the only difference is explicit `+`
signs.

## Three constraints for later sessions

1. **The local TeX Live tree is Basic.** `placeins.sty` and `makecell.sty` are both absent.
   Layout fixes have to be dependency-free: `[!t]` was used instead of a `\FloatBarrier`, and
   the two-line `Exceeds control?` head uses the kernel's `\shortstack` rather than
   `\makecell`. Check with `kpsewhich` before reaching for a package.
2. **The builders must be run as modules.** `python src/build_paper_probe_tables.py` fails on
   `ModuleNotFoundError: No module named 'src'`. Use
   `PYTHONPATH=. paper/figures/.venv/bin/python -m src.build_paper_probe_tables`. The root
   interpreter still lacks `pandas`.
3. **Regeneration is now reproducible.** Re-running both builders after the 2026-08-14 edits
   changed only the comment blocks and the one header line; every number stayed byte-identical.
   If a future regeneration moves a value, that is a real input change and should be
   investigated, not committed.

## Still outstanding

- **Overleaf.** The regenerated `results/tables/*.tex` carry the specifiers and the new head.
  Until they are re-uploaded, Jorge's Overleaf build will keep showing the old layout.
- **Unrelated:** `tests/test_hiring_r4.py` has had two failures since 2026-08-06 (`5ff1a0e`),
  a `KeyError: 'study'`. Untouched by this work and deliberately left out of scope.
