# Float Placement: Diagnosis and Fix Spec

- **Timestamp:** 2026-08-13, Europe/Berlin
- **Audience:** Emre (executor), Jorge (owner)
- **Symptom reported:** tables and figures render back to back. Pages 8-9 carry three
  consecutive floats, page 10 carries three tables, pages 12-13 carry three tables.
- **PDF examined:** `paper/paper/Ulu_Lastra.pdf`, built 2026-08-13 14:59, 36 pages.
- **Status:** diagnosis complete, no files modified. This document is the work order.

---

## 1. Diagnosis

There are **two independent causes**. Fixing only the first will make the build worse,
not better, so they must land together.

### Cause A: the 11 August float fix was never applied to the shipped tables

`src/build_paper_probe_tables.py` and `src/build_paper_mediation_table.py` were edited on
11 Aug to emit `[t]`. The generated `.tex` files in `results/tables/` were **never
regenerated**, so the PDF is still built from the old specifiers. This is exactly the
"Two changes need regeneration" caution in
`paper/idea-notes/2026-08-11-2300-manuscript-session-summary.md`, section "Layout and build".

Verified by reading both sides:

| Generated file in `results/tables/` | Shipping | Builder emits | Builder line |
|---|---|---|---|
| `probe_validation_9model.tex` | `table*[tp]` | `table*[t]` | 211 |
| `concept_saturation.tex` | `table*[tp]` | `table*[t]` | 842 |
| `concept_direction_specificity.tex` | `table*[tp]` | `table*[t]` | 910 |
| `concept_signal_vs_control_9model.tex` | `table*[tp]` | `table*[t]` | 1006 |
| `probe_human_correlation_9model.tex` | `table*[tp]` | `table*[t]` | 211 |
| `hiring_steering_slopes_9model.tex` | `table*[tp]` | `table*[t]` | 1251 |
| `hiring_disparity_marginal_9model.tex` | `table*[p]` | `table*[t]` | 273 |
| `hiring_disparity_race_gender_9model.tex` | `table*[p]` | `table*[t]` | 418 |
| `mediation_9model.tex` | `table*[p]` | `table*[t]` | 124 (mediation builder) |
| `hiring_steering_transition_summary_9model.tex` | `table[htbp]` | `table[tb]` | 651 |

The `p` is what produces the symptom. `p` permits a float to be sent to a **dedicated float
page**, and `\dblfloatpagefraction{0.7}` means LaTeX builds such a page as soon as the
deferred queue can fill 70 percent of it. Pages 10 and 12 of the current PDF contain **no
body text at all** — they are float pages assembled out of the deferred queue. That is the
mechanism, not a specifier being ignored.

### Cause B: three tables are physically too tall to sit on a text page

This is the part regeneration alone does not fix.

`\topfraction` is 0.7, so a double-column float can occupy at most 70 percent of the text
block. Measured against the rendered PDF:

| Table | Body rows | Approx. page share | Can it take `[t]`? |
|---|---:|---:|---|
| `hiring_disparity_marginal_9model` (Table 8) | 36 | ~78% | **No** |
| `hiring_disparity_race_gender_9model` (appendix) | 36 | ~78% | **No** |
| `mediation_9model` (appendix) | 36 | ~75% | **No** |
| `concept_signal_vs_control_9model` (Table 5) | 18 | ~40% | Yes |
| `hiring_steering_slopes_9model` (Table 7) | 18 | ~38% | Yes |
| all others | <=10 | <25% | Yes |

If those three are set to `[t]` at their current height, LaTeX can never place them. They
will be deferred indefinitely and either trigger `Too many unprocessed floats` or be dumped
at the next `\clearpage`, which is a worse layout than today's.

This is the situation the session summary already anticipated: *"Table height is set by row
count, not formatting. Do not answer 'the tables are too large' with a smaller font or more
scaling."*

---

## 2. The fix for the three tall tables

**All three are 36 rows because they print nine models x four groups. In each case three of
the columns contain human data, which does not vary by model, and is therefore printed nine
times over.**

Verified programmatically against the shipped `.tex`. For every one of the four groups,
across all nine models, there is exactly **one** distinct value of (`n`, human
warmth/competence, human callback):

```
hiring_disparity_marginal_9model.tex   (9 models parsed)
  Black   distinct (n, human w/c, human callback) across models: 1   (47,  -0.30 / -0.58, 0.183)
  White   distinct: 1   (180, +0.21 / +0.15, 0.171)
  Female  distinct: 1   (154, +0.17 / -0.00, 0.145)
  Male    distinct: 1   (115, -0.16 / +0.07, 0.181)

hiring_disparity_race_gender_9model.tex   (9 models parsed)
  Black-Female distinct: 1   (28,  -0.24 / -0.66, 0.175)
  Black-Male   distinct: 1   (28,  -0.37 / -0.41, 0.178)
  White-Female distinct: 1   (117, +0.23 / +0.02, 0.148)
  White-Male   distinct: 1   (73,  +0.03 / +0.03, 0.216)
```

So Table 8 is really **a 4-row human reference block plus a 9 x 4 grid of model numbers**,
currently rendered as 36 rows of mostly repetition.

### Target layout for Table 8 (`build_table2`, builder line 256)

Replace the long form with two stacked blocks inside one `table*`:

**Block 1 — human benchmark, 4 rows, printed once:**

| Group | $n$ | Human warmth/competence ($z$) | Human callback |
|---|---:|---:|---:|
| Black | 47 | -0.30 / -0.58 | 0.183 |
| White | 180 | +0.21 / +0.15 | 0.171 |
| Female | 154 | +0.17 / -0.00 | 0.145 |
| Male | 115 | -0.16 / +0.07 | 0.181 |

**Block 2 — model rows, one per model, groups become columns:**

| Model | Black w/c | White w/c | Female w/c | Male w/c | Margin B | Margin W | Margin F | Margin M |
|---|---|---|---|---|---|---|---|---|
| Gemma-3-12B | -0.44/-0.43 | +0.45/+0.44 | +0.11/+0.10 | -0.04/-0.03 | -0.213 | -0.199 | -0.165 | -0.258 |
| ... 8 more rows | | | | | | | | |

**36 rows becomes 13.** No information is lost, the redundancy is removed, and the result
fits comfortably under `\topfraction` with `[t]`.

If nine columns proves too wide even at `\textwidth`, the fallback is to split into two
`table*` floats (a "representation" table and a "callback margin" table), each ~9 rows.
**Do not reach for `\resizebox` instead** — see section 5.

### Same treatment for the two appendix tables

- `hiring_disparity_race_gender_9model` (`build_table_race_gender`, line 402): identical
  structure, four Race x Gender cells instead of four marginal groups. Same pivot.
- `mediation_9model` (`build_paper_mediation_table.py`, line 122): 36 rows are 9 models x
  (Race, Gender) x (Warmth, Competence). Columns are $a$, $b$, IE, 95% CI — none of these
  are constant across models, so **the redundancy argument does not apply here**. Pivot to
  one row per model x grouping (18 rows, warmth and competence side by side), or accept it
  as a deliberate single full-page appendix float with `[p]`. It is in the appendix; a
  full-page table there is normal and costs nothing. **Recommend: leave `[p]`, fix the two
  body-adjacent tables first.**

---

## 3. Execution order

Land these in one commit so the intermediate state is never built.

1. **Rewrite `build_table2`** in `src/build_paper_probe_tables.py` to the pivoted layout.
   Emit `[t]`, drop the `\resizebox`, update the leading comment block (it currently
   explains the `[p]` choice, which will no longer be true).
2. **Rewrite `build_table_race_gender`** the same way.
3. **Regenerate** all tables: `python src/build_paper_probe_tables.py` and
   `python src/build_paper_mediation_table.py`. This is what applies both the pivot and the
   already-committed `[t]` change to the other seven tables in one pass.
4. **Verify no number changed** other than by transposition — see section 4.
5. **Rebuild and compare** against the current 36-page PDF: page count, count of pages with
   zero body text, overfull box count.

**If step 3 fails** because the builders need `data/processed/` artifacts that are not
present in Emre's checkout: edit the generated `results/tables/*.tex` by hand to the same
result, **and still make the builder change**, so the next regeneration does not silently
revert the layout. Do not do one without the other.

---

## 4. Verification, before rebuilding

The pivot moves numbers between cells, which is exactly the kind of edit that silently
drops a value. Check mechanically, not by eye:

- Extract every numeric token from the old and new `.tex` for each pivoted table and
  compare as multisets. Human columns will legitimately appear 9x fewer times; assert that
  the count of each such token drops from 9 to 1 and that **no token disappears entirely**.
- Assert model-specific tokens (model w/c, model margin) survive at count 1 each, unchanged.
- Spot-check three cells end to end against the source CSVs
  (`results/tables/hiring_disparity_<label>.csv`) rather than against the old `.tex`.

Then, on the built PDF:

- Pages carrying two or more floats: target **0** (currently 3 pages: 8, 10, 12).
- Pages with no body text at all: target **0** in the body, appendix float pages acceptable.
- Overfull boxes: must stay at **0**.
- Every float still appears after the paragraph that first cites it, and within a page or
  two of it. The `\clearpage` barriers were removed on 11 Aug precisely because containment
  had been verified at the then-current float count; **that verification has to be redone**
  after any change to float heights, because shorter floats place earlier and drift changes.

---

## 5. Two traps, restated

**`\resizebox` is a false pass.** Three of the tables in scope currently wrap their tabular
in `\resizebox{\textwidth}{!}`. That compiles with zero overfull boxes while rendering type
at roughly half caption size. The overfull-box count is not a legibility check. Fix width by
narrowing content — abbreviate headers, shorten model labels — not by scaling.

**Height is row count.** No specifier, font size, or `\tabcolsep` value makes a 36-row
double-column table share a page with text. If a table must sit with text, it must lose
rows. That is the whole reason section 2 exists.

---

## 6. Overleaf note

Jorge compiles on Overleaf, where the generated tables are uploaded into a `tables/` folder
next to `Ulu_Lastra.tex` and `\tabledir` is switched (see the commented line near the top of
the preamble). **After regeneration, the new `results/tables/*.tex` must be re-uploaded to
Overleaf**, or the Overleaf build will keep showing the old layout and look like the fix
failed. The specifier change lives in those files, not in the manuscript.

The preamble float parameters themselves are correct as they stand and need no change:

```
\topfraction 0.7 · \bottomfraction 0.9 · \textfraction 0.15 · \floatpagefraction 0.9
\dbltopfraction 0.6 · \dblfloatpagefraction 0.7 · \setcounter{dbltopnumber}{1}
```

---

## 7. Unrelated, but still open in the same file

`Ulu_Lastra.tex` still contains the section headed **"Pending Updates (Internal Tracking,
Remove Before Submission)"** and it still compiles into the PDF. Flagged in both
`org/STATUS_2026-08-11.md` and the 11 Aug session summary. Delete before submission.
