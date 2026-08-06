# Six Bias Tables: Race, Gender, and Race × Gender, With and Without Source-Study Breakdown

**Produced:** 2026-08-06 18:28 Europe/Berlin
**Model(s):** All nine checkpoints (Gemma-3-12B, Gemma-3-27B, Llama-3.1-8B, Gemma-4-12B, Gemma-4-26B-A4B, Gemma-4-31B, Qwen3-14B, Qwen3.6-27B, Qwen3.6-35B-A3B)
**Scope:** Extends the marginal and crossed demographic disparity tables with model-versus-human warmth/competence comparability (z-scores), and adds source-study-broken raw-value companions
**Status:** Complete

## Artifacts

- **Scripts:**
  - `src/utils/human_ratings.py` — new `full_distribution_stats` and `add_zscores` helpers (shared z-scoring utility)
  - `src/hiring_r4.py` — `group_statistics` extended to accept any subset of `["race", "gender", "study"]` as `group_cols`, and to aggregate `human_warm`/`human_competent` (previously computed but unused in the joined data)
  - `src/hiring_disparity.py` — additive extension: `human_warm`/`human_competent` and z-score columns added to the existing marginal race/gender output; every pre-existing column verified byte-identical before/after
  - `src/build_paper_probe_tables.py` — four table builders: `build_table2` (rewritten), `build_table2_raw_by_study` (new), `build_table_race_gender` (new), `build_table3` (rewritten, now `longtable`)
- **Inputs:**
  - `results/tables/hiring_audit_<label>.csv` (9 files, unchanged)
  - `results/tables/hiring_disparity_<label>.csv` (9 files, additively extended)
  - `data/raw/SocialPerceptions-Predict-Callback-main/0_data/ratings/names/df_all.csv`
  - `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`
- **Outputs:**
  - `results/tables/hiring_group_r4_<label>.csv` (9 files, additively extended with `human_warm_mean`/`human_competent_mean`/z-score columns)
  - `results/tables/hiring_disparity_marginal_9model.tex` (main text)
  - `results/tables/hiring_disparity_race_gender_9model.tex` (main text, new)
  - `results/tables/hiring_disparity_marginal_raw_9model.tex` (appendix, new)
  - `results/tables/hiring_disparity_crossed_9model.tex` (appendix, rewritten as `longtable`)
- **Figures:** none (tables only)

## What was added, and why

The existing marginal (race, gender) and crossed (race × gender × study)
disparity tables already showed each group's raw model warmth/competence
alongside the real human callback rate, but never the human warmth/competence
ratings behind that callback rate, and never a way to compare model and human
warmth/competence directly, since the two live on different scales: human
ratings are 0-100 Likert-style averages, model warmth/competence are
unbounded raw residual-stream projections (tens of thousands in magnitude).
This meant the paper could show a group's model-side lean and a group's
human-side callback outcome, but never directly answer "does the model's
internal representation of warmth or competence lean the same direction as
how humans actually perceive that group?"

The fix has two parts:

1. **Add human warmth/competence to every group table.** The data was
   already computed (`hiring_audit_<label>.csv` has had `human_warm`/
   `human_competent` per name since the flake_leasure fix), just never
   aggregated into the group-level tables. `hiring_r4.py`'s joined
   `matched` DataFrame already carried these columns through; extending
   `group_statistics`'s aggregation to include them, and doing the same for
   `hiring_disparity.py`'s marginal computation, was additive in both cases.
2. **Standardize both sides to z-scores.** Mean/SD are computed once per
   model over the full 282-name audit (model warmth/competence) and once
   globally over the full 282-name rating set (human warmth/competence, the
   same for every model), then every group mean is expressed as SD above or
   below that overall mean. This mirrors the callback-margin standardization
   already used elsewhere in the paper ("standardized by the within-model
   standard deviation... so that models with different logit scales can be
   compared on equal footing") and turns "is the model parallel to human
   perception" into a direct sign/magnitude comparison instead of an
   apples-to-oranges reading of two different unit systems.

Six tables result, split by whether z-scores (comparability) or raw values
(reference, broken down by source study) are the point:

| Table | Grouping | Content | Location |
|---|---|---|---|
| `tab:disparity_marginal` | race, gender (marginal) | z-scored | main text |
| `tab:disparity_race_gender` | race × gender (crossed) | z-scored | main text |
| `tab:disparity_marginal_raw` | race × study, gender × study | raw | appendix (`longtable`) |
| `tab:disparity_crossed` | race × gender × study | raw | appendix (`longtable`) |

The crossed race × gender × study table (existing, previously a plain
`table` environment) no longer fits a single page once human warmth/competence
columns were added on top of the existing model columns (~100 rows across
nine models); it and its new raw-by-study sibling both switched to
`longtable`, which lets a table break across pages with a repeated header
rather than overflowing.

## What the z-scored tables show

Reading a row means comparing the sign and rough magnitude of the model
column against the human column. A concrete example, Gemma-3-12B (marginal,
main text table):

| Group | Model warmth/competence ($z$) | Human warmth/competence ($z$) | Read |
|---|---|---|---|
| Black | $-0.44$ / $-0.43$ | $-0.30$ / $-0.58$ | parallel (both negative) |
| White | $+0.45$ / $+0.44$ | $+0.21$ / $+0.15$ | parallel (both positive) |

Llama-3.1-8B, by contrast, shows the opposite pattern for race:

| Group | Model warmth/competence ($z$) | Human warmth/competence ($z$) | Read |
|---|---|---|---|
| Black | $+0.45$ / $+0.24$ | $-0.30$ / $-0.58$ | anti-parallel (opposite sign) |
| White | $-0.38$ / $-0.22$ | $+0.21$ / $+0.15$ | anti-parallel (opposite sign) |

This is a table-level restatement of the warmth anti-alignment already
documented in Limitations ("Inverted warmth construct in two models",
Llama-3.1-8B $\rho=-0.287$, Qwen3-14B $\rho=-0.178$) and the Methods
probe-vs-human bullet, now visible directly at the demographic-group level
rather than only as a name-level correlation coefficient.

The human warmth/competence z-scores are, correctly, identical across every
model for the same group (e.g. Black is always $-0.30$ / $-0.58$): they come
from the same human rating data regardless of which model's table they
appear in, since human perception does not depend on which language model is
being evaluated. This is a useful sanity check on the table-generation code,
confirmed by inspection rather than assumed.

## Verification

- `hiring_disparity.py`: `axis`, `group`, `n`, `model_callback_margin`,
  `model_warmth`, `model_competence`, `human_callback` confirmed byte-identical
  before/after the additive extension (`pd.Series.equals`, all True) for
  Gemma-3-12B; mediation indirect effects for all nine models (unaffected,
  same computation path) match the values already reported in the
  corresponding per-model findings reports.
- `hiring_r4.py`: `group_statistics` with `group_cols` in
  `[["race"], ["gender"], ["race","gender"], ["race","study"],
  ["gender","study"], ["race","gender","study"]]` all sum to the same total
  `n_names` (246) for a given model, confirming the flexible grouping does
  not drop or duplicate rows regardless of which columns are grouped on.
- `build_table3`'s regression gate (recomputed `model_margin_mean` per
  (race, gender, study) cell against `hiring_group_r4_<label>.csv`) passed
  for all nine models after every change in this step.
- Manuscript rebuilds clean: `latexmk -pdf -interaction=nonstopmode`,
  `grep -niE "overfull|undefined|float too large"` on the log returns
  nothing, 23 pages (up from 20). Rendered pages for both main-text tables
  and both appendix `longtable`s visually inspected via `pdftoppm`;
  continuation headers ("Tab. S.2 continued" / "Tab. S.3 continued") render
  correctly on subsequent pages.

## Caveats

- **Group sizes are unchanged from the existing marginal/crossed tables**
  (race: 47 Black / 180 White; the crossed and study-broken tables retain
  the same small-cell caveats already documented, e.g. 9-name single-study
  cells). This step adds columns and standardization, not new matching.
- **Marginal (race-only, gender-only) tables intentionally keep
  `hiring_disparity.py`'s existing, looser first-name-only join** (269 of
  282 names matched), while the crossed and study-broken tables use
  `hiring_r4.py`'s stricter (name, study) join (246 observations / 186
  names). This was a deliberate choice to avoid silently changing the
  marginal table's already-published counts, not an oversight; the two
  joins' outputs are not directly interchangeable, and a table combining
  both would need to say so explicitly.
- **Z-scores are computed against the full 282-name distribution, not the
  matched subset.** This matches the "how unusual is this group relative to
  the whole rated-name population" framing, but means the z-score's
  denominator is not the same population as the table's own row counts.
