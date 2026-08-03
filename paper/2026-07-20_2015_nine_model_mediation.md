# Nine-Model Bootstrap Mediation Added to the Manuscript

- **Produced:** 2026-07-20 20:15 Europe/Berlin
- **Model:** Nine Gemma, Qwen, and Llama checkpoints
- **Scope:** Extend the four-model bootstrap-mediation analysis (name group → probe
  score → callback margin) to the five newer checkpoints, and add the combined
  nine-model result as a manuscript appendix table with a main-text forward reference
- **Status:** Complete

## Artifacts

- **Scripts:** `src/build_paper_mediation_table.py` (new); reuses
  `src/hiring_disparity.py::bootstrap_mediation` output format only, no new bootstrap run
- **Inputs:** `results/logs/hiring_mediation_<label>.json` for all nine labels
  (`gemma3_12b`, `gemma3_27b`, `llama31_8b`, `gemma4_12b`, `gemma4_26b_a4b`,
  `gemma4_31b`, `qwen3_14b`, `qwen36_27b`, `qwen36_35b_a3b`)
- **Outputs:** `results/tables/mediation_9model.tex` (appendix Table S.4)
- **Figures:** none (table only; extending Figure 10 / `fig19_hiring_mediation_forest`
  to nine models is deferred, see Caveats)

## What was added

No new bootstrap resampling was run. The five newer checkpoints already had
`results/logs/hiring_mediation_<label>.json` on disk, dated 2026-07-18 to
2026-07-19, produced by the same CPU-only `src/hiring_disparity.py::bootstrap_mediation`
procedure used for the original four models (`n_boot=5000`, `seed=20260527`,
Baron–Kenny/Preacher–Hayes bootstrap). This step only consolidates all nine
models' results, which were sitting unused in `results/logs/`, into a manuscript
table, exactly the same kind of consolidation as the probe-vs-human tables added
in `paper/2026-07-20_1935_probe_human_result_tables.md`.

**Why no GPU run was needed.** `src/hiring_disparity.py` loads no model; it only
joins `results/tables/hiring_audit_<label>.csv` (already produced during each
model's GPU audit run) against the static published human-callback data and runs
OLS + bootstrap in numpy/scipy. The five newer checkpoints' audit CSVs already
existed, so their mediation JSONs were a byproduct of runs already completed for
other purposes; nobody had looked at the mediation numbers specifically until now.

**New table (`tab:mediation_9model`, Table S.4, appendix).** Same four rows per
model as the main-text version (race×warmth, race×competence, gender×warmth,
gender×competence), for all nine models, 36 tests total. Columns: model, grouping,
probe, standardized path coefficients `a` and `b`, indirect effect `a*b`, and its
95% bootstrap CI, with a dagger marking CIs that exclude zero.

**Significant paths in the five new models** (95% CI excludes zero, uncorrected):

| Model | Path | IE | 95% CI |
|---|---|---|---|
| Gemma-4-12B | race → competence | −0.150 | [−0.295, −0.019] |
| Gemma-4-26B-A4B | gender → warmth | +0.038 | [+0.007, +0.078] |
| Gemma-4-26B-A4B | race → competence | +0.131 | [+0.060, +0.231] |
| Gemma-4-31B | race → competence | −0.230 | [−0.483, −0.049] |
| Qwen3.6-27B | race → competence | −0.049 | [−0.103, −0.011] |
| Qwen3.6-27B | gender → competence | −0.123 | [−0.206, −0.061] |
| Qwen3.6-35B-A3B | race → warmth | −0.103 | [−0.169, −0.046] |
| Qwen3.6-35B-A3B | gender → warmth | +0.064 | [+0.027, +0.113] |
| Qwen3.6-35B-A3B | gender → competence | +0.029 | [+0.001, +0.067] |

14 of the 36 combined tests are significant at the uncorrected 95% level (up from
5 of 16 in the original four-model subset). Under a Bonferroni threshold across
all 36 tests (α = 0.05/36), only the pre-existing Llama-3.1-8B race–warmth path
survives; every other entry, old or new, remains suggestive rather than confirmed.

## Bearing on the steerability paradox

The main text builds a steerability paradox on the original four models: Gemma-3-12B
is the most concept-steerable model in that subset yet shows no significant
mediation path, while Llama-3.1-8B is the least steerable yet shows the strongest
one (IE = +0.190). Both Gemma-3 checkpoints were null on all eight tests, which the
main text reads as Gemma's cleanly isolable representations bypassing the hiring
decision rather than driving it.

The Gemma-4 results complicate that reading. All three Gemma-4 checkpoints show a
significant race–competence indirect effect, something no Gemma-3 checkpoint showed
on either axis. Qwen3.6 shows a comparable spread of significant warmth and
competence paths across both checkpoints, consistent with the mediation profile
already observed in Qwen3-14B. Read together, the absence of hiring-level mediation
may be a property specific to the Gemma-3 generation rather than a general property
of highly steerable architectures. Per the user's decision, the main-text
steerability-paradox claim is kept as written, since it is already scoped to "the
original four-model mediation subset," and this qualification is added only in the
appendix alongside the new table, with a one-sentence forward pointer from the
main-text mediation paragraph.

## Manuscript changes

- `paper/paper/Ulu_Lastra.tex`: main-text mediation paragraph (after "All eight
  tests for the two Gemma models are non-significant.") gained one forward-reference
  sentence to `\autoref{tab:mediation_9model}`; a new appendix subsection
  "Bootstrap Mediation, All Nine Models" was added after the crossed race × gender
  table (`tab:disparity_crossed`), containing the qualifying two-paragraph discussion
  above and `\input{../../results/tables/mediation_9model.tex}`.
- No changes to the main-text "sixteen tests" language or Figure 10
  (`fig19_hiring_mediation_forest`), which stay scoped to the original four models.

## Verification

- `python -m src.build_paper_mediation_table --config config/config.yaml` (run in a
  scratchpad venv with pandas/scipy/pyyaml, since system Python lacks them, same
  workaround as the probe-table report) produced `mediation_9model.tex` with 36 rows
  and 14 significant markers, matching the count computed independently from the raw
  JSONs.
- First `latexmk -pdf -interaction=nonstopmode -halt-on-error` build compiled cleanly
  (22 pages, no undefined references, no overfull hboxes) but rendered the new
  forward-reference as "(section  of the Supplementary Materials)" with a blank
  number: `\autoref` cannot number a starred `\subsection*`. Fixed by pointing the
  `\autoref` at the table label (`tab:mediation_9model`) instead of the subsection,
  since the appendix table already carries a real `S.4` number via the existing
  `\renewcommand\thetable{S.\arabic{table}}` in that section, and removed the
  now-unused subsection label.
- Rebuilt; confirmed via `pdftoppm` page renders (main-text page 9, appendix pages
  21–22) that the sentence now reads "(Tab. S.4 in the Supplementary Materials)" as
  a resolved hyperlink, the appendix table renders within column width with no
  clipping, and no overfull-hbox or undefined-reference warnings remain in the log.
  `pdftotext` extraction is broken for this document's fonts (as previously noted in
  the probe-table report), so verification was done by reading rendered page images.
- Anti-formulaic self-check on the two new appendix paragraphs and the one new
  main-text sentence: no em-dashes; paragraph openers vary ("The main text reports…"
  / "The newer checkpoints complicate…"); no signal-only transitions; no repeated
  three-times causal template.

## Caveats

- The forest-plot figure (`fig19_hiring_mediation_forest`) was deliberately **not**
  extended to nine models in this pass; the user asked to defer that ("sonra
  bakarız"). `paper/figures/generate_figures.py::fig19_hiring_mediation_forest`
  already accepts an arbitrary list of `hiring_mediation_<label>.json` paths, so
  extending it later is a config change, not new code.
- The 36-test family is presented uncorrected in the appendix caption, with the
  α = 0.05/36 Bonferroni threshold stated explicitly; no bootstrap was re-run at a
  different confidence level.
- As in the four-model case, model warmth/competence path coefficients (`a`) are on
  each model's own standardized scale and are not comparable in magnitude across
  models; only the sign and CI-exclusion of each row are interpretable across models.
