# Manuscript Work, 10-11 August 2026: What Changed and Why

- **Timestamp:** 2026-08-11 23:00 Europe/Berlin
- **Scope:** `paper/paper/Ulu_Lastra.tex`, `paper/figures/style.py`,
  `src/build_paper_probe_tables.py`, `src/build_paper_mediation_table.py`
- **Purpose:** a single place to understand the manuscript changes made over two days,
  for Emre or anyone else picking the paper up. Per-step detail is in
  `step_logs/STEP_LOG.md`, entries dated 2026-08-10 and 2026-08-11.

---

## Where the manuscript stands

Every section now has prose. Introduction 588 words, Background 980, Methods 3,118,
Results 2,034, Discussion 2,540. No analysis or GPU work is outstanding. What remains is
presentation of tables and figures, one internal section that must be deleted before
submission, and proofreading.

---

## Structural changes

### The section that was dissolved

"Adapting the Method to Hiring" no longer exists. Background and Methods were duplicating
each other through it: the behavioral-audit citations appeared in both, and the 282-name
stimulus set was described twice.

- "Why Warmth and Competence" and "The Link to Hiring" merged into Background
- "Concept Stories Without Identity" merged into the Methods "Story Preparation" paragraph,
  which had previously referred to identity controls "described above" and would otherwise
  have dangled
- "Applicant Names" was **deleted**, not moved. Methods already covered the same material,
  and the deleted paragraph carried a **stale count of 149 names**. Methods and Results both
  report 246 name-study observations across 186 distinct names, which is correct under
  name-plus-study matching. No occurrence of 149 remains.

### Results re-ordered

Order is now: Vector Validation, Steering the Concept Vectors, Alignment with Human
Ratings, Steering Hiring Decisions, Group-Level Disparity and Mediation.

Previously the two divergence findings were separated: warmth inversion sat in the final
paragraph while hiring-steering divergence sat two paragraphs earlier. They are now
adjacent, and the section closes on hiring, which is the paper's title topic. The split
between the last two paragraphs is at the point where the subject changes from what the
probe reads off *names* to what happens to *callbacks*, which is also why mediation now
follows hiring steering rather than preceding it.

Paragraphs were moved whole. Only two transition sentences were written; the word count
moved from 2,028 to 2,088, which is exactly those two additions.

### Methods shortened by relocation, not compression

The seven probe-validation checks were moved **verbatim** into a new appendix subsection,
`appx:probe_checks`, and replaced in the body by a compact summary that still reports every
result. Methods fell from 3,492 to 3,118 words.

They were relocated rather than rewritten because their plain-language register is the
product of the deliberate rewrite pass logged in Steps 19 to 28 on 2026-08-04 and 2026-08-05.
Compressing them in place would have reversed a logged co-author decision without
consultation. **If Methods needs to be shorter still, the same move-not-rewrite approach
applies to the six-direction steering inventory.**

### Two tables moved to the appendix

- `tab:disparity_race_gender` is now `appx:crossed_disparity`. It was tested before removal
  rather than cut on page cost: an interaction between race and gender is present in eight
  of nine checkpoints, but the raw margins are not comparable across models (they range from
  about -0.2 to about 25.8), the smallest cells hold 28 names, and no intervals were
  computed. A Future Work sentence points at it.
- `tab:mediation_9model` is now `appx:mediation`. **`fig19_hiring_mediation_forest` was not
  put in its place**, because that figure is stale: it shows four models across sixteen
  rows, predating the nine-model expansion, and its title contains broken glyphs. All nine
  mediation logs exist, so it can be rebuilt, but a thirty-six-row forest plot would occupy
  roughly the footprint of the table it replaced. A compact strip plot in the style of
  `paper/figures/fig6_cross_model_agreement.py` would suit better.

---

## Content corrections

### The stimulus prompt is now documented

`tab:story_prompt` had been compiling into the PDF as a placeholder reading "Work in
progress". It now carries the system prompt, the per-story prompt as issued, the four
condition descriptions, and the four forbidden-word lists.

The caption discloses three things a reader would otherwise find confusing:

1. The corpus was produced **interactively with Claude Opus 4.8**, not by a batch script.
2. Rules 5 and 6, covering the nameless gender-neutral protagonist and holding the off-axis
   dimension at an ordinary level, were **added during generation** after an audit of an
   early pilot found male Anglo-named protagonists concentrated in the low conditions. This
   is corroborated by `data/stimuli/STIMULI_TRACKER.md`, which records batch 1 as superseded
   for exactly that reason.
3. `src/generate_stimuli.py` holds a batch template for a larger corpus that was specified
   but never run, and it **differs from the text used**: it requests 120 to 180 words and
   contains no identity instruction at all. Realized lengths were 88 to 144 words, mean 100.

A limitation, "The corpus cannot be regenerated exactly", was added. Its closing point is
the important one: the 200 stories are released in full and every downstream result derives
from that fixed text, so the analyses are reproducible even though the generation is not.

**The em-dashes inside that table are quoted stimulus material.** A source comment marks
them so they are not normalized to the manuscript's no-em-dash style rule.

### Meta-commentary removed

Two sentences explained a formatting decision or a workflow state to the reader rather than
reporting the science. Both were removed, and a repository-wide sweep found no others.
Manuscript prose should not describe how the paper is organized or what remains to be run.

### Broken cross-references

`\autoref` on a label attached to an unnumbered `\subsection*` resolves against the
enclosing numbered counter and renders as the bare word "section". This affected all three
appendix labels. They now use `\nameref`, which prints the subsection title.

**Rule for the future: appendix subsections are unnumbered, so reference them with
`\nameref`, never `\autoref`.**

### Figures that were never cited

Three figures had no citation in the body: `fig:paper_figure1_axis_arrows` and
`fig:paper_figure2_layer_emergence` had none anywhere, and
`fig:hiring_bidirectional_examples` was cited only from the appendix, which is why it
appeared in the list of figures but nowhere in the text. All three now carry a sentence in
Results. Both older figures were kept at the authors' request after confirming each still
supports a claim.

### Citations added

`shrm2025talent`, `oreopoulos2011why`, `correll2007getting` and `neumark2019older` were
added to Background, framed as the kind of correspondence study the benchmark meta-analysis
aggregates rather than as a list. Distinct cited sources now number 28.

---

## Layout and build

### Table and figure placement

Floats were rendering two and three to a page with text pages ending early. **Five causes,
all of which had to be fixed:**

1. Two figures carried `[p]`, which *forbids* sharing a page with body text.
2. Six `\clearpage` barriers flushed the entire pending float queue at once.
3. The preamble permitted a 95-percent-float page: `\textfraction{0.05}`,
   `\dbltopfraction{0.9}`, `\dbltopnumber{3}`.
4. Several floats sat adjacent in the source. LaTeX queues in source order, so adjacent
   floats emit together regardless of specifier.
5. The generated tables carried `[tp]`, and the `p` still permitted shared float pages.

Fixes applied: `[p]` to `[t]` on figures; barriers removed with containment re-verified from
the `.aux`; float-area limits retightened to `\textfraction{0.15}`, `\topfraction{0.7}`,
`\dbltopfraction{0.6}`, `\dbltopnumber{1}`; Results float order rebuilt programmatically so
each float follows the paragraph that first cites it, at most one per paragraph boundary;
and all eight `table*` specifiers changed to `[t]` in both builders.

Result: 38 pages to 36, pages carrying two or more floats from four to zero, emptiest
Results page from 192 to 396 words, zero overfull boxes throughout.

### Figure typography

`paper/figures/style.py` changed from Helvetica sans-serif at 11pt to serif at 9pt with
`mathtext.fontset: cm`, so figure text matches the manuscript's Computer Modern body font.

### Two changes need regeneration

**The `style.py` change and the `[t]` table specifiers are both invisible until the figures
and tables are regenerated.** If a build still shows sans-serif figure text or stacked
tables, that is the reason, not a failed fix.

---

## Cautions carried forward

- **`\resizebox` produces a false pass.** Scaling a table to `\columnwidth` compiles with
  zero overfull boxes while rendering type at roughly half caption size. Narrow the content
  instead: abbreviate headers and labels. Overfull-box count is not a legibility check.
- **Table height is set by row count, not formatting.** Do not answer "the tables are too
  large" with a smaller font or more scaling.
- **Containment must be re-checked if floats are added back to the body.** The barriers were
  originally added because floats drifted up to nine pages from their prose. Removing them
  is safe at the current float count, which was verified, not assumed.
- **The human race gap in the benchmark is near zero** (Black 0.183, White 0.171), very
  slightly favouring Black-signalling names. Any prose describing model disparities as
  running "opposite to the human benchmark" is inconsistent with the data. The defensible
  framing for race is amplification of a near-zero gap; genuine directional disagreement
  exists for gender.

---

## Outstanding

The action list at the top of
`paper/idea-notes/2026-08-10-1651-table-restructure-spec.md` carries nine numbered items
with current status. Items 1, 2, 3 and 4 and parts of 8 and 9 remain.

Separately, `paper/idea-notes/2026-08-10-2130-related-work-to-verify.md` records two papers
close enough to this work to affect the contribution claim, neither yet read in full or
cited.

**Before submission:** the section headed "Pending Updates (Internal Tracking, Remove
Before Submission)" is still in the manuscript and still compiles into the PDF.
