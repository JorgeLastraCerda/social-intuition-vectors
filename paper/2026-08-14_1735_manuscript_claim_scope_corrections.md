# Manuscript Claim-Scope Corrections

- **Produced:** 2026-08-14 17:35 Europe/Berlin
- **Model:** Nine Gemma, Llama, and Qwen checkpoints
- **Scope:** Competence human-alignment counts in Introduction and Discussion, and construct-scope language in the Discussion opener and Conclusion
- **Status:** Complete and locally verified; no analysis rerun and no table regenerated

## Artifacts

- **Scripts:** `tests/test_paper_table_builders.py`
- **Inputs:** `results/logs/hiring_probe_vs_human_{gemma3_12b,gemma3_27b,llama31_8b,gemma4_12b,gemma4_26b_a4b,gemma4_31b,qwen3_14b,qwen36_27b,qwen36_35b_a3b}.json`
- **Outputs:** `paper/paper/Ulu_Lastra.tex`, `paper/paper/Ulu_Lastra.pdf`
- **Figures:** None; this pass changed prose only.

## Summary

This pass corrects two false statements about the competence correlation and
narrows two universal-scope claims that outran the paper's own construct-validity
argument. No table builder ran and no generated table changed, because all four
edits are hand-written manuscript prose.

## Competence Alignment Counts

The Introduction claimed that competence "tracks human ratings in every
checkpoint" and the Discussion claimed its correlation is "positive in all nine,
reaching significance in all but one." Both are wrong. Counts derived directly
from the nine `hiring_probe_vs_human_*.json` logs, using Spearman $\rho$ and
$p < 0.05$:

| Axis | Positive and significant | Significant negative | Nonsignificant |
|---|---|---|---|
| Competence | 7 | 0 | 2 (Llama-3.1-8B $\rho=-0.058$, Gemma-4-26B-A4B $\rho=-0.044$) |
| Warmth | 5 | 3 | 1 (Gemma-4-12B $\rho=+0.009$) |

The Discussion sentence carried two independent errors: competence is not
positive in all nine, and the significant count is seven of nine rather than
eight. Results and Limitations already stated this correctly, so the defect was
confined to two passages left behind by the four-model to nine-model expansion.
Both now read "seven of the nine," with the Discussion naming the remaining two
as small, negative, and nonsignificant.

The surrounding claim that competence is the more orderly dimension survives the
correction: competence never turns significantly negative, while warmth does so
in three checkpoints. The split-half clause is also unaffected, since competence
reconstructs more stably than warmth in all nine models.

## Construct Scope

Two sentences asserted that warmth and competence are present in every model,
which contradicts the Discussion paragraph "Naming a Direction Is a Claim We
Make." That paragraph argues the label comes from the stimuli rather than from
the model, and that the construct governing model behavior on our stimuli may
not be the construct the label names. The direction-specificity result points
the same way: the dense target is the largest intervention in only one of four
model-axis rows, and the calibrated random control's interval contains the
dense-target effect in all three Gemma-4 checkpoints.

Both sentences now describe what was demonstrated rather than what it might
mean. The Discussion opener reports that warmth- and competence-associated
contrastive directions are recoverable in all nine models while what they
correspond to cannot be assumed identical. The Conclusion states that a
direction separating our warmth and competence stories exists in all of these
systems, and that what it corresponds to appears to be a property of the
individual model.

Three related passages were deliberately left unchanged. The abstract already
describes a procedure that "recovers warmth and competence directions across
models." The Introduction sentence "We test whether warmth and competence are
encoded this way" poses a research question in the vocabulary of the Stereotype
Content Model rather than asserting a result. The Discussion sentence at the
head of the section is explicitly hedged with "Read on its own" and is
complicated by the paragraph that follows it.

## Regression Coverage

Two tests were added to `tests/test_paper_table_builders.py`.
`test_manuscript_human_alignment_counts_match_source_logs` recomputes the 7/2 and
5/4 splits from the source JSON before asserting the prose, so a future rerun
that changes a correlation fails the test rather than silently desynchronizing
the manuscript again. `test_manuscript_avoids_unqualified_encoding_claims` pins
both the removed and the replacement construct-scope phrasings.

## Verification

The focused file passes 18 of 18 tests. The repository `tests/` scope passes 113
and retains the same two pre-existing failures in `tests/test_hiring_r4.py`, a
pandas `KeyError: 'study'` unrelated to manuscript prose. Two `pdflatex` passes
produce a 34-page PDF with no overfull boxes, undefined references, unresolved
citations, or float-too-large warnings. Appendix tables S.1 through S.15 remain
in strictly increasing page order, which was re-checked because the Conclusion
edit changes paragraph length.

## Note on the Turkish Translation

`paper/paper/Ulu_Lastra-tr.tex` was last modified on 2026-07-15 and still
describes four architectures. It predates the nine-model expansion and is not
the active manuscript under `AGENTS.md`, so it was left untouched. It will need
a full retranslation, not a patch, if it is ever revived.
