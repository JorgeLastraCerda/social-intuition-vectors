# Manuscript Closeout Corrections

- **Produced:** 2026-08-14 17:20 Europe/Berlin
- **Model:** Nine Gemma, Llama, and Qwen checkpoints
- **Scope:** Table semantics, result counts, mediation inference language, corpus terminology, and appendix order
- **Status:** Complete and locally verified; layout frozen after one final render

## Artifacts

- **Scripts:** `src/build_paper_probe_tables.py`, `src/build_paper_mediation_table.py`, `tests/test_paper_table_builders.py`
- **Inputs:** `results/tables/gemma_scope_causality_gemma3_{12b,27b}_local.csv`, `results/tables/hiring_steering_raw_*.csv`, `results/logs/hiring_probe_vs_human_*.json`, `results/logs/hiring_mediation_*.json`, `data/stimuli/neutral_corpus.jsonl`
- **Outputs:** `results/tables/concept_direction_specificity.tex`, `results/tables/hiring_steering_slopes_9model.tex`, `results/tables/mediation_9model.tex`, `paper/paper/Ulu_Lastra.tex`, `paper/paper/Ulu_Lastra.pdf`

## Summary

This bounded closeout pass resolves the remaining table and prose inconsistencies without adding analyses or changing the abstract. The specificity table now bolds each row's true maximum. The hiring-steering table prints three-decimal $R^2$ values, so Qwen3-14B warmth is shown as 0.799 and remains unbolded under the raw $R^2 \geq 0.800$ rule. The Results list now names all seven qualifying model-axis rows, including Qwen3.6-27B competence.

The human-alignment text now reports significant negative name-level warmth correlations in three models: Llama-3.1-8B, Gemma-4-26B-A4B, and Qwen3-14B. It does not promote these correlations into a claim that each model contains a fully inverted construct. Methods and Results now describe the validation table consistently as four statistics summarizing three of five single-model checks. All neutral-corpus references use the same unit: 1,500 length-matched introductory passages drawn from distinct Wikipedia articles.

## Mediation Inference Boundary

The stored mediation JSON artifacts provide point estimates, unadjusted 95% bootstrap intervals, and interval-exclusion flags. They do not contain a documented multiplicity-adjusted analysis. The manuscript, generated table, Discussion, Limitations, and appendix now state only the supported result: 14 of 36 unadjusted intervals exclude zero, and all path comparisons are exploratory. No corrected winner is identified and no mediation bootstrap was rerun. This report supersedes the correction claim in `paper/2026-07-20_2015_nine_model_mediation.md`; that historical report remains unchanged.

## Appendix Order and Verification

Two targeted page barriers make the physical appendix order match the numbering: S.11 on page 28, S.12 on pages 28 to 30, S.13 on page 31, S.14 on page 32, and S.15 on pages 33 to 34. The final PDF contains 34 pages. Rendered inspection of the changed main-text and appendix pages found no clipping, overlap, or unreadable table content. The LaTeX log contains no overfull boxes, undefined references, unresolved citations, or float-too-large warnings.

The focused paper-table test file passes 16 of 16 tests. The repository `tests/` scope passes 111 tests and retains the same two pre-existing failures in `tests/test_hiring_r4.py`; this closeout introduced no new project-test failure. A root-level discovery run also encountered unrelated missing dependencies under the untracked `ccu/` tree, which was left untouched.

## Freeze Decision

The eight reported issues are closed at their source generators and active manuscript sites, with regression coverage for the failure modes most likely to recur. No further formatting iteration is warranted unless new evidence, a changed analysis artifact, or an explicit publication requirement invalidates one of these acceptance checks.
