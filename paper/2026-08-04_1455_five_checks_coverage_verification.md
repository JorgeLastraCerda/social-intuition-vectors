# Five-Checks Coverage Verification Across Nine Models

- **Produced:** 2026-08-04 14:55 Europe/Berlin
- **Model:** Nine Gemma, Qwen, and Llama checkpoints
- **Scope:** Systematically confirm that all five probe-validation checks
  described in `paper/paper/Ulu_Lastra.tex` (Methods, five-checks list:
  5-fold CV, topic-holdout, Cohen's d with random-direction null, split-half
  cosine stability, cross-axis classification) were actually run for every
  one of the nine models, not just described in prose
- **Status:** Complete; one coverage gap found and closed

## Artifacts

- **Scripts:** `paper/figures/generate_figures.py` (`fig2_random_baseline`,
  pre-existing function, no new code written for this report)
- **Inputs:** `data/processed/concept_vectors_gemma3_27b/` (for the fix);
  `results/logs/validate_probes_*.json`, `results/logs/split_half_stability_*.json`,
  `paper/figures/*/fig2_random_baseline.png` (read-only, for the verification pass)
- **Outputs:** `paper/figures/gemma3_27b/fig2_random_baseline.{png,pdf}` (new)
- **Figures:** `paper/figures/gemma3_27b/fig2_random_baseline.png` (new; see
  above)

## Why this report exists

This session's manuscript audit already found and closed two coverage gaps
in the five-checks list: split-half cosine stability had no code anywhere in
the repository or its git history (`paper/2026-08-04_1432_split_half_stability_reproduced.md`),
and cross-axis classification's zero-shot variant was only logged for five
of nine models (`paper/2026-06-20_1337_cross_axis_metric_correction.md`,
2026-08-04 update). After closing both, the user asked for a systematic
check confirming every check now genuinely covers all nine models, rather
than trusting that no further gaps remained.

## Method

For each of the nine models, checked for the presence of the specific field
or artifact each check produces:

| Check | Artifact checked |
|---|---|
| 5-fold CV | `cv_mean` in `results/logs/validate_probes_*.json` (`warmth`/`competence`) |
| Topic-holdout | `topic_cv_mean` in the same file |
| Cohen's d (point estimate) | `cohens_d` in the same file |
| Cohen's d (random-direction null) | `paper/figures/<label>/fig2_random_baseline.png` |
| Split-half cosine | `results/logs/split_half_stability_<label>.json` |
| Cross-axis (recalibrated) | `cross_warmth_on_competence_cv` in `validate_probes_*.json` |
| Cross-axis (zero-shot) | `cross_warmth_to_competence_topic_transfer_mean` in the same file |

## Finding

Every check was present for all nine models except one: **Gemma-3-27B had no
Cohen's d random-baseline null figure.** No `paper/figures/gemma3_27b/`
directory existed at all, and `paper/README.md`'s figure inventory had no
z-score entry for it, unlike all eight other models (which each have a
`fig2_random_baseline.{png,pdf}` under their own subdirectory or, for the
default model, at the figures root). All other checks, for all nine models,
were confirmed present with no further gaps.

## Fix

Ran the existing figure-generation function directly against Gemma-3-27B's
already-extracted vectors, no GPU or cluster access required:

```
python3 paper/figures/generate_figures.py --fig 2 \
    --vec-dir data/processed/concept_vectors_gemma3_27b \
    --out-dir paper/figures/gemma3_27b
```

Result: warmth $d = 2.95$ ($z = 4.3$), competence $d = 3.27$ ($z = 4.5$); 0 of
1,000 random directions exceeded either. This is consistent with the
manuscript's own Limitations text, which separately cites a layer-sweep
Gemma-3-27B warmth Cohen's $d$ of 2.95 at the selected layer
(`paper/paper/Ulu_Lastra.tex`), giving an independent cross-check that the
newly generated null-comparison figure is correct. `paper/README.md`'s
figure inventory was updated with the new entry.

## Coverage after this report (all nine models, all five checks)

| Check | Coverage |
|---|---|
| 5-fold CV | 9/9 |
| Topic-holdout | 9/9 |
| Cohen's d (point estimate) | 9/9 |
| Cohen's d (random-direction null) | 9/9 (was 8/9) |
| Split-half cosine | 9/9 (was 0/9 before this session) |
| Cross-axis (recalibrated) | 9/9 |
| Cross-axis (zero-shot) | 9/9 (was 5/9 before this session) |

## Next Steps

None outstanding for check coverage. Separately, none of the five checks'
nine-model numeric results are currently narrated in the manuscript's
Results section, which is marked in-source as a placeholder pending a
collaborative redesign (`paper/paper/Ulu_Lastra.tex`, comment at the top of
`\section*{Results}`); surfacing these numbers there is a distinct, not yet
scheduled, decision.
