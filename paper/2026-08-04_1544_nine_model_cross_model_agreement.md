# Cross-Model Spearman Agreement, Extended to All Thirty-Six Model Pairs

- **Produced:** 2026-08-04 15:44 Europe/Berlin
- **Model:** Nine Gemma, Qwen, and Llama checkpoints (all $\binom{9}{2}=36$ pairs)
- **Scope:** Compute the "Cross-model Spearman agreement" check described in
  `paper/paper/Ulu_Lastra.tex` (Methods) for every pair of the nine models,
  after finding that only 4 of the 36 possible pairs (all within-family) had
  ever been computed
- **Status:** Complete

## Artifacts

- **Scripts:** `src/validate_cross_model_agreement.py` (pre-existing, no new
  code written)
- **Inputs:** `data/processed/concept_vectors*/{warmth_vec,competence_vec,
  X_high_warmth,X_low_warmth,X_high_competence,X_low_competence}.npy` for
  all nine models
- **Outputs:** `results/tables/cross_model_agreement_9model.csv` (new, 72
  rows: 36 pairs $\times$ 2 axes); pre-existing partial outputs
  `results/tables/probe_story_agreement_gemma4.csv` (3 Gemma-4-internal
  pairs) and `results/tables/qwen36_cross_model_agreement.csv` (1 Qwen3.6
  pair) are now subsumed by the new file and can be treated as historical
  intermediate artifacts
- **Figures:** none

## Why this report exists

While rewriting the "Cross-model Spearman agreement" bullet in
`paper/paper/Ulu_Lastra.tex` for plain-language clarity, the user asked
whether this check had actually been run for all nine models. Searching
`results/tables/` for agreement outputs found only two files, together
covering just 4 of the 36 possible model pairs: three pairs among the three
Gemma-4 checkpoints (`probe_story_agreement_gemma4.csv`) and one pair
between the two Qwen3.6 checkpoints (`qwen36_cross_model_agreement.csv`).
The four original models (Gemma-3-12B, Gemma-3-27B, Qwen3-14B, Llama-3.1-8B)
were not represented in any pair, and no pair crossed model families (e.g.
no Gemma vs. Qwen comparison existed at all). This is a materially larger
gap than the other coverage gaps found earlier this session (split-half:
0/9 models; cross-axis zero-shot: 5/9 models), since here only about 11%
(4/36) of the relevant comparisons existed.

## Method

`src/validate_cross_model_agreement.py::compute_agreement_records` takes any
number of model vector directories and labels, and for every pair projects
each of the 200 stories onto both models' own (unit-normalized) direction
vectors, then computes Spearman's $\rho$ between the two resulting sets of
per-story scores in two ways: `overall_rho` (all 200 stories pooled) and
`within_condition_rho` (rank-transformed within each of the four conditions
before pooling, removing the trivial contribution of the high-versus-low
split itself). Ran once with all nine model directories and labels supplied
together; the script's nested loop automatically produced all 36 pairs for
both axes. No GPU or cluster access was required, only already-extracted
`.npy` arrays already committed to git.

## Results

Across all 36 pairs:

| Axis | Metric | Min | Max | Mean |
|---|---|---:|---:|---:|
| Warmth | overall $\rho$ | 0.741 | 0.978 | 0.862 |
| Warmth | within-condition $\rho$ | 0.095 | 0.833 | 0.416 |
| Competence | overall $\rho$ | 0.782 | 0.992 | 0.897 |
| Competence | within-condition $\rho$ | 0.204 | 0.899 | 0.511 |

`overall_rho` is uniformly high (0.74–0.99) across every pair, including
every cross-family comparison (e.g. Gemma vs. Qwen, Gemma vs. Llama), not
just within-family pairs. Some of that agreement is a comparatively easy
target, since all nine models trivially agree that high-condition stories
score above low-condition stories; `within_condition_rho`, which removes
that trivial contribution, is lower and more variable (0.10–0.90) but
remains positive in every single pair for both axes, meaning no pair of
models disagrees on the fine-grained ranking of stories within a condition.
The full 72-row table is in `results/tables/cross_model_agreement_9model.csv`.

## Interpretation

Every one of the 36 possible model pairs, including every pairing across
architecture families, shows positive agreement on both metrics. This
supports the manuscript's claim that the nine models are converging on a
shared cross-architecture construct rather than nine unrelated ones. The
`within_condition_rho` values (mean 0.42 warmth, 0.51 competence) are the
more informative number if this check's numeric result is ever surfaced in
the manuscript, since `overall_rho` alone is inflated by the easy high/low
separation that every model gets right by construction.

## Caveats

- `within_condition_rho` values range fairly widely (0.10 to 0.90); the
  weakest pairs are not identified by architecture family in this report.
  If this check's results are added to the manuscript, a per-pair or
  per-family breakdown (not just the pooled summary above) would be more
  informative than the aggregate min/max/mean given here.
- This reuses the pre-existing, already-reviewed script exactly as written;
  no new methodological choices were made.

## Next Steps

- Optional: if the user wants this check's numeric result surfaced in the
  manuscript, the `within_condition_rho` summary above is the recommended
  number to cite, with the caveat above about per-pair variability.
- Optional: the two now-subsumed partial CSVs
  (`probe_story_agreement_gemma4.csv`, `qwen36_cross_model_agreement.csv`)
  could be removed once nothing else references them, since
  `cross_model_agreement_9model.csv` is a strict superset; left in place for
  now since removing tracked files was out of scope for this report.
