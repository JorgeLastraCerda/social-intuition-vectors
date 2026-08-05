# Split-Half Cosine Stability, Reproduced and Logged for All Nine Models

- **Produced:** 2026-08-04 14:32 Europe/Berlin
- **Model:** Nine Gemma, Qwen, and Llama checkpoints
- **Scope:** Reproduce the split-half cosine stability check (one of the five
  probe-validation checks listed in `paper/paper/Ulu_Lastra.tex`, Methods) with
  new, dedicated, git-tracked code, since no implementation for this check
  existed anywhere in the repository or its git history prior to this report
- **Status:** Complete

## Artifacts

- **Scripts:** `src/compute_split_half_stability.py` (new)
- **Inputs:** `data/processed/concept_vectors*/{X_high_warmth,X_low_warmth,X_high_competence,X_low_competence}.npy`
  and `meta.json`, for all nine model directories (`concept_vectors`,
  `concept_vectors_gemma3_27b`, `concept_vectors_gemma4_12b`,
  `concept_vectors_gemma4_26b_a4b`, `concept_vectors_gemma4_31b`,
  `concept_vectors_llama31_8b`, `concept_vectors_qwen36_27b`,
  `concept_vectors_qwen36_35b_a3b`, `concept_vectors_qwen3_14b`)
- **Outputs:** `results/logs/split_half_stability_<label>.json` for all nine
  labels (`gemma3_12b`, `gemma3_27b`, `gemma4_12b`, `gemma4_26b_a4b`,
  `gemma4_31b`, `llama31_8b`, `qwen36_27b`, `qwen36_35b_a3b`, `qwen3_14b`)
- **Figures:** none

## Why this report exists

During a manuscript audit (see `step_logs/STEP_LOG.md`, 2026-08-04 entries),
the user asked whether the "split-half cosine stability" check described in
`paper/paper/Ulu_Lastra.tex` (Methods, five-checks list) and first reported in
`paper/2026-06-16_2001_concept_stories_probe_findings.md` (warmth 0.83,
competence 0.88 for Gemma-3-12B-it) was ever actually implemented as
reproducible code. A thorough search — local filesystem, local `git log -S`
across all commits, the SCCKN cluster (`/work/emrecan.ulu/normalcy-axis` and
`/work/emrecan.ulu/normalcy-axis-parity`), SCCKN `git log -S`, SCCKN bash
history, and every notebook on both machines — found none. The June report's
numbers are real-looking but were never backed by any script committed
anywhere. This report closes that gap with new code and a full nine-model
re-run, and records the script's location here specifically so it is not
lost again.

## Method

`src/compute_split_half_stability.py::split_half_direction_cosine`: for a
given condition pair (e.g. `X_high_warmth`, `X_low_warmth`), each side's 50
story vectors are independently permuted (`numpy.random.default_rng(seed)`,
`seed=20260527`, the project's standard seed) and split into two halves of
25. Half A's direction is `mean(high half A) - mean(low half A)`; half B's
direction is built the same way from the complementary half. The reported
statistic is the cosine similarity between the two independently-built half
directions. The same procedure is run once for warmth and once for
competence, per model. No GPU or cluster access is required: the script only
reads already-extracted `.npy` arrays that are already committed to git.

## Results

| Model | Split-half cosine (warmth) | Split-half cosine (competence) |
|---|---|---|
| Gemma-3-12B-it | 0.815 | 0.897 |
| Gemma-3-27B-it | 0.808 | 0.879 |
| Gemma-4-12B | 0.772 | 0.869 |
| Gemma-4-26B-A4B | 0.687 | 0.852 |
| Gemma-4-31B | 0.733 | 0.856 |
| Llama-3.1-8B-Instruct | 0.700 | 0.831 |
| Qwen3.6-27B | 0.760 | 0.860 |
| Qwen3.6-35B-A3B | 0.714 | 0.858 |
| Qwen3-14B | 0.708 | 0.855 |

All nine models show moderate-to-high split-half cosine (0.69–0.90), and in
every model the competence direction is more stable than the warmth
direction. Gemma-3-12B-it's reproduced values (warmth 0.815, competence
0.897) are close to, but not identical to, the June report's numbers (0.83,
0.88); the small difference is expected, since the June computation's exact
random split and code were never recorded, so this is an independent
re-derivation under a documented, fixed seed rather than a byte-for-byte
replication.

## Caveats

- The pairing convention (independent permutations for the high- and
  low-condition halves, rather than a single joint permutation shared across
  both conditions) is a reasonable reading of "each condition's 50 stories
  are randomly split into halves of 25" but is not specified further in the
  original description; a different pairing convention could shift the exact
  values slightly.
- This reproduction used one random split per model (seeded), not a
  distribution over many splits; the manuscript bullet describes a single
  reported statistic, consistent with this choice.
- Results are not yet wired into `paper/paper/Ulu_Lastra.tex` (no numeric
  values are currently cited in that bullet) or into
  `src/validate_probes.py`'s automated JSON output; see Next Steps.

## Next Steps

- Optional: fold `split_half_direction_cosine` into `src/validate_probes.py`
  so it runs and logs alongside the other four checks in
  `results/logs/validate_probes_*.json`, giving all five checks the same
  reproducibility trail (flagged previously in `step_logs/STEP_LOG.md`,
  2026-08-04, Step 7).
- Optional: cite the nine-model table above directly in the manuscript's
  split-half bullet or in a table, if the user wants the check's actual
  numbers surfaced rather than only its description.
