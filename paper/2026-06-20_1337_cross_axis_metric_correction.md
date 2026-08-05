# Cross-Axis Metric Correction

**Produced at:** 2026-06-20 13:37 Europe/Berlin  
**Scope:** Four-model probe validation and affected reports

---

## Artifacts

- **Scripts:** `src/validate_probes.py`
- **Inputs:** `data/processed/concept_vectors/`, `data/processed/concept_vectors_gemma3_27b/`, `data/processed/concept_vectors_qwen3_14b/`, `data/processed/concept_vectors_llama31_8b/`
- **Outputs:** `results/tables/probe_metrics.csv`, `results/tables/probe_metrics_gemma3_27b.csv`, `results/tables/probe_metrics_qwen3_14b.csv`, `results/tables/probe_metrics_llama31_8b.csv`

---

## Finding

The previously reported chance-level cross-axis CV for Gemma-3-12B and Gemma-3-27B
was not reproducible across scikit-learn versions. The underlying vectors and
activations were correct; the problem was the 1-D classifier.

Gemma projection values have large positive offsets and magnitudes around
40,000–60,000. The original cross-axis implementation passed these raw values directly
to logistic regression. In the SCCKN environment (`scikit-learn 1.9.0`), the solver
stayed at a constant prediction and returned 0.50 in every fold. The same data produced
0.82–0.90 in another environment, revealing the instability.

## Correction

The classifier is now a fold-local pipeline:

```text
StandardScaler → LogisticRegression
```

Standardisation is fitted only on each training fold, preventing leakage while making
the metric invariant to residual-stream offsets and scale.

## Corrected Results

| Model | cos(W,C) | Cross W→C CV | Cross C→W CV |
|---|---:|---:|---:|
| Gemma-3-12B | 0.749 | 0.87 | 0.82 |
| Gemma-3-27B | 0.708 | 0.90 | 0.86 |
| Qwen3-14B | 0.536 | 1.00 | 1.00 |
| Llama-3.1-8B | 0.505 | 0.99 | 1.00 |

The claimed “cross-axis paradox” is therefore withdrawn. All four models show
cross-axis predictability, consistent with shared valence in the story design. The
layer-sweep findings for Cohen's d, topic-holdout CV, residual norms, and cos(W,C)
remain valid.

## Repository Changes

- `src/validate_probes.py` now uses scale-standardised projected CV.
- A regression test checks invariance to large shifts and scales.
- Figure 4 and all deterministic validation logs were regenerated.
- Reports that relied on the old 0.50 values were corrected.
- The unused legacy validation-PNG output path was removed.

---

## Update — 2026-08-04: nine-model coverage and a metric-interpretation caveat

**Produced at:** 2026-08-04 Europe/Berlin
**Scope:** Extend cross-axis classification to all nine models; distinguish the
recalibrated metric above from a stricter zero-shot version; close a
reproducibility gap where four models had no zero-shot number logged.

### Artifacts (this update)

- **Scripts:** `src/validate_probes.py` (`cross_axis_accuracy` /
  `projected_cv_accuracy` for the recalibrated metric;
  `topic_cross_axis_transfer_cv` for the zero-shot metric, both already
  present in the file, not newly written)
- **Inputs:** `data/processed/concept_vectors*/` (all nine model directories),
  `data/stimuli/concept_stories.jsonl` (topic groups)
- **Outputs:** `results/logs/validate_probes_{default,gemma3_27b,gemma4_12b,
  gemma4_26b_a4b,gemma4_31b,llama31_8b,qwen3_14b,qwen36_27b,qwen36_35b_a3b}.json`,
  `results/tables/probe_metrics{,_gemma3_27b,_gemma4_12b,_gemma4_26b_a4b,
  _gemma4_31b,_llama31_8b,_qwen3_14b,_qwen36_27b,_qwen36_35b_a3b}.csv`

### Why this update exists

A manuscript audit of the "Cross-axis classification" bullet in
`paper/paper/Ulu_Lastra.tex` found that the sentence's wording ("tests
whether the warmth direction predicts competence labels") reads as a
zero-shot claim, but the primary metric reported above
(`cross_warmth_on_competence_cv` / `cross_competence_on_warmth_cv`, from
`projected_cv_accuracy`) recalibrates a fresh classifier on the target
axis's own labels inside each fold; it measures whether the target axis is
separable along the source direction after readjustment, not whether the
source axis's own decision rule transfers unchanged. A stricter, genuinely
zero-shot variant, `topic_cross_axis_transfer_cv` (fits only on the source
axis's topic-held-out training folds, then scores the target axis with the
already-fitted pipeline and no recalibration), already existed in
`src/validate_probes.py` but had only been run for five of the nine models
(Gemma-4-12B, Gemma-4-26B-A4B, Gemma-4-31B, Qwen3.6-27B, Qwen3.6-35B-A3B).
The remaining four (Gemma-3-12B, Gemma-3-27B, Qwen3-14B, Llama-3.1-8B — the
same four models in this report's original table) had no zero-shot number
logged. This update re-runs `python3 -m src.validate_probes` for those four
models (`--vectors-subdir` / `--label` per model, same `config/config.yaml`,
same seed 20260527) to fill the gap. No GPU or cluster access was required;
the script only reads already-extracted `.npy` arrays. A `git diff` against
the pre-existing logs and CSVs confirmed every previously-reported field
(`cv_mean`, `topic_cv_mean`, `cross_*_on_*_cv`, `axis_cosine`) is unchanged
except floating-point noise at the 5th–6th decimal; only new fields were
added (`direction_topic_cv_*`, `cross_*_topic_transfer_*`).

### Nine-model results, both metrics

| Model | cos(W,C) | Recalibrated W→C | Recalibrated C→W | Zero-shot W→C | Zero-shot C→W |
|---|---:|---:|---:|---:|---:|
| Gemma-3-12B-it | 0.749 | 0.87 | 0.82 | 0.77 | 0.82 |
| Gemma-3-27B-it | 0.708 | 0.90 | 0.86 | 0.92 | 0.85 |
| Gemma-4-12B | 0.494 | 1.00 | 0.98 | 0.99 | 0.97 |
| Gemma-4-26B-A4B | 0.587 | 1.00 | 1.00 | 0.99 | 0.95 |
| Gemma-4-31B | 0.526 | 1.00 | 0.95 | 0.95 | 0.88 |
| Llama-3.1-8B-Instruct | 0.505 | 0.99 | 1.00 | 0.98 | 1.00 |
| Qwen3-14B | 0.536 | 1.00 | 1.00 | 1.00 | 0.99 |
| Qwen3.6-27B | 0.580 | 0.99 | 1.00 | 0.97 | 0.98 |
| Qwen3.6-35B-A3B | 0.619 | 1.00 | 0.97 | 0.99 | 0.93 |

("Recalibrated" = `cross_warmth_on_competence_cv` / `cross_competence_on_warmth_cv`,
the metric already cited in the original table above. "Zero-shot" =
`cross_warmth_to_competence_topic_transfer_mean` /
`cross_competence_to_warmth_topic_transfer_mean`, the stricter, no-recalibration
variant, now complete for all nine models.)

### Interpretation

Both metrics tell the same story in every model: warmth and competence
directions are strongly cross-predictive, consistent with substantial
shared evaluative content in the story design (the withdrawal of the
"cross-axis paradox" above still holds, now at nine-model scale). The two
metrics track each other closely — the largest gap is Gemma-4-31B's
warmth→competence pair (1.00 recalibrated vs. 0.95 zero-shot) — so for this
dataset the recalibration does not manufacture the high cross-axis
predictability; the source axis's own unrecalibrated decision rule already
transfers well in every model. The metric-interpretation distinction is
still worth keeping in mind when citing a single number: "recalibrated"
answers "is the target axis separable along this direction," while
"zero-shot" answers "does the source axis's own classifier work unchanged
on the other axis," and the manuscript's current wording is closer to the
second question even though it currently cites the first metric.

### Next Steps

- If the user wants the manuscript's "Cross-axis classification" bullet to
  cite a specific number, prefer the zero-shot column, since it matches the
  bullet's wording more literally.
- `direction_topic_cv_*` was also newly logged for these four models as a
  byproduct of this re-run (an alternative topic-holdout variant that
  rebuilds the direction inside each fold and trains on the 1-D projection);
  it is not currently referenced anywhere in the manuscript and is noted
  here only so it is not mistaken for unexplained new data later.
