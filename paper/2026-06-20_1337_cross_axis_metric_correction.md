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
