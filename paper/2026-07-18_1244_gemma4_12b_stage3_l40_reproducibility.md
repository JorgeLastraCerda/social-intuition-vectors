# Gemma 4 12B Stage 3 and L40/L40S Reproducibility

- **Produced:** 2026-07-18 12:44 CEST
- **Model:** `google/gemma-4-12B-it`
- **Scope:** Stage 3 all-layer warmth and competence probe sweep on 200 synthetic concept stories, with a same-hardware reproducibility audit
- **Status:** Complete; exact-L40 result passes all technical and cross-stage consistency gates

## Artifacts

- **Scripts:** `src/layer_sweep.py`; `src/validate_gemma4_stage.py`; `jobs/sge/gemma4_12b_stage3_retry.sh`; `jobs/sge/gemma4_12b_stage3_finalize.sh`; `jobs/sge/submit_gemma4_12b_stage3_retry.sh`; `jobs/sge/submit_gemma4_12b_l40_repro.sh`
- **Inputs:** `data/stimuli/concept_stories.jsonl`; `data/processed/concept_vectors_gemma4_12b/`; `results/tables/probe_metrics_gemma4_12b.csv`; `results/logs/validate_probes_gemma4_12b.json`
- **Outputs:** `results/tables/layer_sweep_gemma4_12b.csv`; `results/tables/layer_sweep_gemma4_12b.meta.json`; `results/tables/layer_sweep_gemma4_12b_l40_repro.csv`; `results/tables/layer_sweep_gemma4_12b_l40_repro.meta.json`
- **Logs:** `results/logs/gemma4_stage3_retry_submission_12b_20260718T102845Z.json`; `results/logs/gemma4_stage3_retry_submission_12b_l40_repro_20260718T103904Z.json`

## Execution outcome

The first independent retry was submitted to the SCCKN L40 pool and dispatched to an NVIDIA L40S on `scc213`. It completed successfully in 146 seconds with 23.174 GB maximum virtual memory and produced a complete, finite 48-layer table. The dependent CPU finalizer validated and synchronized that write-once result.

Because Stage 1 activations had been extracted on an NVIDIA L40 on `scc192`, the L40S sweep did not satisfy the planned cross-stage numerical gate. A separately labeled audit was therefore run on the exact L40 hardware class without deleting or overwriting the L40S output. The exact-L40 job completed in 100 seconds with 20.994 GB maximum virtual memory and passed the 48-layer structural validator.

## Probe-layer consistency

| Result | Warmth d | Competence d | cos(W,C) | Topic holdout W/C | Mean residual norm |
|---|---:|---:|---:|---:|---:|
| Stage 2 stored activations | 8.633730 | 9.035413 | 0.493539 | 1.00 / 1.00 | n/a |
| Stage 3 exact L40, layer 31 | 8.633730 | 9.035413 | 0.493539 | 1.00 / 1.00 | 97.8189 |
| Stage 3 L40S, layer 31 | 8.461919 | 8.982933 | 0.492562 | 1.00 / 1.00 | 97.8765 |

The exact-L40 sweep reproduces all three Stage 2 probe-layer geometry metrics with zero difference at the six-decimal reporting precision. Relative to exact L40, the L40S run changes warmth d by -0.171811, competence d by -0.052480, and cosine by -0.000977. Across all layers, the largest absolute L40/L40S differences are 0.638015 for warmth d, 0.913529 for competence d, 0.094955 for cosine, and only 0.2366 for mean residual norm.

## Exact-L40 depth profile

| Quantity | Layer | Fraction | Value |
|---|---:|---:|---:|
| Maximum warmth d | 26 | 0.5532 | 10.563076 |
| Maximum competence d | 27 | 0.5745 | 10.445699 |
| Maximum cos(W,C) | 25 | 0.5319 | 0.616757 |
| Configured probe layer | 31 | 0.6596 | d = 8.633730 / 9.035413 |

Warmth and competence are already separable at layer 0, with topic-holdout accuracies of 0.82 and 0.96. Both reach 1.00 by layer 20 and remain perfect through layer 46. The final layer drops to 0.94/0.97 topic holdout and d = 0.533405/1.270310, showing the same middle-layer amplification and late collapse seen in the larger Gemma 4 models.

## Interpretation and caveats

The exact Stage 2 reproduction on the same L40 hardware supports the technical validity of the 12B layer sweep. It also shows that the earlier RTX PRO 6000 failure was not caused by insufficient model capacity: both L40-family jobs loaded the model within approximately 23 GB.

The L40/L40S difference should be treated as a numerical reproducibility warning for bfloat16 activation geometry, not as a fully isolated hardware effect. Each device class was observed once in Stage 3, so run-to-run kernel nondeterminism is not separately estimated. The scientific pattern is nevertheless stable across both runs: perfect probe-layer topic holdout, very large d values, a middle-layer peak, and substantial warmth/competence alignment.

As in Stage 2, strong cross-axis classification and cos(W,C) around 0.49 at the probe layer show that warmth and competence remain entangled with a shared evaluative component. The exact-L40 result resolves technical reproducibility but does not resolve this scientific caveat or establish external validity.
