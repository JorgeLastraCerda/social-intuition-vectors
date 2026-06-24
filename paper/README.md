# paper/ — Findings and Approach Reports

This directory holds human-readable write-ups of empirical findings and methodological decisions. One file per meaningful result or approach; figures live in `figures/`.

## Naming convention

```
YYYY-MM-DD_HHMM_<short-slug>.md
```

Use the date and 24-hour local time when the result was produced, in the
`Europe/Berlin` timezone. Zero-pad the hour and minute and omit the colon so
lexicographic filename order is chronological. Use the result-production time,
not a later edit time. When recovering historical reports whose production time
was not recorded explicitly, use the report's earliest Git commit time. Keep the
slug short and descriptive (e.g. `concept_stories_probe_findings`,
`steering_callback_results`).

## Relationship to the step log

- `step_logs/STEP_LOG.md` is the **index and trail**: it records that a result was produced, which files were touched, and what decision was taken. Entries are concise.
- Files in this directory carry the **detail**: full tables, figures, interpretation, and caveats.

Each findings report should be linked from the corresponding STEP_LOG entry via its file path.

## figures/

Single-model figures (Gemma-3-12B baseline):

| File | Description |
|------|-------------|
| `generate_figures.py` | Script that produces all figures; supports single-model, cross-model, layer-sweep, Gemma Scope metrics, causality, ablation, and feature-match inputs |
| `style.py` | Shared matplotlib style constants |
| `fig1_joint_density.{png,pdf}` | Joint density of warmth and competence projections (Gemma) |
| `fig2_random_baseline.{png,pdf}` | Warmth/competence direction vs random-baseline null (Gemma) |
| `fig3_lorenz_concentration.{png,pdf}` | Lorenz concentration curve of direction weight mass (Gemma) |
| `fig4_axis_geometry.{png,pdf}` | Vector geometry and scale-standardised 1-D cross-axis discriminability (Gemma) |

Per-model figures for Qwen3-14B and Llama-3.1-8B in subdirectories:

| Path | Description |
|------|-------------|
| `qwen3_14b/fig1_joint_density.{png,pdf}` | Joint density — Qwen3-14B |
| `qwen3_14b/fig2_random_baseline.{png,pdf}` | Random baseline (z=14.1/14.6) — Qwen3-14B |
| `qwen3_14b/fig3_lorenz_concentration.{png,pdf}` | Lorenz concentration — Qwen3-14B |
| `llama31_8b/fig1_joint_density.{png,pdf}` | Joint density — Llama-3.1-8B |
| `llama31_8b/fig2_random_baseline.{png,pdf}` | Random baseline (z=15.0/15.1) — Llama-3.1-8B |
| `llama31_8b/fig3_lorenz_concentration.{png,pdf}` | Lorenz concentration — Llama-3.1-8B |

Cross-model figures:

| File | Description |
|------|-------------|
| `fig5_cross_model.{png,pdf}` | Grouped bars: CV accuracy, Cohen's d, cos(W,C) across 3 models |
| `fig6_cross_model_story_agreement.{png,pdf}` | 3×3 Spearman ρ heatmaps for per-story ranking agreement |
| `fig7_same_story_demo.{png,pdf}` | 6 exemplar stories in z-scored warmth/competence space, 3 models overlaid |
| `fig8_layer_emergence.{png,pdf}` | Layer sweep: (left) Cohen's d emergence curves vs depth, 4 models; (right) cos(W,C) vs depth — Gemma-12B and 27B both elevated through most depths |
| `fig9_gemma_scope_decomposition.{png,pdf}` | Gemma Scope 2 reconstruction, sparsity, topic-holdout concept signal, and decoded-direction alignment across 16k/65k/262k widths |
| `fig10_gemma_scope_steering.{png,pdf}` | Local-regime held-out concept steering for dense, SAE, shared, axis-specific, other-axis, and random directions |
| `fig11_gemma_scope_ablation.{png,pdf}` | Error-preserving ablation of target, shared, other-axis, and random 65k feature sets |
| `fig12_gemma_scope_feature_matching.{png,pdf}` | One-to-one 12B↔27B feature-profile matches compared with a 500-permutation row-shuffle null |

## Current reports

| File | Produced at (Europe/Berlin) | Model(s) | Scope | Status |
|------|-----------------------------|----------|-------|--------|
| `2026-06-16_2001_concept_stories_probe_findings.md` | 2026-06-16 20:01 | Gemma-3-12B-it | Phase 4+5: extraction + validation on 200 concept stories | Complete — steering and hiring evaluation to follow |
| `2026-06-19_1808_cross_model_concept_findings.md` | 2026-06-19 18:08 | Gemma-3-12B + Qwen3-14B + Llama-3.1-8B | Phase 4+5: three-model replication + cross-model agreement analysis | Complete — Phase B expansions (layer sweep, 27B, topic-holdout) to follow |
| `2026-06-20_1137_layer_sweep_topic_holdout.md` | 2026-06-20 11:37 | Gemma-3-12B + Qwen3-14B + Llama-3.1-8B | Phase B1+B2: topic-holdout CV + layer-sweep emergence and geometry | Complete; cross-axis interpretation corrected |
| `2026-06-20_1303_gemma_scale_paradox.md` | 2026-06-20 13:03 | Gemma-3-27B vs Gemma-3-12B | Phase B3: within-family scale test — Gemma depth-wise geometry persists at 27B | Corrected; filename retained for history |
| `2026-06-20_1337_cross_axis_metric_correction.md` | 2026-06-20 13:37 | Four models | Reproducibility correction for unscaled 1-D cross-axis logistic regression | Complete |
| `2026-06-20_1451_gemma_scope2_feature_causality.md` | 2026-06-20 14:51 | Gemma-3-12B + Gemma-3-27B | Gemma Scope 2 sparse decomposition, cross-scale feature matching, concept steering, and feature ablation | Complete for direct concept causality; hiring evaluation remains future work |
| `2026-06-24_1136_hiring_causality_results.md` | 2026-06-24 11:36 | Gemma-3-12B-it | Phase 6+7: hiring-callback causal sweep and probe-vs-human validation | Complete for 12B baseline |
| `2026-06-24_1300_hiring_causality_27b_results.md` | 2026-06-24 13:00 | Gemma-3-27B-it | Phase 6+7 replication at 27B: steering inert for warmth, reversed baseline association, stronger probe-vs-human alignment | Complete; demographic disparity requires D-Phase7-A/B decisions |
