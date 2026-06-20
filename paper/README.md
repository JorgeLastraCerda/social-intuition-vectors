# paper/ — Findings and Approach Reports

This directory holds human-readable write-ups of empirical findings and methodological decisions. One file per meaningful result or approach; figures live in `figures/`.

## Naming convention

```
YYYY-MM-DD_<short-slug>.md
```

Use the date the result was produced, not the date it was written up. Keep the slug short and descriptive (e.g. `concept_stories_probe_findings`, `steering_callback_results`).

## Relationship to the step log

- `step_logs/STEP_LOG.md` is the **index and trail**: it records that a result was produced, which files were touched, and what decision was taken. Entries are concise.
- Files in this directory carry the **detail**: full tables, figures, interpretation, and caveats.

Each findings report should be linked from the corresponding STEP_LOG entry via its file path.

## figures/

Single-model figures (Gemma-3-12B baseline):

| File | Description |
|------|-------------|
| `generate_figures.py` | Script that produces all figures; supports `--fig N`, `--vec-dir`, `--out-dir`, `--metrics`, `--logs`, `--vec-dirs`, `--labels`, `--stories` |
| `style.py` | Shared matplotlib style constants |
| `fig1_joint_density.{png,pdf}` | Joint density of warmth and competence projections (Gemma) |
| `fig2_random_baseline.{png,pdf}` | Warmth/competence direction vs random-baseline null (Gemma) |
| `fig3_lorenz_concentration.{png,pdf}` | Lorenz concentration curve of direction weight mass (Gemma) |
| `fig4_axis_geometry.{png,pdf}` | Vector geometry and cross-axis discriminability heatmap (Gemma; cos computed from data) |

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
| `fig8_layer_emergence.{png,pdf}` | Layer sweep: (left) Cohen's d emergence curves vs depth, 4 models; (right) cos(W,C) vs depth — Gemma-12B and 27B both elevated at all depths; Qwen/Llama plateau near 0.50 |

## Current reports

| File | Date | Model(s) | Scope | Status |
|------|------|----------|-------|--------|
| `2026-06-16_concept_stories_probe_findings.md` | 2026-06-16 | Gemma-3-12B-it | Phase 4+5: extraction + validation on 200 concept stories | Complete — steering and hiring evaluation to follow |
| `2026-06-19_cross_model_concept_findings.md` | 2026-06-19 | Gemma-3-12B + Qwen3-14B + Llama-3.1-8B | Phase 4+5: three-model replication + cross-model agreement analysis | Complete — Phase B expansions (layer sweep, 27B, topic-holdout) to follow |
| `2026-06-20_layer_sweep_topic_holdout.md` | 2026-06-20 | Gemma-3-12B + Qwen3-14B + Llama-3.1-8B | Phase B1+B2: topic-holdout CV (generalization confirmed) + layer sweep (emergence curves + cross-axis paradox resolved) | Complete |
| `2026-06-20_gemma_scale_paradox.md` | 2026-06-20 | Gemma-3-27B vs Gemma-3-12B | Phase B3: within-family scale test — paradox is scale-invariant; cos(W,C) depth profiles nearly identical at 12B and 27B | Complete — B4 (scale norm), B5 (full revision) to follow |
