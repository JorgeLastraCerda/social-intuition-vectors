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

| File | Description |
|------|-------------|
| `generate_figures.py` | Script that produces all figures; run with `--fig N` to regenerate a single figure |
| `style.py` | Shared matplotlib style constants |
| `fig1_joint_density.{png,pdf}` | Joint density of warmth and competence projections |
| `fig2_random_baseline.{png,pdf}` | Warmth direction vs equal-magnitude random baseline |
| `fig3_lorenz_concentration.{png,pdf}` | Lorenz concentration curve of direction weight mass |
| `fig4_axis_geometry.{png,pdf}` | 2-D projection of warmth / competence axes |

## Current reports

| File | Date | Model | Scope | Status |
|------|------|-------|-------|--------|
| `2026-06-16_concept_stories_probe_findings.md` | 2026-06-16 | Gemma-3-12B-it | Phase 4 (vector extraction) + Phase 5 (probe validation) on 200 concept stories | Complete — steering and hiring evaluation to follow |
