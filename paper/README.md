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

## Artifacts block (mandatory)

Every report must include an `## Artifacts` section immediately **after** the
header metadata block (Produced/Model/Scope/Status) and **before** the first
content section (Summary / Executive summary). It must list, by repo path, all
scripts, inputs, outputs, and figures the result depends on. Omit a sub-bullet
only when the category is genuinely empty.

Template:

```markdown
## Artifacts

- **Scripts:** `src/<file>.py` (and notebooks `notebooks/NN_*.ipynb` if used)
- **Inputs:** `data/processed/<vectors_subdir>/`, `data/stimuli/<file>`
- **Outputs:** `results/tables/<file>.csv`, `results/logs/<file>.json`
- **Figures:** `paper/figures/<figN>.{png,pdf}` (and `results/figures/<file>` if applicable)
```

This rule is also stated in `AGENTS.md` → "Findings Reports".

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
| `fig13_dense_steering_doseresponse.{png,pdf}` | 2-row × 4-col grid: raw_dense (solid) and random (dashed) dose-response for all four models; free per-panel y-axis (raw logit effects span ~100×) |
| `fig14_dense_steering_normalized.{png,pdf}` | 1×2 (warmth \| competence): cross-model steerability normalized by baseline concept gap; 12B steepest, 27B flattest; shared y-axis |
| `fig15_dense_steering_signal_vs_control.{png,pdf}` | 1×2 grouped bars at peak strength (α=+0.10): raw_dense vs random direction; ⚠ annotation where 27B competence random effect dominates signal |
| `fig16_hiring_probe_vs_human.{png,pdf}` | Grouped bars of Spearman ρ (model probe score vs. human rating) for 4 models × {warmth, competence}; negative bars for Llama/Qwen warmth show anti-alignment |
| `fig17_hiring_steering_callback.{png,pdf}` | 2 rows × 4 cols: mean Δcallback-margin over 60 names ± 95% CI for warmth and competence steering across all models |
| `fig18_hiring_disparity.{png,pdf}` | Two-panel: (A) model race/gender gaps in within-model SD units + human reference; (B) direction-agreement grid vs. human benchmark |
| `fig19_hiring_mediation_forest.{png,pdf}` | Forest plot of bootstrap indirect effects (name-group → probe → callback); significant rows filled, grouped by model |

Paper-draft figures (prefixed `paper_figure*`; produced 2026-06-24 for supervisor presentation):

| File | Description |
|------|-------------|
| `paper_figure1_axis_arrows.{png,pdf}` | 2×2 panels: warmth/competence story clouds with real-angle direction arrows; oblique basis preserves true inter-axis angle per model (Gemma ~41–45°, Qwen/Llama ~57–59°) |
| `paper_figure2_layer_emergence.{png,pdf}` | Single-panel layer sweep: Cohen's d vs layer fraction for warmth (solid) and competence (dotted) across all 4 models; probe-layer marker at frac=0.66; d=0.80 reference line |
| `paper_figure3_diverging_steering.{png,pdf}` | Position + boundary chart: x = absolute Yes/No logit margin; x=0 = decision boundary; bull's-eye dot = baseline (no steering); line+arrow = steerable range at ±0.10 × mean residual norm; every row crosses the boundary (steering is sufficient to flip the answer); Gemma-3-12B & 27B, raw_dense direction |

## Current reports

| File | Produced at (Europe/Berlin) | Model(s) | Scope | Status |
|------|-----------------------------|----------|-------|--------|
| `2026-06-16_2001_concept_stories_probe_findings.md` | 2026-06-16 20:01 | Gemma-3-12B-it | Phase 4+5: extraction + validation on 200 concept stories | Complete — steering and hiring evaluation to follow |
| `2026-06-19_1808_cross_model_concept_findings.md` | 2026-06-19 18:08 | Gemma-3-12B + Qwen3-14B + Llama-3.1-8B | Phase 4+5: three-model replication + cross-model agreement analysis | Complete — Phase B expansions (layer sweep, 27B, topic-holdout) to follow |
| `2026-06-20_1137_layer_sweep_topic_holdout.md` | 2026-06-20 11:37 | Gemma-3-12B + Qwen3-14B + Llama-3.1-8B | Phase B1+B2: topic-holdout CV + layer-sweep emergence and geometry | Complete; cross-axis interpretation corrected |
| `2026-06-20_1303_gemma_scale_paradox.md` | 2026-06-20 13:03 | Gemma-3-27B vs Gemma-3-12B | Phase B3: within-family scale test — Gemma depth-wise geometry persists at 27B | Corrected; filename retained for history |
| `2026-06-20_1337_cross_axis_metric_correction.md` | 2026-06-20 13:37 | Four models | Reproducibility correction for unscaled 1-D cross-axis logistic regression | Complete |
| `2026-06-20_1451_gemma_scope2_feature_causality.md` | 2026-06-20 14:51 | Gemma-3-12B + Gemma-3-27B | Gemma Scope 2 sparse decomposition, cross-scale feature matching, concept steering, and feature ablation | Complete for direct concept causality; hiring evaluation remains future work |
| `2026-06-24_1136_hiring_causality_results.md` | 2026-06-24 11:36 | Gemma-3-12B-it | Phase 6+7: hiring-callback causal sweep and probe-vs-human validation | Historical single-model report; final interpretation superseded by the four-model Phase 7 report. |
| `2026-06-24_1300_hiring_causality_27b_results.md` | 2026-06-24 13:00 | Gemma-3-27B-it | Phase 6+7 replication at 27B: steering non-monotone/fragile (not inert), reversed baseline association, stronger probe-vs-human alignment | Historical single-model report; broad-regime "warmth inert" interpretation superseded by local-regime non-monotone/fragile result. |
| `2026-06-27_1446_dense_steering_4model.md` | 2026-06-27 14:46 | Gemma-3-12B · Gemma-3-27B · Llama-3.1-8B · Qwen3-14B | Phase 6 extension: dense (SAE-free) steering replicated across all 4 models; normalized steerability, signal-vs-control, Gemma scale paradox | Complete |
| `2026-06-27_1541_hiring_phase7_4model.md` | 2026-06-27 15:41 | Gemma-3-12B · Gemma-3-27B · Llama-3.1-8B · Qwen3-14B | Phase 7 consolidated: probe-vs-human alignment, steering→callback, demographic disparity, bootstrap mediation; steerability paradox | Complete; B1 quantisation caveat documented, SCCKN hiring re-runs completed with no content changes, 27B non-monotone finding documented. |
| `2026-06-27_1650_stimulus_quality_audit.md` | 2026-06-27 16:50 | claude-opus-4-8 (audit) | Full quality audit of the 200 concept stories: structural balance, leakage checks, narrative quality, scored rubric (8.5/10), paper implications | Complete; dataset accepted for current analyses |
| `2026-06-27_1757_probe_human_data_audit.md` | 2026-06-27 17:57 | Gemma-3-12B · Gemma-3-27B · Llama-3.1-8B · Qwen3-14B | Data-quality audit for Test 2 probe-vs-human alignment: Gallo & Hausladen name ratings + model concept vectors; scored rubric (8.0/10), limitations, acceptance decision | Complete; data accepted for current analyses |
| `2026-06-30_1251_r4_disparity_name_level.md` | 2026-06-30 12:51 | Gemma-3-12B · Gemma-3-27B | R4 disparity: group-level (race × gender) and name-level OLS joining model callback margins with Gallo & Hausladen human callback benchmark; 149/282 names matched | Complete for 12B+27B. 27B: race gap +1.18 SD (Black > White, **opposes** human), gender gap −0.51 SD (matches human). 12B: quantisation-limited. |
| `2026-07-02_1000_bf16_quantisation_limitation.md` | 2026-07-02 10:00 | All four models | **IMPORTANT LIMITATION:** bf16 quantisation of callback margins (Bug B1) — root cause, partial fix applied, model-by-model impact, rerun outcome, and mandatory paper disclosure language | Active limitation; SCCKN hiring re-runs completed with no content changes |
