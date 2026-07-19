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

Per-model figures for the three Gemma 4 models (Stage 1 extraction geometry):

| Path | Description |
|------|-------------|
| `gemma4_12b/fig1_joint_density.{png,pdf}` | Joint density — Gemma-4-12B-it |
| `gemma4_12b/fig2_random_baseline.{png,pdf}` | Random baseline (z=14.1/12.8) — Gemma-4-12B-it |
| `gemma4_12b/fig3_lorenz_concentration.{png,pdf}` | Lorenz concentration — Gemma-4-12B-it |
| `gemma4_12b/fig4_axis_geometry.{png,pdf}` | Axis geometry, cos=0.494 — Gemma-4-12B-it |
| `gemma4_26b_a4b/fig1_joint_density.{png,pdf}` | Joint density — Gemma-4-26B-A4B-it |
| `gemma4_26b_a4b/fig2_random_baseline.{png,pdf}` | Random baseline (z=15.0/14.3) — Gemma-4-26B-A4B-it |
| `gemma4_26b_a4b/fig3_lorenz_concentration.{png,pdf}` | Lorenz concentration — Gemma-4-26B-A4B-it |
| `gemma4_26b_a4b/fig4_axis_geometry.{png,pdf}` | Axis geometry, cos=0.587 — Gemma-4-26B-A4B-it |
| `gemma4_31b/fig1_joint_density.{png,pdf}` | Joint density — Gemma-4-31B-it |
| `gemma4_31b/fig2_random_baseline.{png,pdf}` | Random baseline (z=13.1/8.6) — Gemma-4-31B-it |
| `gemma4_31b/fig3_lorenz_concentration.{png,pdf}` | Lorenz concentration — Gemma-4-31B-it |
| `gemma4_31b/fig4_axis_geometry.{png,pdf}` | Axis geometry, cos=0.526 — Gemma-4-31B-it |

Cross-model figures (Gemma 4 Stage 2 probe validation, three-model comparison):

| Path | Description |
|------|-------------|
| `gemma4_cross/fig5_cross_model.{png,pdf}` | Grouped bars: CV accuracy, Cohen's d, cos(W,C) across the three Gemma 4 models — CV flat at ceiling, 31B competence d visibly lowest, 26B-A4B cosine visibly highest |
| `gemma4_cross/fig6_cross_model_story_agreement.{png,pdf}` | Overall and within-condition Spearman ρ heatmaps — overall ρ=0.905–0.960; within-condition warmth ρ=0.434–0.574 and competence ρ=0.618–0.645 |
| `gemma4_cross/fig7_same_story_demo.{png,pdf}` | Qualitative overlay of 6 Gemma-4-12B-selected exemplar stories across the three variants |

Cross-model figures (Gemma 4 Stage 3 consolidated layer sweep, three-model comparison):

| Path | Description |
|------|-------------|
| `gemma4_cross/fig8_layer_emergence.{png,pdf}` | Cohen's d and cos(W,C) vs layer fraction, three Gemma 4 models — mid-layer amplification peak and mid-layer cosine peak in all three, 12B row uses the exact-L40 sweep |
| `gemma4_cross/fig8b_stage3b_validation.{png,pdf}` | Stage 3B all-layer direction topic holdout, strict cross-axis transfer, and paired-topic 95% bootstrap intervals for Cohen's d and cos(W,C) |

Full-run figures for Qwen3.6-27B and Qwen3.6-35B-A3B:

| Path | Description |
|------|-------------|
| `qwen36_27b/fig{1,2,3,4}_*.{png,pdf}` | Qwen3.6-27B Stage 1 extraction geometry, random baseline, concentration, and axis geometry |
| `qwen36_27b/fig5_cross_model.{png,pdf}` | Qwen3.6-27B Stage 2 probe accuracy, effect size, and axis overlap |
| `qwen36_27b/fig8_layer_emergence.{png,pdf}` | Qwen3.6-27B Stage 3 all-layer profile |
| `qwen36_35b_a3b/fig{1,2,3,4}_*.{png,pdf}` | Qwen3.6-35B-A3B Stage 1 extraction geometry, random baseline, concentration, and axis geometry |
| `qwen36_35b_a3b/fig5_cross_model.{png,pdf}` | Qwen3.6-35B-A3B Stage 2 probe accuracy, effect size, and axis overlap |
| `qwen36_35b_a3b/fig8_layer_emergence.{png,pdf}` | Qwen3.6-35B-A3B Stage 3 all-layer profile |
| `qwen36_cross/fig5_cross_model.{png,pdf}` | Two-model probe-layer comparison |
| `qwen36_cross/fig8_layer_emergence.{png,pdf}` | Two-model layer-depth comparison |

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
| `fig20_pca_denoising.{png,pdf}` | PCA denoising summary for Gemma-3-12B and Gemma-3-27B: neutral cumulative variance, raw vs denoised warmth/competence cosine, and concept separation before/after denoising |

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
| `2026-07-02_1921_pca_denoising_results.md` | 2026-07-02 19:21 | Gemma-3-12B · Gemma-3-27B | PCA denoising of warmth/competence vectors against neutral Wikipedia residual activations | Complete; denoising reduces but does not eliminate shared warmth/competence geometry |
| `2026-07-15_0035_gemma4_transformerlens_pipeline.md` | 2026-07-15 00:35 | Gemma-4-31B · Gemma-4-26B-A4B | TransformerLens Bridge implementation and gated SCCKN replication workflow | Implementation complete; SCCKN smoke and production runs pending |
| `2026-07-15_0839_gemma4_12b_smoke.md` | 2026-07-15 08:39 | Gemma-4-12B-it | TransformerLens Bridge compatibility, hook, steering, and single-GPU memory smoke | Complete; all smoke gates passed. Independent 31B retry remains queued. |
| `2026-07-18_1208_gemma4_stage3_layer_sweep.md` | 2026-07-18 12:08 | Gemma-4-26B-A4B-it · Gemma-4-31B-it | Stage 3 all-layer warmth/competence probe sweep and independent RTX retry | Complete for 26B-A4B and 31B; 12B Stage 3 OOM remains separate. |
| `2026-07-18_1244_gemma4_12b_stage3_l40_reproducibility.md` | 2026-07-18 12:44 | Gemma-4-12B-it | Stage 3 all-layer sweep plus exact-L40 versus L40S cross-stage reproducibility audit | Complete; exact L40 reproduces Stage 2 probe metrics exactly at six decimals, while the L40S run shows small bfloat16 geometry drift. |
| `2026-07-18_1308_gemma4_stage1_extraction_geometry.md` | 2026-07-18 13:08 | Gemma-4-12B-it · Gemma-4-26B-A4B-it · Gemma-4-31B-it | Stage 1: extraction geometry — vector norms, cos(W,C), random-baseline null separation, Lorenz weight concentration | Complete for all three models; 26B-A4B most concentrated and most cross-axis-entangled, while 31B has the lowest cross-axis CV cell. |
| `2026-07-18_1336_qwen36_27b_native_hf_smoke.md` | 2026-07-18 13:36 | Qwen3.6-27B | Native Hugging Face Stage 1–3 technical smoke on 40 concept stories and one RTX PRO 6000 | Complete; all hook, parity, memory, output, and scheduler gates passed without TransformerLens. |
| `2026-07-18_1326_gemma4_stage2_probe_validation.md` | 2026-07-18 13:26 | Gemma-4-12B-it · Gemma-4-26B-A4B-it · Gemma-4-31B-it | Stage 2: full-feature and direction-specific topic validation, cross-axis transfer, and condition-aware story-ranking agreement | Target-axis CV is 1.00 throughout; strict topic-held-out transfer is 0.88–0.99; within-condition ρ is 0.434–0.645. |
| `2026-07-18_1340_gemma4_stage3_layer_sweep_consolidated.md` | 2026-07-18 13:40 | Gemma-4-12B-it · Gemma-4-26B-A4B-it · Gemma-4-31B-it | Stage 3: consolidated all-layer sweep — Cohen's d and cos(W,C) depth profiles, probe-layer reproduction of Stage 2, 12B exact-L40 vs L40S hardware audit | Complete for all three models; all peak before frac=0.66, mid-layer cosine peak (0.62–0.74) in all three, 12B exact-L40 reproduces Stage 2 with zero difference. |
| `2026-07-18_1453_gemma4_stage3b_validation.md` | 2026-07-18 14:53 | Gemma-4-12B-it · Gemma-4-26B-A4B-it · Gemma-4-31B-it | Stage 3B: all-layer direction validation, strict cross-axis topic transfer, paired-topic bootstrap uncertainty, and SCCKN provenance | Complete; legacy Stage 3 is reproduced exactly, probe-layer direction CV is 1.00/1.00, and strict transfer remains 0.88–0.99. |
| `2026-07-18_1404_qwen36_27b_stage1_extraction.md` | 2026-07-18 14:04 | Qwen3.6-27B | Full Stage 1 extraction on 200 stories using native Hugging Face hooks | Complete; passive-hook, hidden-state, text-only, memory, and output gates passed. |
| `2026-07-18_1408_qwen36_27b_stage2_validation.md` | 2026-07-18 14:08 | Qwen3.6-27B | Full Stage 2 probe validation | Complete; both axes achieve 1.00 five-fold and topic-held-out accuracy. |
| `2026-07-18_1408_qwen36_27b_stage3_layer_sweep.md` | 2026-07-18 14:08 | Qwen3.6-27B | Full Stage 3 all-layer sweep | Complete; 64 finite layers and exact probe-layer reproduction. |
| `2026-07-18_1414_qwen36_35b_a3b_stage1_extraction.md` | 2026-07-18 14:14 | Qwen3.6-35B-A3B | Full Stage 1 extraction on 200 stories using native Hugging Face hooks | Complete; one RTX PRO 6000, BF16, no fallback. |
| `2026-07-18_1418_qwen36_35b_a3b_stage2_validation.md` | 2026-07-18 14:18 | Qwen3.6-35B-A3B | Full Stage 2 probe validation | Complete; both axes achieve 1.00 five-fold and topic-held-out accuracy. |
| `2026-07-18_1418_qwen36_35b_a3b_stage3_layer_sweep.md` | 2026-07-18 14:18 | Qwen3.6-35B-A3B | Full Stage 3 all-layer sweep | Complete; 40 finite layers and exact probe-layer reproduction. |
| `2026-07-18_1421_qwen36_full_stage_comparison.md` | 2026-07-18 14:21 | Qwen3.6-27B · Qwen3.6-35B-A3B | Cross-model synthesis of all six full stage runs | Complete; all jobs and technical gates passed, with story- and layer-level comparison. |
| `2026-07-18_1453_qwen36_27b_strict_stage2_validation.md` | 2026-07-18 14:53 | Qwen3.6-27B | Stage 2 fold-internal direction reconstruction and strict cross-axis topic transfer | Complete; direction CV 1.00/1.00 and strict transfer 0.97/0.98. |
| `2026-07-18_1454_qwen36_35b_a3b_strict_stage2_validation.md` | 2026-07-18 14:54 | Qwen3.6-35B-A3B | Stage 2 fold-internal direction reconstruction and strict cross-axis topic transfer | Complete; direction CV 1.00/1.00 and strict transfer 0.99/0.93. |
| `2026-07-18_1604_gemma4_remaining_pipeline.md` | 2026-07-18 16:04 | Gemma-4-12B-it · Gemma-4-26B-A4B-it · Gemma-4-31B-it | SAE-free remaining-test implementation: strengthened dense controls, neutral-PCA denoising, hiring steering, conditional 282-name expansion, and independent SCCKN execution | Implementation and local validation complete; empirical SCCKN runs pending. |
| `2026-07-18_1612_gemma4_12b_remaining_smoke.md` | 2026-07-18 16:12 | Gemma-4-12B-it | Pinned remaining-test technical smoke on exact NVIDIA L40 | Complete; all revision, hook, parity, steering, memory, hardware, and scheduler gates passed. |
| `2026-07-18_1612_gemma4_26b_a4b_remaining_smoke.md` | 2026-07-18 16:12 | Gemma-4-26B-A4B-it | Pinned remaining-test technical smoke on RTX PRO 6000 | Complete; all revision, hook, parity, steering, memory, hardware, and scheduler gates passed. |
| `2026-07-18_1612_gemma4_31b_remaining_smoke.md` | 2026-07-18 16:12 | Gemma-4-31B-it | Pinned remaining-test technical smoke on RTX PRO 6000 | Complete; all revision, hook, parity, steering, memory, hardware, and scheduler gates passed. |
| `2026-07-18_1623_gemma4_12b_hiring_audit.md` | 2026-07-18 16:23 | Gemma-4-12B-it | Full 282-name unsteered hiring audit and probe-versus-human association | Complete; warmth alignment is absent, competence alignment is modest positive, and callback margin is negatively associated with the model competence probe. |
| `2026-07-18_1623_gemma4_26b_a4b_neutral_extraction.md` | 2026-07-18 16:23 | Gemma-4-26B-A4B-it | Neutral-corpus residual extraction for PCA denoising | Complete; finite 1500×2816 matrix accepted for independent PCA. |
| `2026-07-18_1747_calibrated_steering_pipeline.md` | 2026-07-18 17:47 | Gemma-3-12B-it · Gemma-4-12B-it · Qwen3.6-27B pilot | SD-matched random/cross-axis controls plus norm-preserving steering robustness | Implementation complete and locally validated; three independent RTX PRO 6000 pilots pending. |
| `2026-07-18_2201_gemma4_12b_calibrated_steering.md` | 2026-07-18 22:01 | Gemma-4-12B-it | Calibrated additive and norm-preserving steering with 99 SD-matched random directions | Complete SCCKN supporting result; CCU H100 primary replication pending. |
| `2026-07-18_2201_qwen36_27b_calibrated_incomplete.md` | 2026-07-18 22:01 | Qwen3.6-27B | Calibrated native-HF steering validation diagnosis | Incomplete; rejected because non-contiguous topic identifiers were sampled as contiguous indices. |
| `2026-07-18_2314_gemma4_12b_ccu_calibrated_steering.md` | 2026-07-18 23:14 | Gemma-4-12B-it | Primary H100 calibrated additive and norm-preserving steering replication | Complete; all 40,440 raw rows and validation gates passed. |
| `2026-07-18_2314_gemma4_12b_posthoc_hiring.md` | 2026-07-18 23:14 | Gemma-4-12B-it | Demographic disparity, mediation, group R4, and name-level R4 | Complete; 269-name disparity/mediation join and 149-name R4 join validated. |
| `2026-07-18_2314_gemma4_12b_full282_gate.md` | 2026-07-18 23:14 | Gemma-4-12B-it | Predeclared conditional expansion gate | Gate fired on four broad-regime criteria; three 282-name runs required. |
| `2026-07-18_2318_gemma4_12b_local_full282.md` | 2026-07-18 23:18 | Gemma-4-12B-it | Raw local-strength steering across all 282 rated names | Complete; both axes are monotone with positive endpoint effects. |
| `2026-07-18_2322_gemma4_12b_broad_full282.md` | 2026-07-18 23:22 | Gemma-4-12B-it | Raw broad-strength steering across all 282 rated names | Complete; both axes remain non-monotone and reverse at +0.50. |
| `2026-07-18_2325_gemma4_12b_denoised_full282.md` | 2026-07-18 23:25 | Gemma-4-12B-it | Denoised local-strength steering across all 282 rated names | Complete; denoising attenuates but preserves monotone causal effects. |
| `2026-07-18_2335_gemma4_26b_a4b_posthoc_hiring.md` | 2026-07-18 23:35 | Gemma-4-26B-A4B-it | Demographic disparity, mediation, group R4, and name-level R4 | Complete; two mediation intervals exclude zero. |
| `2026-07-18_2335_gemma4_31b_posthoc_hiring.md` | 2026-07-18 23:35 | Gemma-4-31B-it | Demographic disparity, mediation, group R4, and name-level R4 | Complete; competence is the strongest name-level correlate. |
| `2026-07-18_2335_gemma4_26b_a4b_full282_gate.md` | 2026-07-18 23:35 | Gemma-4-26B-A4B-it | Predeclared conditional expansion gate | Gate fired on eight competence-response criteria; three 282-name runs queued. |
| `2026-07-18_2338_gemma4_26b_a4b_calibrated_steering.md` | 2026-07-18 23:38 | Gemma-4-26B-A4B-it | Calibrated target, cross-axis, and 99-direction random-control steering | Complete; structural gates pass, but target endpoints are weak under the random null. |
| `2026-07-18_2338_gemma4_31b_denoised_hiring.md` | 2026-07-18 23:38 | Gemma-4-31B-it | Missing denoised local 60-name hiring steering | Complete; both axes are non-monotone with negative +0.10 endpoints. |
| `2026-07-18_2338_gemma4_31b_full282_gate.md` | 2026-07-18 23:38 | Gemma-4-31B-it | Predeclared conditional expansion gate | Gate fired on sixteen criteria; three 282-name runs queued. |
| `2026-07-18_2346_gemma4_26b_a4b_local_full282.md` | 2026-07-18 23:46 | Gemma-4-26B-A4B-it | Raw local-strength steering across all 282 rated names | Complete; warmth is monotone but weak, while competence remains non-monotone. |
| `2026-07-19_0022_gemma4_26b_a4b_broad_full282.md` | 2026-07-19 00:22 | Gemma-4-26B-A4B-it | Raw broad steering across all 282 names | Complete; warmth is monotone, competence is non-monotone and negative-ended. |
| `2026-07-19_0022_gemma4_26b_a4b_denoised_full282.md` | 2026-07-19 00:22 | Gemma-4-26B-A4B-it | Denoised local steering across all 282 names | Complete; denoising preserves competence instability. |
| `2026-07-19_0022_gemma4_31b_local_full282.md` | 2026-07-19 00:22 | Gemma-4-31B-it | Raw local steering across all 282 names | Complete; both axes are non-monotone with negative endpoints. |
| `2026-07-19_0022_gemma4_31b_broad_full282.md` | 2026-07-19 00:22 | Gemma-4-31B-it | Raw broad steering across all 282 names | Complete; both axes reverse strongly at +0.50. |
| `2026-07-19_0022_gemma4_31b_denoised_full282.md` | 2026-07-19 00:22 | Gemma-4-31B-it | Denoised local steering across all 282 names | Complete; denoising does not restore monotonicity. |
| `2026-07-19_0030_gemma4_31b_calibrated_steering.md` | 2026-07-19 00:30 | Gemma-4-31B-it | Calibrated target, cross-axis, and 99-direction random-control steering | Complete; structural gates pass, but neither target exceeds the calibrated random null. |
| `2026-07-19_0944_qwen36_resumable_parallel_pipeline.md` | 2026-07-19 09:44 | Qwen3.6-27B · Qwen3.6-35B-A3B | Resumable calibrated and native-HF hiring pipeline across two RTX PRO 6000 GPUs and one H100 | Implementation validated; initial three independent GPU runs ready. |
| `2026-07-19_0949_qwen36_27b_hiring_audit.md` | 2026-07-19 09:49 | Qwen3.6-27B | Full 282-name native-HF warmth/competence and callback audit | Complete; modest positive human alignment and positive callback associations. |
| `2026-07-19_0953_qwen36_27b_local_hiring.md` | 2026-07-19 09:53 | Qwen3.6-27B | Raw local-strength hiring steering across 60 names | Complete; both axes are monotone with positive, bootstrap-resolved endpoints. |
| `2026-07-19_0953_qwen36_27b_posthoc_hiring.md` | 2026-07-19 09:53 | Qwen3.6-27B | Demographic disparity, mediation, group R4, and name-level R4 | Complete; competence mediation intervals exclude zero, while model-human name-level callback alignment is weak. |
| `2026-07-19_0955_qwen36_27b_broad_hiring.md` | 2026-07-19 09:55 | Qwen3.6-27B | Raw broad-strength hiring steering across 60 names | Complete; both axes remain monotone and positive-ended from -0.50 to +0.50. |
| `2026-07-19_1003_qwen36_27b_neutral_extraction.md` | 2026-07-19 10:03 | Qwen3.6-27B | Native-HF neutral-corpus activation extraction | Complete; validated finite 1500×5120 matrix with exact remote-to-local hash parity. |
| `2026-07-19_1003_qwen36_27b_pca_denoising.md` | 2026-07-19 10:03 | Qwen3.6-27B | Neutral-PCA denoising of warmth and competence vectors | Complete; 27 components reduce cross-axis cosine modestly but do not remove leakage. |
| `2026-07-19_1006_qwen36_35b_a3b_hiring_audit.md` | 2026-07-19 10:06 | Qwen3.6-35B-A3B | Full 282-name native-HF warmth/competence and callback audit | Complete; positive human alignment and positive callback associations on both axes. |
| `2026-07-19_1006_qwen36_35b_a3b_posthoc_hiring.md` | 2026-07-19 10:06 | Qwen3.6-35B-A3B | Demographic disparity, mediation, group R4, and name-level R4 | Complete; warmth mediation intervals exclude zero, while name-level model-human callback alignment is absent. |
| `2026-07-19_1008_qwen36_27b_denoised_hiring.md` | 2026-07-19 10:08 | Qwen3.6-27B | Neutral-PCA-denoised local hiring steering across 60 names | Complete; both axes remain monotone and positive, with moderate attenuation. |
| `2026-07-19_1008_qwen36_27b_full282_gate.md` | 2026-07-19 10:08 | Qwen3.6-27B | Predeclared conditional expansion gate | Gate did not fire; no 282-name steering expansions are required. |
| `2026-07-19_1011_qwen36_35b_a3b_local_hiring.md` | 2026-07-19 10:11 | Qwen3.6-35B-A3B | Raw local-strength hiring steering across 60 names | Complete; both axes are monotone with positive, bootstrap-resolved endpoints. |
| `2026-07-19_1014_qwen36_35b_a3b_broad_hiring.md` | 2026-07-19 10:14 | Qwen3.6-35B-A3B | Raw broad-strength hiring steering across 60 names | Complete; warmth is non-monotone and competence reverses at +0.50, triggering conditional expansion. |
| `2026-07-19_1017_qwen36_full282_runner.md` | 2026-07-19 10:17 | Qwen3.6-27B · Qwen3.6-35B-A3B | Independent resumable full-282 Qwen hiring runner | Implemented; three gate-controlled regimes have separate checkpoints, labels, validators, and sentinels. |
| `2026-07-19_1024_qwen36_35b_a3b_neutral_extraction.md` | 2026-07-19 10:24 | Qwen3.6-35B-A3B | Native-HF neutral-corpus activation extraction | Complete; validated finite 1500×2048 matrix with exact remote-to-local hash parity. |
| `2026-07-19_1024_qwen36_35b_a3b_pca_denoising.md` | 2026-07-19 10:24 | Qwen3.6-35B-A3B | Neutral-PCA denoising of warmth and competence vectors | Complete; 17 components reduce cross-axis cosine modestly but increase leakage. |
| `2026-07-19_1027_qwen36_35b_a3b_denoised_hiring.md` | 2026-07-19 10:27 | Qwen3.6-35B-A3B | Neutral-PCA-denoised local hiring steering across 60 names | Complete; both axes remain monotone and positive with negligible attenuation. |
| `2026-07-19_1027_qwen36_35b_a3b_full282_gate.md` | 2026-07-19 10:27 | Qwen3.6-35B-A3B | Predeclared conditional expansion gate | Gate fired on four broad-regime criteria; three 282-name runs required. |
| `2026-07-19_1039_qwen36_full282_validator_fix.md` | 2026-07-19 10:39 | Qwen3.6-35B-A3B | Backward-compatible full-282 validator and published-output recovery | Implemented; completed 2,820-row output preserved for fixed-commit validation without recomputation. |
| `2026-07-19_1041_qwen36_35b_a3b_local_full282.md` | 2026-07-19 10:41 | Qwen3.6-35B-A3B | Raw local-strength steering across all 282 names | Complete; fixed-commit recovery validates monotone positive effects without recomputation. |
| `2026-07-19_1053_qwen36_35b_a3b_broad_full282.md` | 2026-07-19 10:53 | Qwen3.6-35B-A3B | Raw broad-strength steering across all 282 names | Complete; competence reversal reproduces almost exactly and is not a small-panel artifact. |
| `2026-07-19_1105_qwen36_35b_a3b_denoised_full282.md` | 2026-07-19 11:05 | Qwen3.6-35B-A3B | Denoised local steering across all 282 names | Complete; stable positive local effects survive both panel expansion and PCA transformation. |
| `2026-07-19_1156_qwen36_27b_local_full282.md` | 2026-07-19 11:56 | Qwen3.6-27B | Raw local-strength steering across all 282 names | Complete post-hoc balanced replication; both axes remain monotone and closely reproduce the 60-name estimates. |
| `2026-07-19_1205_qwen36_27b_broad_full282.md` | 2026-07-19 12:05 | Qwen3.6-27B | Raw broad-strength steering across all 282 names | Complete post-hoc balanced replication; both axes stay monotone and positive, unlike the 35B competence reversal. |
| `2026-07-19_1214_qwen36_27b_denoised_full282.md` | 2026-07-19 12:14 | Qwen3.6-27B | Denoised local steering across all 282 names | Complete post-hoc balanced replication; both axes remain monotone and reproduce the 60-name estimates. |
| `2026-07-19_1215_qwen36_27b_full282_balanced_synthesis.md` | 2026-07-19 12:15 | Qwen3.6-27B | Original 60-name versus post-hoc 282-name steering synthesis | Complete; all six endpoint estimates change by less than 3%, with signs and monotonicity preserved. |
| `2026-07-19_1223_qwen36_27b_calibrated_steering.md` | 2026-07-19 12:23 | Qwen3.6-27B | Topic-corrected SD-matched target, cross-axis, and 99-direction random-control steering | Complete; target directions exceed the random null, while strong cross-axis effects limit axis-specific interpretation. |
| `2026-07-19_1224_qwen36_35b_a3b_calibrated_steering.md` | 2026-07-19 12:24 | Qwen3.6-35B-A3B | SD-matched target, cross-axis, and 99-direction random-control steering | Complete; strong target and cross-axis effects exceed the random null under both intervention geometries. |
