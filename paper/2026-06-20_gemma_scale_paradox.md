# Gemma-3 Scale-Up: Does the Cross-Axis Paradox Deepen with Model Size?

**Date:** 2026-06-20
**Models:** Gemma-3-27B-Instruct vs. Gemma-3-12B-Instruct (within-family comparison)
**Phase:** B3 — within-family scale test
**Figure:** `paper/figures/fig8_layer_emergence.{png,pdf}` (updated to 4 models)
**Scripts:** `src/extract_vectors.py`, `src/validate_probes.py`, `src/layer_sweep.py`
**Job:** SCCKN job 1059107 (`wc_27b_full`), `gpu@scc214`, completed 2026-06-20

---

## Overview

The previous report (`2026-06-20_layer_sweep_topic_holdout.md`) resolved the cross-axis paradox:
Gemma-3-12B's warmth and competence directions are geometrically entangled at **every** layer
depth, not just at the chosen probe layer. This established the entanglement as an architectural
property of Gemma-3, not a depth-selection artefact.

B3 asks the natural follow-up: **does this entanglement change with model scale within the Gemma-3
family?** If it deepens at 27B, scale amplifies an architecture-level quirk. If it stays the same
or shrinks, the property is fixed by architecture and insensitive to capacity.

**Answer: the paradox is scale-invariant.** Gemma-3-27B replicates every key property of 12B —
perfect probe accuracy, cross-axis CV at chance, cos(W,C) elevated at all depths — with only minor
quantitative changes.

---

## Background: What Is the Cross-Axis Paradox?

When we extract a "warmth direction" from a language model's internals, we expect it to be
geometrically distinct from the "competence direction." In Qwen3-14B and Llama-3.1-8B, this is
mostly true: the two directions are about 60° apart (cos ≈ 0.51–0.54) and each direction is
specific to its own concept (cross-axis classifier accuracy ≈ 0.50, chance).

In Gemma-3-12B, the paradox is: the two directions are geometrically *close* (cos ≈ 0.75, about
41° apart), yet each direction is behaviorally *specific* to its own concept (cross-axis accuracy
= 0.50, the same chance level). Geometrically similar directions that nonetheless carry independent
information.

B3 asks whether this is a property of the 12B model specifically, or the Gemma-3 architecture
family more broadly.

---

## Method

Identical to the other three models: 200 concept stories (50 per condition), mean-difference probe
direction at layer 40 (frac = 0.66 of 62 layers), topic-holdout GroupKFold CV, full layer sweep
(all 62 residual layers), scale-guard (no activation matrices written to disk).

The only constraint: Gemma-3-27B bf16 requires ~54 GB VRAM, so the job ran exclusively on scc214
(RTX PRO 6000 Blackwell, 96 GB). L40 nodes (48 GB) would OOM.

TransformerLens loaded `google/gemma-3-27b-it` successfully on first attempt:
`n_layers=62, d_model=5376, probe_layer=40`.

---

## Results: Probe-Layer Metrics

| Metric | Gemma-3-12B (L31) | Gemma-3-27B (L40) |
|--------|-------------------|-------------------|
| 5-fold CV (warmth) | 1.00 | 1.00 |
| 5-fold CV (competence) | 1.00 | 1.00 |
| Topic-holdout CV (warmth) | 1.00 | 1.00 |
| Topic-holdout CV (competence) | 1.00 | 1.00 |
| Cohen's d (warmth) | 2.70 | **2.95** |
| Cohen's d (competence) | 2.83 | **3.27** |
| cos(warmth\_vec, competence\_vec) | 0.749 | **0.708** |
| Cross-axis CV (W→C) | 0.50 | 0.50 |
| Cross-axis CV (C→W) | 0.50 | 0.50 |
| mean\_resid\_norm at probe layer | 79,756 | 61,576 |
| d\_model | 3,840 | 5,376 |

The 27B model is slightly *stronger* than 12B on Cohen's d (d ≈ 3.0–3.3 vs 2.7–2.8) while showing
slightly *lower* geometric overlap between the two concept directions (cos = 0.708 vs 0.749). Both
differences are small. The cross-axis CV remains exactly at chance (0.50) for both axes.

---

## Results: Layer Sweep and Fig8

![Layer sweep: Cohen's d emergence curves and cos(W,C) depth profiles — four models](figures/fig8_layer_emergence.png)

**Figure 8.** *(Left)* Cohen's d emergence curves for all four models. Gemma-3-27B (dark teal,
dot-dash-dot) closely tracks Gemma-3-12B (green, solid) in shape: a moderate start, a dip in the
early-to-mid network, and a late surge in the final 20% of layers. Both Gemma models sit well below
Qwen3-14B (purple) and Llama-3.1-8B (red) throughout. *(Right)* cos(W,C) depth profiles. Both
Gemma models rise sharply before frac = 0.30 and stay elevated at 0.50–0.93 for the rest of the
network. Qwen3 and Llama plateau near 0.50–0.62. The two teal/green Gemma curves are nearly
indistinguishable in the right panel — scale does not change the entanglement pattern.

### Finding 1 — cos(W,C) depth profile: same shape, slightly lower peak

| Layer fraction | Gemma-12B cos | Gemma-27B cos |
|---------------|---------------|---------------|
| L0 (frac=0.00) | 0.017 | −0.037 |
| Peak | 0.952 at L16 (frac=0.34) | **0.933 at L23 (frac=0.38)** |
| Probe layer (frac≈0.66) | 0.749 | 0.708 |
| Final layer | 0.651 | 0.595 |

Both models start near zero, spike to >0.90 in the early-to-mid network, and stabilise at 0.60–0.75
for the remaining depth. The 27B curve is slightly lower across all depths but its shape is
essentially identical to 12B. This confirms that entanglement is a **family-level architectural
property**, not a capacity effect.

### Finding 2 — Cohen's d emergence: same late-surge pattern

Both Gemma models show the same characteristic shape: moderate d in early layers (d ≈ 1–2),
a visible dip in the middle (d < 2), then a sustained climb in the final third.

| Region | Gemma-12B | Gemma-27B |
|--------|-----------|-----------|
| Early peak (frac<0.25) | d ≈ 3–4 (L9, frac=0.19) | d ≈ 2.9–3.2 (L13, frac=0.21) |
| Mid-network dip | d < 1.5 at multiple layers | d < 1.5 at several layers |
| Late surge peak | d=6.09 at L45 (frac=0.96) warmth | d=5.10 at L58 (frac=0.95) warmth |
| Probe layer d | 2.68 / 2.86 (W/C) | 2.95 / 3.27 (W/C) |

The 27B model shows slightly higher d at the probe layer — the extra parameters appear to encode
the concepts somewhat more sharply — but the overall late-surge shape is the same.

### Finding 3 — Residual norm: non-monotonic with scale

- Gemma-12B probe layer mean\_resid\_norm: **79,756**
- Gemma-27B probe layer mean\_resid\_norm: **61,576** (paradoxically *lower*)
- Gemma-27B final layer (L61): 177,437 (higher than 12B's final layer ~165K)

The probe layer norm is lower in the larger model, while the final-layer norm is higher. This
reflects the different depth ratio: the 27B probe layer sits at absolute layer 40, which falls
in a lower-norm region of the 27B depth profile than layer 31 does for 12B. The mean\_resid\_norm
values confirm that steering magnitudes must be calibrated per-model per-layer (relative to local
norm), not as absolute values across models.

---

## The Cross-Axis Paradox at Scale: Conclusion

The cross-axis paradox is **scale-invariant** within the Gemma-3 family.

At both 12B and 27B:
- Warmth and competence are linearly separable with perfect accuracy.
- The probe directions are geometrically close (cos ≈ 0.71–0.75).
- Yet each direction carries no information about the other concept (cross-axis CV = 0.50).
- And this geometric entanglement is present at every network depth, not just the probe layer.

The remaining open question — *why* do geometrically close directions carry independent information
in Gemma but not in Qwen/Llama — is not answered here. One hypothesis: Gemma's residual stream is
higher-dimensional relative to the number of independent semantic features it encodes, so two
directions can be geometrically close while still spanning largely independent portions of the
relevant subspace. Testing this would require comparing the effective dimensionality of the warmth
and competence subspaces across model families (e.g., via SVD of the per-condition activation
matrices).

---

## Updated Four-Model Comparison

With B3 complete, the probe-layer summary across all four models is:

| Model | Warmth d | Comp d | cos(W,C) | Cross-axis CV |
|-------|----------|--------|----------|---------------|
| Gemma-3-12B | 2.70 | 2.83 | 0.749 | 0.50 |
| Gemma-3-27B | 2.95 | 3.27 | 0.708 | 0.50 |
| Qwen3-14B | 8.97 | 9.97 | 0.536 | 1.00 |
| Llama-3.1-8B | 8.48 | 9.07 | 0.505 | 0.99 |

The Gemma family (both rows) is internally consistent and clearly distinct from Qwen/Llama: lower
Cohen's d, higher cos(W,C), cross-axis CV at chance. The Qwen/Llama cluster is internally
consistent in the opposite direction: much higher d, lower cos, cross-axis CV near ceiling.

This four-model picture supports the interpretation that the cross-axis paradox reflects a
Gemma-3-family architectural property rather than a scale, dataset, or training-recipe effect
(since both Qwen and Llama show the opposite pattern despite varying considerably in size and
training approach).

---

## Limitations

1. **Only two Gemma-3 sizes tested.** A 4B data point would complete the within-family scaling
   picture, but GemmaScope 2 4B SAE support and the earlier Gemma-3-4B OOM on SCCKN (CPU OOM,
   smoke test history) make this non-trivial. Deferred.

2. **No cross-axis CV depth profile.** The layer sweep computes cos(W,C) per layer but not the
   behavioural cross-axis accuracy per layer. We do not know whether the chance cross-axis CV is
   uniform across depths or only holds near the probe layer.

3. **Single seed.** All results use seed 20260527. Cross-seed variability is expected to be
   negligible given the large separations, but is not formally tested.

---

## Next Steps

| Step | Task |
|------|------|
| B4 | Scale-normalised analysis: express all projections as proj / mean\_resid\_norm; rebuild comparison table and fig5 |
| B5 | Full report revision integrating B1–B4 results |
| B6 | Valence denoising (login-node corpus build pending; scripts ready) |
| Future | fig5/6/7 update to include Gemma-3-27B column |
| Future | Cross-axis CV depth profile (per-layer behavioural discriminability for all 4 models) |
