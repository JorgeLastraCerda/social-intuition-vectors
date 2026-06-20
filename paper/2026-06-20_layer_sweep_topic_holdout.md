# Layer Sweep and Topic-Holdout Validation

**Date:** 2026-06-20  
**Models:** Gemma-3-12B-Instruct, Qwen3-14B, Llama-3.1-8B-Instruct  
**Phase:** B1 (topic-holdout cross-validation) + B2 (layer sweep)  
**Figure:** `paper/figures/fig8_layer_emergence.{png,pdf}`  
**Scripts:** `src/validate_probes.py` (B1), `src/layer_sweep.py` (B2)

---

## Overview

This report covers two extensions to the initial three-model probing result
(`2026-06-19_cross_model_concept_findings.md`): a stricter validation that rules out
a specific type of memorisation (B1), and a scan across every layer of each network
to find where warmth and competence representations first appear and how strong they
get (B2).

The headline finding from B2 answers the open question left by the previous report:
why does Gemma-3-12B show higher warmth/competence overlap (cos = 0.75) than Qwen3-14B
(0.54) or Llama-3.1-8B (0.51), even though Gemma's cross-axis discriminability is
*lower*?  The layer sweep shows that Gemma's entanglement is not specific to the probe
layer — it is present at every depth.  The paradox is an architectural feature, not a
depth-selection artifact.

---

## Background: What Are We Probing?

A language model is a sequence of computational layers.  Each layer transforms an
internal representation of the text — called the *residual stream* — into a slightly
updated version.  By the final layer, the model's "understanding" of a passage is
encoded in a high-dimensional vector of floating-point numbers.

We want to know whether warmth and competence — the two core dimensions of social
judgment in the Stereotype Content Model — are encoded as *linear directions* in that
vector space.  If they are, we can extract them, measure them, and eventually steer
them.

To extract a direction, we do this for each concept (warmth and competence separately):
1. Show the model 100 stories that express the concept strongly.
2. Show the model 100 matched stories where the concept is absent.
3. Take the average internal representation for each group and subtract one from the
   other.  The resulting vector is the *probe direction*.

A probe is useful if it is *linearly separable*: a simple one-feature classifier
trained on the projection of stories onto this direction should be able to tell high
from low stories reliably.

The key quality metric used in this report is **Cohen's d**: the gap between the two
groups (high vs. low) in units of their combined spread.  A d of 0.8 is conventionally
"large"; values above 2 are very large; the models here reach d = 8–11.

---

## B1 — Topic-Holdout Cross-Validation

### The problem with standard cross-validation

In our previous results, a standard 5-fold cross-validation gave 100% accuracy on all
three models.  A sceptical reader might wonder: is the probe really detecting *warmth*,
or is it exploiting incidental vocabulary shared between the training and test splits?

Our 200 stories are drawn from 50 *topics* (e.g., "helping a neighbour move house").
Each topic has exactly one high-warmth and one low-warmth story, and the same 50 topics
are reused for the competence axis.  A standard random split can place the
high-warmth version of "helping a neighbour move house" in training and its low-warmth
counterpart in the test set.  Those two stories share lots of words — move, boxes,
neighbour — that the probe might pick up on without understanding warmth at all.

### The fix: hold out entire topics

Topic-holdout validation uses `GroupKFold`: every story from a given topic goes into
either training or test, never both.  In the test fold, the model sees situations it
has *never* encountered in training.  This tests whether the probe generalises to new
situations, not just new phrasings of familiar situations.

With 50 topics and 5 folds, each test fold contains 10 entirely new topics.

### Result

**All three models: topic-holdout CV = 1.00 (± 0.00) on both axes.**

This is not a ceiling artefact to dismiss — it is a positive finding.  The very large
Cohen's d values (d = 8–11 for Qwen and Llama, d = 2.7 for Gemma at the probe layer)
predict that the separation is wide enough to survive even the strictest topic holdout.
The probe is not exploiting topic vocabulary; it is capturing something that generalises
across entirely new social situations.

Interpretation: warmth and competence are encoded in a way that is robust to the
specific scenario.  A representation of "warmth" in a moving-house story looks similar
enough to warmth in a hospital story that a linear classifier trained on one set
transfers perfectly to the other.

---

## B2 — Layer Sweep

### What is a layer sweep?

Instead of probing one fixed layer (we have been using 66% of the way through the
network, following the Sofroniew & Lindsey 2026 convention), the layer sweep probes
*every* layer.

For each layer and each model, we extract residual-stream activations, build warmth and
competence probe directions using the same mean-difference procedure, and compute:
- **Topic-holdout CV** (same strict validation as B1)
- **Cohen's d** (separation strength)
- **cos(warmth_vec, competence_vec)** (how geometrically entangled the two concepts are)
- **mean residual norm** (the typical vector length at that layer — needed for scale-normalised steering in B4)

All layers are captured in a single forward pass per story, so the cost is comparable
to a single extraction run.  No activation matrices are written to disk; only the
per-layer metrics CSV is saved (scale guard).

### Topic-holdout CV across layers

Topic-holdout CV = 1.00 at every layer beyond a shallow threshold (approximately
frac = 0.2 for all models).  At the very first layer (frac = 0), CV is already high:
0.86–0.93.  This means the signals for warmth and competence are present even in the
earliest transformations of the input — an unexpectedly early emergence.

Because CV = 1.00 is a ceiling everywhere, it cannot rank layers.  **Cohen's d is the
discriminative metric for the rest of this analysis.**

---

## Results: Cohen's d Emergence Curves

See Figure 8 (left panel) for the full emergence curves.

**Table: Cohen's d at selected layers**

| Model | L0 (frac=0.00) | Peak (frac) | Probe layer | Peak d (warmth / comp) |
|-------|---------------|-------------|-------------|------------------------|
| Gemma-3-12B | 1.60 / 2.80 | L45 (0.96) | L31 (0.66): 2.68 / 2.86 | 6.09 / 4.38 |
| Qwen3-14B | 2.80 / 3.72 | L25 (0.64) | L26 (0.67): 8.97 / 9.97 | 9.92 / 10.81 |
| Llama-3.1-8B | 4.11 / 3.90 | L10–14 | L20 (0.65): 8.48 / 9.07 | 10.61 / 11.55 |

### Finding 1 — Representations emerge early

For Qwen3-14B and Llama-3.1-8B, Cohen's d exceeds 4.0 before frac = 0.15 (within the
first sixth of the network).  By frac = 0.25, both models surpass d = 6.0.  The probed
concepts are not formed deep in the network through many layers of abstraction; they
are already present in the network's early processing.

This is consistent with the hypothesis that warmth and competence are partly encoded
via surface linguistic features (word choice, sentence structure) that are established
early.  However, the fact that topic-holdout CV is also 1.00 at these early layers
means the early representations already generalise across topics, suggesting they
capture something more than superficial vocabulary.

### Finding 2 — Different emergence profiles by architecture

The three models show qualitatively different Cohen's d trajectories:

- **Llama-3.1-8B** (32 layers): rises steeply to peak around L10–14 (frac = 0.32–0.45,
  d ≈ 10–11.5), then declines gradually.  The probe layer (L20, frac = 0.65) sits on
  the descending slope with d ≈ 8.5–9.1 — still very strong.

- **Qwen3-14B** (40 layers): rises to peak around L22–25 (frac = 0.56–0.64,
  d ≈ 9.9–10.8), then declines.  The probe layer (L26, frac = 0.67) is almost exactly
  at the peak.

- **Gemma-3-12B** (48 layers): a completely different shape.  Early layers show
  moderate d (1–3), the middle of the network dips (d < 1.5 at several layers), and
  d surges in the final 20% of the network (reaching d = 6.09 for warmth at L45,
  frac = 0.96).  The probe layer (L31, frac = 0.66) sits in the middle of this surge
  with d ≈ 2.7–2.9.

The practical implication: for Gemma, deeper layers would yield stronger probe
directions.  For Qwen and Llama, frac = 0.66 is near-optimal.

### Finding 3 — Probe layer frac = 0.66 is valid, not optimal for all models

The probe layer is well within the "plateau" region for Qwen and Llama, where d is
high and stable across a wide range of depths.  A different probe layer in the range
0.4–0.8 would give similar results.

For Gemma, frac = 0.66 captures the beginning of the late surge, not the peak.  The
scientific results reported in this study (topic-holdout CV, cross-axis behaviour) are
unaffected — CV = 1.00 everywhere — but d at the probe layer (2.7) is well below the
layer-sweep maximum (6.1).  If steering vectors are built from the probe layer
activations, Gemma's steering efficiency will be lower than Qwen/Llama.  A follow-up
should test frac = 0.90 for Gemma-specific experiments.

---

## The Cross-Axis Paradox: Resolution

### What the paradox was

The previous report noted a paradox:

- Gemma has **high geometric overlap** between the warmth and competence directions
  (cos = 0.749), but **low cross-axis behavioural discriminability** (the warmth probe
  does not confidently classify competence stories and vice versa).
- Qwen and Llama have **lower overlap** (cos ≈ 0.51–0.54), but **high cross-axis
  discriminability** (the warmth probe partially predicts competence stories).

Two hypotheses were proposed:
- **H2 (depth effect):** the high cosine in Gemma is an artefact of which layer we
  chose (frac = 0.66).  At a different layer, the axes would separate.
- **H3 (architectural effect):** the entanglement is a consistent property of the
  Gemma family's representations, not a layer-specific accident.

### What the layer sweep shows

See Figure 8 (right panel) for the cos(W,C) depth profiles.

**Gemma:** cosine starts near 0 at L0 (cos = 0.017), rises steeply through the early
and middle layers (reaching 0.95 at L16, frac = 0.34), then remains elevated — between
0.49 and 0.84 — for the final two-thirds of the network.  There is no layer where
Gemma's cos(W,C) drops to the 0.30 target.  The probe layer (L31, cos = 0.749) is
representative of the model's overall geometry, not an outlier.

**Qwen3-14B:** cosine rises from 0.098 to a peak of 0.622 (L23, frac = 0.59), then
declines to 0.550 at the final layer.  It never exceeds 0.63.

**Llama-3.1-8B:** cosine rises from 0.174 to 0.581 (L12, frac = 0.39), then plateaus
remarkably stably at 0.50–0.52 for the remainder of the network.

**Conclusion: H2 is falsified; H3 is supported.**  Gemma's warmth and competence
directions are persistently entangled at every depth.  This is an architectural or
training property of Gemma-3, not an accident of layer choice.

The remaining puzzle — why high geometric overlap co-exists with low cross-axis
discriminability in Gemma but not in Qwen/Llama — is not resolved here.  One
possibility is that Gemma's probe directions, while geometrically close, each point
to a portion of the space the opposite probe does not reach, maintaining axis-specific
discriminability.  Another possibility is that the effective dimensionality of the
relevant subspace differs.  This is a target for follow-up analysis (potentially via
SAE decomposition or PCA on the combined warmth+competence projection space).

---

## Residual Norm Variation

The mean residual-stream norm at the probe layer varies enormously across models:

| Model | mean_resid_norm at probe layer |
|-------|-------------------------------|
| Gemma-3-12B (L31) | 79,756 |
| Qwen3-14B (L26) | 207 |
| Llama-3.1-8B (L20) | 11.4 |

This ~7,000-fold range means that raw projection values (warmth score = activation
dot probe direction) are not comparable across models.  Steering vectors calibrated
in absolute magnitude for one model would be orders of magnitude too weak or too
strong on another.

All steering magnitudes in the main pipeline are already expressed as a multiple of
`mean_resid_norm` at the steered layer (AGENTS.md hard constraint), so this does not
affect existing results.  The layer sweep's per-layer norm column provides the values
needed to compute relative steering strengths across the full depth profile for B4
(scale-normalised analysis).

---

## Limitations

1. **Single seed / single sample.**  Each layer sweep uses the same 200 stories as the
   probe extraction.  The mean-difference direction at each layer has no confidence
   interval.  A bootstrap or leave-one-out extension would add uncertainty bounds to
   the d curves.

2. **Topic-holdout CV = 1.00 everywhere is informative but does not locate the optimal
   layer.**  Cohen's d is the signal for depth selection; it is computed from in-sample
   directions and should be treated as approximate rather than a ground-truth ranking.

3. **Early-layer d may reflect surface features.**  High Cohen's d at frac < 0.1 is
   consistent with the model encoding warmth/competence cues in vocabulary and syntax,
   which are established early.  Whether these early directions reflect the same
   concept as the deeper ones (or a lexical proxy) is not tested here.

4. **Cross-axis CV depth profile not yet computed.**  The sweep records cos(W,C) per
   layer, but not the behavioural cross-axis accuracy (how well the warmth probe
   classifies competence stories, at each layer).  This would more directly test
   whether the cross-axis paradox is constant across depth or localised.

---

## Next Steps

| Step | Task |
|------|------|
| B3 | Gemma-3-27B-Instruct sweep — does the paradox deepen with scale within the Gemma family? |
| B4 | Scale-normalised analysis using per-layer `mean_resid_norm` from this sweep |
| B5 | Full report revision incorporating B1–B4 results |
| B6 | Valence denoising (scripts ready; corpus build + extraction pending on SCCKN) |
| Future | Cross-axis CV depth profile (per-layer behavioural discriminability) |
| Future | Late-layer Gemma probe (frac = 0.90) for stronger steering directions |
