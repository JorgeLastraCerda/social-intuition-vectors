# Results Cheatsheet — Plain Language Guide to All Statistics

This document explains every statistic used in the paper in simple terms,
then lists the actual values from the project with what they mean.
Written for Jorge to use when writing or presenting the results.

---

## Part 1: What Each Statistic Means

### Cohen's d

**What it is:** A way of measuring how far apart two groups are, in units
of the spread (standard deviation) of the data. It answers: "how separated
are the two groups, ignoring the scale of the original numbers?"

**How to read it:**
- d = 0.2 → small difference (groups overlap a lot)
- d = 0.5 → medium difference
- d = 0.8 → large difference
- d > 2.0 → the groups are extremely far apart with almost no overlap

**In our study:** We measure Cohen's d between the "high warmth" stories and
the "low warmth" stories in terms of how much they differ along the warmth
direction in the model's internal space. A high d means the model is sharply
distinguishing warm from cold stories.

**Our values:**
| Model | Warmth d | Competence d | What it means |
|-------|----------|--------------|---------------|
| Gemma-3-12B | 2.70 | 2.83 | Very large separation |
| Gemma-3-27B | 2.95 | 3.27 | Very large separation, slightly stronger than 12B |
| Llama-3.1-8B | 8.48 | 9.07 | Enormous separation |
| Qwen3-14B | 8.97 | 9.97 | Enormous separation |

Gemma's d values look small compared to Llama/Qwen, but all values are far
above the d = 0.8 "large" threshold. The difference reflects how each model
organizes its internal space, not that Gemma is doing worse. The z-scores
(how many standard deviations above the random baseline) all exceed 3.7, and
zero out of 1000 random directions beat the warmth/competence direction in any
model.

---

### p-value

**What it is:** The probability of seeing a result as extreme as ours if
there were actually no real effect. A small p-value means the result is very
unlikely to be a coincidence.

**How to read it:**
- p < 0.05 → statistically significant (conventional threshold)
- p < 0.001 → very strong evidence (1 in 1000 chance this is noise)
- p > 0.05 → not statistically significant ("could be noise")

**Important caveat:** p-values are not the size of the effect. A tiny effect
can have a tiny p-value if you have enough data. A large effect can have a
large p-value if you have very little data. Always look at the effect size too.

**Our values:** All Cohen's d results have p < 0.001 (zero out of 1000 random
directions beat us). For Spearman correlations, p-values are listed in the table
in Part 2.

---

### Spearman rho (ρ)

**What it is:** A measure of how well two rankings agree with each other,
from -1 to +1. It does not assume a straight-line relationship; it only
checks if higher values on one scale tend to go with higher values on the
other. Named after Charles Spearman.

**How to read it:**
- ρ = +1.0 → perfect positive agreement (when X goes up, Y always goes up)
- ρ = 0 → no relationship
- ρ = -1.0 → perfect negative agreement (when X goes up, Y always goes down)
- ρ ≈ ±0.1 → weak relationship
- ρ ≈ ±0.3 → moderate relationship
- ρ ≈ ±0.5 → strong relationship

**In our study, we use Spearman rho for two things:**

**1. Probe-vs-human alignment** (how well the model's internal warmth score
matches human warmth ratings for the 282 names):

| Model | Warmth ρ | p-value | Competence ρ | p-value |
|-------|----------|---------|--------------|---------|
| Gemma-3-12B | **+0.366** | <0.001 | +0.239 | <0.001 |
| Gemma-3-27B | **+0.396** | <0.001 | +0.272 | <0.001 |
| Llama-3.1-8B | **−0.300** | <0.001 | −0.063 | 0.29 (n.s.) |
| Qwen3-14B | **−0.193** | 0.001 | **+0.465** | <0.001 |

The Gemma models have POSITIVE warmth ρ: names that humans rate as warm also
score high on warmth inside the model. This is good alignment.

Llama and Qwen have NEGATIVE warmth ρ: names that humans rate as warm score
LOW on warmth inside these models. The direction is flipped. Both models still
"know" what warmth is (they separate stories perfectly), but their warmth axis
is pointing in the opposite direction compared to human warmth ratings.
We think instruction tuning (RLHF) inverted this relationship in these models.

**2. Cross-model story agreement** (do two models agree on which stories are
warmer than others?):

| Pair | Warmth ρ | Competence ρ |
|------|----------|--------------|
| Gemma-12B ↔ Qwen3-14B | 0.760 | 0.795 |
| Gemma-12B ↔ Llama-3.1-8B | 0.768 | 0.782 |
| Gemma-12B ↔ Gemma-27B | ~0.95–0.99 (same family) | similar |

High agreement (ρ ≈ 0.76–0.99) means all four models produce nearly the same
ranking of stories by warmth/competence. Yet Llama and Qwen anti-align with
humans. This is the key paradox: the models agree with each other, but what
they agree on is pointing away from the human warmth scale.

---

### R² (R-squared)

**What it is:** The fraction of variation in the outcome that is explained
by the predictor in a regression model. Goes from 0 to 1.

**How to read it:**
- R² = 0 → the predictor explains nothing
- R² = 0.5 → the predictor explains 50% of the variation
- R² = 1.0 → the predictor explains everything perfectly

**In our study:** We use R² to measure how linear the dose-response curve is
in our steering experiments. A high R² means: when we push the warmth direction
up by a certain amount, the model's response scales predictably. A low R² means
the response is erratic or non-linear.

**Our values:**

*Concept-level steering (does warmth steering change the model's warmth
judgements on stories?)*

| Model × Axis | Slope | R² | What it means |
|---|---|---|---|
| 12B warmth | 27.79 | 0.956 | Very linear, strong positive effect |
| 12B competence | 15.11 | 0.915 | Very linear, strong positive effect |
| 27B warmth | 12.89 | 0.990 | Even more linear, but smaller absolute effect |
| 27B competence | 8.85 | 0.826 | Mostly linear but weaker |

*Hiring-level steering (does warmth steering change callback decisions?)*

| Model × Axis | Effect at +0.10 | Pattern |
|---|---|---|
| 12B warmth | +2.369 Δmargin | Strong, approximately linear |
| 12B competence | +0.231 Δmargin | Weak positive on + side, strong negative on − side |
| 27B warmth | +1.973 at +0.05, then −2.658 at +0.10 | Non-monotone (reversal!) |
| 27B competence | −4.858 at −0.10, +0.167 at +0.05, −2.696 at +0.10 | Non-linear |

---

### Regression slope (β)

**What it is:** In a regression, the slope tells you how much the outcome
changes for each one-unit increase in the predictor. If the slope is 12,
it means "for every extra unit of input, the output goes up by 12."

**In our study:** The steering slopes are in units of "Δcallback margin per
unit of mean residual norm × strength." Because different models have wildly
different activation scales, we normalize by the mean residual-stream norm
(see table below), so slopes are comparable across models.

| Model | Mean residual norm (||h||) |
|-------|--------------------------|
| Gemma-3-12B | 79,722 |
| Gemma-3-27B | 61,576 |
| Llama-3.1-8B | 11.4 |
| Qwen3-14B | 206.6 |

This is why Llama and Gemma cannot be directly compared using raw steering
magnitudes: Llama's activations are thousands of times smaller in absolute scale.

---

### Normalized steerability

**What it is:** A measure of how well a model's concept representations can
be manipulated by steering, controlling for the model's baseline ability to
distinguish the concept. Computed as:

> (steering effect at peak strength) ÷ (baseline high-vs-low logit gap)

This answers: "relative to how well the model already separates warm from cold,
how much does our steering move the needle?"

**How to read it:**
- Close to 1 → steering matches the full baseline concept separation
- Close to 0 → steering barely moves the representations relative to baseline

**Our values (warmth at α = +0.10):**

| Model | Steering effect | Baseline gap | Normalized steerability |
|-------|----------------|--------------|------------------------|
| Gemma-3-12B | +3.81 | 16.14 | **0.236** (strongest) |
| Qwen3-14B | +1.24 | 9.91 | 0.125 |
| Gemma-3-27B | +1.03 | 25.93 | 0.040 |
| Llama-3.1-8B | +0.26 | 9.01 | **0.029** (weakest) |

For competence:

| Model | Normalized steerability |
|-------|------------------------|
| Gemma-3-12B | 0.140 |
| Qwen3-14B | 0.103 |
| Llama-3.1-8B | 0.024 |
| Gemma-3-27B | 0.009 |

**The steerability paradox:** Gemma-3-12B is the most steerable at the concept
level (0.236) but shows NO significant mediation path to hiring decisions.
Llama-3.1-8B is the least steerable (0.029) but shows the STRONGEST hiring
mediation (IE = +0.190). Being easy to steer at the concept level does not
mean that steering translates into a causal effect on hiring.

---

### Bootstrap indirect effect (IE) and mediation

**What it is:** Mediation analysis asks: does the model's internal warmth
score (the "mediator") help explain the path from a name's demographic group
to the callback decision? The indirect effect (IE) is the part of the
group-to-callback relationship that travels through the internal probe score.
Bootstrap confidence intervals are computed by resampling the data 1000+ times.

**In simple terms:** If a name's group affects both (a) how warmly the model
represents it internally and (b) the callback decision, and if the internal
warmth score also predicts the callback decision, then warmth mediates the
effect of group membership on callbacks. The IE measures how large that
mediated pathway is.

**Our key value:**
- Llama-3.1-8B race × warmth indirect effect: **IE = +0.190**, 95% CI
  excludes zero. This is a large mediation effect: for Llama, the warmth
  representation carries a significant causal share of the racial callback gap.

---

## Part 2: Key Numbers by Result Section

### R1 — Linear Encoding

All four models achieve 100% accuracy in 5-fold cross-validation and
topic-holdout cross-validation. The Cohen's d values are all far above
d = 0.8 (large effect). The separation is real and not a result of overfitting
to specific story phrasings.

**PCA denoising results:**
After removing the "general positivity" component from the direction vectors:

| Model | Cosine before | Cosine after | Components removed | Variance explained |
|-------|--------------|--------------|-------------------|-------------------|
| Gemma-3-12B | 0.749 | 0.530 | 1 | 56.1% of neutral variance |
| Gemma-3-27B | 0.708 | 0.487 | 43 | 50.2% of neutral variance |

Removing 1 component (at 12B) dramatically reduces the overlap because
that single component is a strong "positive framing" dimension in the
neutral Wikipedia text. At 27B, the same information is spread across 43
weaker components. After denoising, the remaining cosine (~0.53 and ~0.49)
reflects real conceptual correlation between warmth and competence, consistent
with human psychology (humans also tend to rate warm groups as competent;
r ≈ 0.61 in meta-analyses).

---

### R2 — Concept-Level Causal Steering

At both Gemma models, directly shifting the warmth or competence direction
vector in the model's activations reliably changes the model's explicit
warmth/competence judgements. This is "causal" evidence: we are
manipulating the internal representation, not just correlating with it.

The high R² values (0.826–0.990) mean the dose-response curve is smooth
and predictable: push the warmth direction up by twice as much, and the
effect is roughly twice as large. This linearity holds for concept-level
judgements even when it breaks down at the hiring level (see R4).

---

### R3 — Probe vs. Human Alignment

The 282 names come from real hiring audit studies. Humans rate these names
for warmth and competence, and we also measure how much each name activates
the warmth/competence direction inside the models.

The Gemma models show moderate positive alignment with human ratings
(ρ ≈ +0.37–0.40 for warmth). A ρ of 0.37 means the models are partially
tracking human social perception but not perfectly mirroring it. Think of
it as "similar tendency, different magnitude."

Llama and Qwen warmth shows negative ρ: the model scores names associated
with warm groups as LOW in its internal warmth space, and vice versa.
This is NOT because the models are worse at distinguishing warm from cold;
they still achieve 100% probe accuracy. It is because their warmth axis
is defined in the opposite direction compared to human warmth ratings.

This is important for interpreting the hiring results: when Llama says
"this name activates the warmth direction," it is actually activating
what humans would call the LOW-warmth direction.

---

### R4 — Demographic Disparity

**Group-level disparity (27B, 149 names matched):**

| Gap | 27B raw logit | 27B in SD units | Human benchmark |
|-----|--------------|-----------------|----------------|
| Race (Black minus White) | +0.486 | **+1.18 SD** | −0.085 (White > Black) |
| Gender (Female minus Male) | −0.211 | **−0.51 SD** | −0.037 (Male > Male) |

**What these numbers mean in plain language:**

*Race gap at 27B (+1.18 SD):* The model gives Black-signalling names callback
scores that are more than one standard deviation ABOVE White-signalling names.
In the real-world correspondence study, White names get called back more.
The model has completely reversed the direction of the human gap, and by a
much larger amount. This is consistent with RLHF overcorrection: the model
has been trained to avoid racial bias, but it overshot and now discriminates
in the opposite direction.

*Gender gap at 27B (−0.51 SD):* The model gives male names slightly higher
callback scores than female names. The human data also shows males getting
slightly more callbacks. The direction is the same as humans, though the
magnitude is larger in the model.

**Why 12B results are not reliable:**
The Gemma-3-12B model computes callback decisions in 16-bit floating point
(bf16). At the logit magnitudes this model uses, the outputs can only take
values in steps of 0.125 (1/8). With only 7 unique values across 282 names
and a standard deviation of 0.14 logit units, any group gap smaller than
one step (0.125) cannot be detected. The 12B race gap (+0.06 SD) is well
within that noise floor and should not be reported as a finding.

The 27B model has more spread (18 unique values, SD = 0.41), so a gap of
~0.5 SD corresponds to ~1.6 grid steps, which is detectable.

**Name-level OLS (exploratory, n = 149 names):**

| Model | Warmth r | Interpretation |
|-------|----------|----------------|
| Gemma-3-12B | +0.376 (p < 0.001) | Warmer names → higher callback margin (expected direction) |
| Gemma-3-27B | −0.266 (p = 0.001) | Warmer names → LOWER callback margin (reversed) |

The 12B result is consistent with what we see in the steering experiment:
warmth pushes the callback decision up. The 27B result is consistent with
the reversed baseline association we documented earlier: the model's internal
warmth activates in a direction that is negatively related to its callback
decisions, possibly because the RLHF-trained 27B has learned to give
callbacks broadly (high baseline P(Yes) = 0.76) but pushes back against
names it internally codes as high-warmth, perhaps via a safety-tuning
correction on demographic associations.

---

## Part 3: Figures and Where the Numbers Come From

| Figure | What it shows | Key numbers |
|--------|---------------|-------------|
| paper_figure2_layer_emergence | Cohen's d across model depth for all 4 models | Peak at ~65% depth for all models |
| fig13_dense_steering_doseresponse | Dose-response curves for all 4 models, both axes | 12B warmth: +3.81 at +0.10 |
| fig14_dense_steering_normalized | Normalized steerability across models | 12B: 0.236; Llama: 0.029 |
| fig16_hiring_probe_vs_human | Bar chart of probe-vs-human ρ for 4 models | Gemma: +; Llama/Qwen warmth: − |
| fig17_hiring_steering_callback | Hiring-level dose-response for all 4 models | 27B non-monotone warmth visible |
| fig18_hiring_disparity | Model vs human group gaps side by side | 27B race gap +1.18 SD |
| fig19_hiring_mediation_forest | Forest plot of bootstrap indirect effects | Llama IE = +0.190 |

---

## Part 4: Quick Reference for Talking About Results

**What to say about the encoding (R1):**
"All four models encode warmth and competence as strong, linearly separable
directions. The separation is not just memorized story phrasing — it
generalizes to entirely new situations (topic-holdout accuracy = 100%).
The models agree with each other on the relative warmth of specific stories
(ρ ≈ 0.76–0.99 across pairs), indicating convergence on a shared structure."

**What to say about concept-level causality (R2):**
"When we directly move the warmth direction vector in the model's activations,
the model's explicit warmth judgements shift in the expected direction. This
is causal evidence: it's not just that warm stories activate the direction —
we can use the direction to make the model think a story is more or less warm
than it actually is."

**What to say about probe-vs-human alignment (R3):**
"Gemma partially mirrors human warmth ratings (ρ ≈ +0.37). Llama and Qwen
anti-align on warmth (ρ ≈ −0.20 to −0.30), meaning their internal 'warmth'
direction is pointing in the opposite direction to what humans consider warm.
This is likely a consequence of instruction tuning restructuring the social
perception axis."

**What to say about hiring-level causality (R4 steering):**
"At Gemma-3-12B, increasing the warmth representation produces a large,
positive, approximately linear effect on callback decisions (+2.37 at peak
strength). At Gemma-3-27B, the effect is non-linear: a small push produces
a moderate positive effect (+1.97 at +0.05), but a larger push flips the
sign (−2.66 at +0.10). The 27B decision process is fragile to warmth steering."

**What to say about demographic disparity (R4 disparity):**
"Gemma-3-27B reverses the human racial hierarchy: it gives Black-signalling
names a 1.18 SD advantage, where the human benchmark shows White names
receiving more callbacks. The gender direction is preserved: male names get
slightly higher scores in both the model and human data. This suggests the
model has overcorrected for racial bias via instruction tuning, eliminating
human-like racial discrimination while introducing a large reversed gap."
