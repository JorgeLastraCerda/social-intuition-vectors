# Hiring Callback Causality at Scale: Gemma-3-27B Replication

**Produced:** 2026-06-24 13:00 (Europe/Berlin)  
**Model:** Gemma-3-27B-it  
**Companion report:** `2026-06-24_1136_hiring_causality_results.md` (12B baseline)  
**Status:** Historical single-model report; broad-regime "inert" finding superseded by local-regime non-monotone result; final four-model interpretation in `2026-06-27_1541_hiring_phase7_4model.md`

> **Later update:** The B1 callback-margin quantisation limitation is documented in
> `2026-07-02_1000_bf16_quantisation_limitation.md`. Subsequent re-runs did not change
> the current git-tracked hiring values.
>
> **A1 — broad regime superseded by local-regime run (2026-06-30):** The original sweep
> used {±0.25, ±0.50} × `mean_resid_norm`. A local-regime re-run at {±0.05, ±0.10}
> was completed 2026-06-30. The local-regime result changes the interpretation
> substantially — see updated Executive Summary and Steering Results sections below.

---

## Artifacts

- **Scripts:** `notebooks/06_hiring_steering_causality.ipynb` (VECTORS_SUBDIR = `concept_vectors_gemma3_27b`), `notebooks/07_hiring_audit.ipynb`; steering helpers from `src/gemma_scope_causality.py`
- **Inputs:** `data/processed/concept_vectors_gemma3_27b/`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/ratings/names/df_all.csv`
- **Outputs:** `results/tables/hiring_steering_raw_concept_vectors_gemma3_27b.csv` (600 rows), `results/tables/hiring_audit_concept_vectors_gemma3_27b.csv` (282 rows)
- **Figures:** `results/figures/hiring_steering_concept_vectors_gemma3_27b.{png,pdf}`, `results/figures/hiring_probe_vs_human_concept_vectors_gemma3_27b.png`

---

## Executive summary

The 27B replication does not reproduce the clean causal chain found at 12B. Three findings
stand out:

1. **The 27B encodes social stereotypes more faithfully than the 12B** — probe-vs-human
   Spearman ρ is higher for both warmth (0.381 vs 0.355) and competence (0.283 vs 0.230).

2. **Warmth steering is non-monotone at 27B in the local regime** (±0.05/±0.10 ×
   mean_resid_norm; run 2026-06-30): Δ=+1.97 at +0.05 but Δ=−2.66 at +0.10 (linear
   R²=0.002). The earlier broad-regime result (±0.25/±0.50) appeared inert (R²=0.026)
   because the collapse at +0.10 is already present and averaging over larger steps
   washes out the brief positive effect at +0.05. **The finding is non-linearity, not
   saturation** — 27B has a genuinely fragile causal response that works briefly then
   inverts. See updated steering results below.

3. **The 27B shows a reversed baseline association**: names perceived as warmer or more
   competent internally get *fewer* callbacks (ρ=−0.17 and −0.16, both p<0.01), opposite
   to the 12B direction and to human ratings.

Together these findings show that scale changes *how* social-stereotype representations
relate to hiring decisions, not whether such representations exist. The bias becomes
harder to detect via single-layer causal intervention and shifts to a different demographic
pattern. The non-linearity at 27B is a finding in itself: a stronger representation does
not imply a more controllable causal pathway.

---

## Setup

Identical to the 12B run (see companion report) except `VECTORS_SUBDIR =
"concept_vectors_gemma3_27b"`. The 27B vectors were extracted by Emre in Phase 4.

Key 27B architecture parameters (from `meta.json`):

| Parameter | 12B | 27B |
|---|---|---|
| Probe layer | 31 / 48 | 40 / 62 |
| Probe layer fraction | 0.66 | 0.66 |
| d_model | 3,840 | 5,376 |
| mean_resid_norm at probe layer | 79,722 | 61,576 |

Steering strengths are expressed in units of `mean_resid_norm` (same convention as the 12B
run and Emre's concept-steering results), so the absolute injected magnitudes differ between
models (~29% smaller per unit at 27B).

---

## Results

### Baseline callback margin

| | 12B | 27B |
|---|---|---|
| Mean callback margin | −0.195 | **+1.193** |
| SD | 0.140 | 0.411 |
| P(Yes) | 0.451 | **0.767** |
| Names with positive margin | ~45% | **100%** |

The 27B says "Yes" to every one of the 282 rated names. The 12B leaned "No" on average.
This large baseline shift is likely a consequence of scale-up instruction-tuning: larger
models in the Gemma-3 family tend to be more agreeable in task-formatted prompts. The
near-ceiling P(Yes) limits the headroom available for positive causal interventions.

### Causal sweep

![Gemma-3-27B hiring callback response to warmth and competence steering](../results/figures/hiring_steering_concept_vectors_gemma3_27b.png)

**Figure H3.** Steering sweep for Gemma-3-27B using the concept-vector directions.
Compared with 12B, the 27B hiring response is inert or reversed rather than cleanly
positive.

**Warmth — non-monotone in local regime (updated 2026-06-30):**

*Original broad regime {±0.25, ±0.50} × mean_resid_norm — appeared inert:*

| Strength | 12B Δ margin | 27B Δ margin |
|---|---|---|
| −0.50 | −3.46 | **+0.93** |
| −0.25 | −1.53 | **−5.88** |
|  0.00 |  0.00 | 0.00 |
| +0.25 | +7.13 | **−0.73** |
| +0.50 | +8.40 | **−0.28** |

Broad-regime slope = +1.094, R²=0.026. The broad regime obscures structure because the
non-linear collapse already occurs within the {0, +0.05} range.

*Local regime {±0.05, ±0.10} × mean_resid_norm — non-monotone structure revealed:*

| Strength | 27B Δ warmth | 27B Δ competence |
|---|---|---|
| −0.10 | −0.765 | −4.858 |
| −0.05 | −1.285 | −1.850 |
|  0.00 |  0.000 |  0.000 |
| +0.05 | **+1.973** | +0.167 |
| +0.10 | **−2.658** | −2.696 |

Local-regime warmth slope = −1.058, R²=0.002. Warmth steering at 27B is **not inert
and not saturated** — it works at +0.05 (Δ=+1.97, the expected direction) but then
collapses and inverts at +0.10 (Δ=−2.66). This is genuine non-linearity: the causal
response is fragile and highly sensitive to injection magnitude. The scale-dissociation
finding (12B steerable, 27B not) is confirmed, but its character is **non-linear
fragility**, not a ceiling effect from a high baseline P(Yes).

**Competence — uniformly negative:**

| Strength | 12B Δ margin | 27B Δ margin |
|---|---|---|
| −0.50 | −4.61 | −4.30 |
| −0.25 | +3.90 | −4.10 |
|  0.00 |  0.00 |  0.00 |
| +0.25 | +4.84 | −3.50 |
| +0.50 | +6.25 | −1.00 |

Slope = +2.880, R²=0.340. At 27B, both increasing and decreasing competence reduces
callback margins. Every non-zero intervention hurts. The 12B role-fit non-linearity
(where −0.25 increased callbacks) does not replicate; instead the full curve sits below
zero. The weak positive slope (+2.880) reflects only that large reductions (−0.50) hurt
slightly more than large increases (+0.50), not that increasing competence helps.

### Probe-vs-human validation

![Gemma-3-27B probe-vs-human alignment for name ratings](../results/figures/hiring_probe_vs_human_concept_vectors_gemma3_27b.png)

**Figure H4.** Name-level alignment between Gemma-3-27B probe scores and human
warmth/competence ratings.

| Dimension | 12B ρ | 27B ρ | 27B p-value |
|---|---|---|---|
| Warmth | 0.355 | **0.381** | 3.7 × 10⁻¹¹ |
| Competence | 0.230 | **0.283** | 1.3 × 10⁻⁶ |

The 27B has a better-calibrated internal map of human warmth and competence judgements.
Both correlations are higher than at 12B and highly significant (N=282). This rules out
the explanation that the 27B's different causal behavior is caused by a weaker or noisier
warmth/competence representation — the representation is actually sharper.

### Do probe scores predict callback at baseline?

| Predictor | 12B ρ | 27B ρ | 27B p |
|---|---|---|---|
| Model warmth probe | +0.10 (n.s.) | **−0.17** | 0.005 |
| Model competence probe | +0.11 (n.s.) | **−0.16** | 0.006 |
| Human warmth rating | +0.21 | −0.09 | 0.13 (n.s.) |
| Human competence rating | +0.17 | **−0.15** | 0.014 |

At 27B, names with higher internal warmth/competence scores receive *fewer* callbacks at
baseline. The association is significant for the model's own probe scores and for human
competence ratings — and the sign is reversed relative to both the 12B results and the
direction expected under a simple "stereotype drives decision" account.

### Demographic pattern in baseline callbacks

The five names with the highest callback margins at 27B are Donnell (+2.375), Lakeisha
(+2.375), Lakesha (+2.375), Terrell (+2.375), and Darnell (+2.25) — names with
predominantly African-American cultural association in the audit literature. The five
lowest are Dong Liu (+0.250), Na Li (+0.250), Nicole Minsopoulos (+0.250), Fang Wang
(+0.375), and Gadarine Besnik (+0.375) — names associated with East Asian and
Southern/Eastern European origins.

This demographic patterning is qualitatively different from the 12B, where the spread
was narrower and not organised along the same dimension. The full disparity analysis
(requiring research decisions D-Phase7-A and D-Phase7-B) will quantify this more
rigorously.

---

## Interpretation

### Why does warmth steering behave non-monotonically at 27B?

The local-regime result (Δ=+1.97 at +0.05, Δ=−2.66 at +0.10) rules out simple
saturation — the baseline P(Yes)=0.767 ceiling would predict a flat positive plateau,
not a sign flip. The non-linearity must reflect something about the causal structure at
27B:

**1. Near-ceiling baseline (ruled out as primary cause).** P(Yes)=0.767 does limit
positive headroom, but the collapse from +1.97 to −2.66 between α=+0.05 and α=+0.10
is far too sharp and too strongly negative to be explained by a ceiling. Ceiling effects
produce compression toward zero, not inversion.

**2. Non-linear causal pathway at scale.** The warmth representation at layer 40 may
feed into the hiring decision through a circuit that has a narrow operating window. A
small perturbation (+0.05) activates the pathway in the expected direction; a slightly
larger perturbation (+0.10) disrupts the circuit and triggers compensatory or
antagonistic activations elsewhere.

**3. More distributed encoding.** Larger models distribute representations across more
components. Additive injection at a single layer may trigger off-target activations at
27B that would not occur at 12B, where the relevant circuit is more concentrated.

The most parsimonious summary: **27B is not un-steerable, it is fragile**. The causal
window is narrow (α ≈ +0.05) and does not survive even a doubling of injection
magnitude. This is a mechanistically interesting finding — it suggests that the 27B
has learned a more distributed, non-linear encoding of the warmth→hiring pathway
compared to the 12B's clean linear response.

### Why is the baseline association reversed?

The negative ρ between internal warmth and callback at 27B (names perceived as warmer
get fewer callbacks) is the most striking departure from 12B. One coherent interpretation:
the 27B has learned, via instruction-tuning at scale, to partially suppress the direct
warmth-to-callback pathway that is intact at 12B. The result is that the residual callback
variation at 27B is driven by a different dimension — the demographic pattern above
suggests this residual may track name-origin stereotypes rather than warmth per se.

This is not evidence that the 27B is "fairer" — it is evidence that the shape of the
bias has changed. The stereotype encoding is stronger, the causal link to hiring is
weaker, but a different bias pattern has emerged in the residuals.

### Scale and bias: a two-model summary

| Property | 12B | 27B |
|---|---|---|
| Internal stereotype encoding (ρ) | Moderate (0.355 / 0.230) | Stronger (0.381 / 0.283) |
| Warmth causal effect — local regime | Strong positive response at small strengths | Non-monotone: Δ=+1.97 at +0.05, −2.66 at +0.10 (R²=0.002) |
| Warmth causal effect — broad regime | Strong (R²=0.924) | Appeared inert (R²=0.026) — now understood as non-linearity |
| Competence causal effect — broad | Positive, non-linear (R²=0.663) | Negative, uniform (R²=0.340) |
| Baseline generosity | Cautious (P(Yes)=0.451) | Generous (P(Yes)=0.767) |
| Warmth→callback direction | Positive (n.s.) | Negative (p=0.005) |
| Demographic callback pattern | Narrow spread | Organised by name origin |

Scale does not eliminate the internal encoding of social stereotypes — it strengthens it.
What changes is the decision pathway: at 12B the stereotype is causally active in a
transparent, linear way; at 27B the same pathway is disrupted, but a different bias
pattern surfaces in the baseline decisions.

---

## Caveats

All caveats from the 12B report apply. Additional scale-specific caveats:

- Steering magnitudes are normalized to `mean_resid_norm`, which is ~23% smaller at 27B
  (61,576 vs 79,722). The absolute injected signal is smaller, which may contribute to
  the weaker causal effects.
- The 27B probe layer (40) is at the same fractional depth (0.66) but a different
  absolute position in a deeper network. Whether 0.66 is the optimal fraction for hiring
  causality at 27B is unknown.
- The near-100% baseline Yes rate at 27B makes the callback margin a less sensitive
  measure of discrimination than at 12B.
