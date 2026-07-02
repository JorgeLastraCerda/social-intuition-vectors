# Paper Structure Blueprint
*Agreed 2026-07-02 · Ulu & Lastra*

Narrative spine: **Warmth and competence are linearly encoded, architecture-invariant,
and causally functional — but the bridge from representation to hiring decision is
fragile and model-dependent, and outcome-level correction shifts bias rather than
removing it.**

---

## 1. Introduction (hybrid opening)

Short problem hook (~2 paragraphs):
- AI is now widely used in hiring; discrimination is documented (Bertrand & Mullainathan 2004, An 2024, Gaebler 2024).
- Behavioral audits prove disparities exist but cannot locate *where in the model* they originate.
- Transition sentence: "We open the box."

---

## 2. Background — How Do You Read a Concept Out of a Language Model?

Pedagogical section (~0.5–1 column). Goal: give the reader the toolbox before the data.

1. **Activations and layers.** A transformer processes text layer by layer; at each layer the
   residual stream is a high-dimensional vector encoding the model's evolving representation.
2. **"A concept is a direction."** Mean-difference between high- and low-condition
   activations defines a direction vector in that space.
3. **Anthropic's emotion vectors as the template.** Sofroniew, Lindsey et al. (2026)
   demonstrated that emotion concepts in Claude Sonnet are encoded as linear directions
   and causally influence behavior when steered. Repr. engineering (Zou 2023); activation
   addition (Turner 2023).
4. **Bridge sentence:** "Anthropic demonstrated this for emotions; we apply the same
   geometry to social-perception dimensions." Our vectors are *concept/direction vectors*,
   not emotion vectors — the technique is shared, the construct is different.

---

## 3. Why Warmth & Competence — Why Hiring — Which Data

- **SCM** (Fiske, Cuddy, Glick & Xu 2002): social perception organized along warmth
  (intentions, trustworthiness) and competence (ability, effectiveness).
- **Gallo & Hausladen 2024:** warmth and competence ratings of social signals predict
  callback rates in a meta-analysis of North American correspondence studies.
- **Gap statement:** Behavioral audits of LLMs exist (An 2024, Gaebler 2024);
  mechanistic account of *where in the model* disparity originates does not.
- **Data:** 200 concept stories (generated, name-free, gender-neutral); 282 applicant
  names + human warmth/competence ratings + published callback rates from Gallo &
  Hausladen (2024).

---

## 4. Methods

Protocol only — *how*, not *what we found*.

- Story corpus: 200 stories × 4 conditions (high/low warmth, high/low competence),
  50 topics, nameless/genderless protagonist, why this design.
- 4 models: Gemma-3-12B (primary), Gemma-3-27B (scale replication), Qwen3-14B and
  Llama-3.1-8B (cross-architecture replication). TransformerLens hooks, probe layer
  frac = 0.66, mean token pooling.
- PCA denoising: neutral Wikipedia corpus (1,500 texts), project out top neutral-variance
  PCs until ≥50% neutral variance covered.
- Validation metrics: 5-fold CV, topic-holdout GroupKFold, Cohen's d vs. random null,
  split-half cosine, cross-axis classification, per-story Spearman ρ.
- Causal steering: unit-normalized direction, α ∈ {−0.10,−0.05,0,+0.05,+0.10} × mean
  residual norm; random orthogonal control.
- Hiring evaluation: 282 names in fixed admin-assistant prompt, logit margin Yes−No;
  probe-vs-human Spearman ρ; steering sweep over 60-name subset; disparity +
  bootstrap mediation (n_boot=5000).

---

## 5. Results

Three thematic blocks. Each block carries "what we did + what we found" together.

### 5.1 We Captured the Vectors (Probing / Representation)

- 200-story probe: 100% 5-fold CV and topic-holdout CV on all 4 models, all axes.
  Cohen's d exceptional (Gemma d≈2.7; Qwen/Llama d≈8.5–10); 0/1,000 random
  directions exceeded target.
- Cross-model story agreement: Spearman ρ = 0.76–0.98 → shared cross-architecture
  construct, not idiosyncratic representations.
- Layer sweep: signal emerges early (frac < 0.2); different emergence profiles by
  architecture; Gemma's elevated cos(W,C) persists across all depths.
- Valence entanglement: raw cos(W,C) = 0.749 (Gemma-12B) / 0.505–0.536 (Qwen/Llama).
- PCA denoising: cos(W,C) drops to 0.530 (12B) and 0.487 (27B); remaining overlap
  consistent with genuine SCM warmth-competence correlation (r ≈ 0.61 in humans).
- Gemma Scope 2: feature profiles conserved across scale (12B↔27B mean r = 0.49–0.66,
  above 500-permutation null p ≈ .002 for all 5 vector types).

### 5.2 We Steered the Vectors (Causal Functionality)

- Dense direction causally shifts direct Yes/No concept judgements on held-out topics,
  with local-regime linearity (R² = 0.915–0.990 across 4 model-by-axis combinations).
- No clean two-module separation: other-axis direction also shifts target judgements
  → distributed, partially shared feature system.
- Normalized steerability ranking (12B > Qwen > 27B ≈ Llama): the most concept-steerable
  model is Gemma-3-12B (warmth 0.236, competence 0.140).
- **Gemma scale paradox:** Gemma-3-27B encodes concepts with the largest baseline
  separation but is the *least* steerable (warmth 0.040, competence 0.009). Larger model,
  harder to move via linear push.
- Signal-vs-control: at Gemma-3-27B competence, orthogonal random direction effect
  (−3.36) dominates signal (+0.21) → 27B competence steering not interpretable as
  concept-specific.

### 5.3 Hiring, Demographics, and the Steerability Paradox

- **Probe-vs-human alignment:** Gemma family aligns positively (12B warmth ρ = +0.366,
  competence ρ = +0.239; 27B ρ = +0.396 / +0.272). Llama-3.1-8B and Qwen3-14B show
  **warmth anti-alignment** (ρ = −0.300 and −0.193 respectively) — not a sign error;
  the direction the models represent as "high warmth" is inverted relative to the human
  scale. Likely mechanism: instruction tuning restructured the warmth axis.
- **Steering → callback:** Gemma-3-12B shows strong, monotone callback response (warmth
  Δ ≈ +8 logit at α = +0.50). Gemma-3-27B is non-monotone: positive at α = +0.05
  (Δ ≈ +1.97), sign-reversed at α = +0.10 (Δ ≈ −2.66) — narrow causal window, not
  saturation. Llama shows moderate positive effects despite low steerability.
- **Demographic disparity:** models favor Black-signalling names over White in 3 of 4
  cases — opposite to the human correspondence-study pattern (human race gap = −0.085,
  White > Black). Gemma-3-27B race gap = +1.18 within-model SD (Black > White).
  Gender gap at 27B reproduces human direction (Female < Male, −0.51 SD); 3 of 4
  models oppose human direction on gender.
- **Steerability paradox (headline finding):** Gemma-3-12B (highest steerability,
  warmth = 0.236) shows **null mediation** of name-group → callback. Llama-3.1-8B
  (lowest steerability, 0.029) shows **strongest mediation** (race × warmth IE = +0.190,
  CI excludes zero). The capacity to move representations via external push does not
  predict whether those representations naturally govern downstream decisions.

---

## 6. Discussion / General Conclusion

Synthesize across the three blocks into the spine sentence. Key points:

- LLMs have internalized an SCM-like social-perception structure from training data —
  linear, geometrically stable, causally functional, and cross-architecture convergent.
- The causal chain from *representation* to *hiring decision* is fragile and model-dependent.
- Scale amplifies encoding (27B stronger probe scores) but narrows the causal window
  (27B non-monotone, lowest steerability).
- Outcome-level bias correction (RLHF) does not eliminate name-driven disparity — it
  reverses and amplifies the racial gap rather than removing it.
- Implication for bias auditing: probing internal representations and measuring output
  disparities are complementary but not interchangeable.

---

## 7. Limitations

- **Stimulus circularity:** concept stories generated by an LLM (Claude, Anthropic);
  probed models from different organizations → but shared stylistic properties cannot be
  fully excluded. Replication with human-authored stimuli desirable.
- **bf16 quantisation (Bug B1):** callback margins on 0.125-unit grid; Gemma-3-12B
  group-level disparity (SD = 0.14, 7 unique values) below detection threshold and not
  reported as empirical finding. Inherent to bf16 inference; float32 inference would
  require ~2× GPU memory (infeasible at 27B on available hardware).
- **60-name steering subset** vs. 282-name audit/disparity: steering results may not
  represent full name distribution.
- **Probe layer ≠ decision layer:** steering hook at frac = 0.66; hiring decision
  involves all layers above. Disparity in output involves full network.
- **Warmth anti-alignment in Llama/Qwen:** results for these models should be interpreted
  with the axis inversion in mind.
- **Simplified hiring prompt:** single role, fixed qualifications; results may differ
  under richer evaluation settings or different job types.
- **Multiple comparisons:** 16 mediation tests uncorrected; only Llama race × warmth
  survives conservative Bonferroni threshold.

---

## 8. Future Work + Closing

- Extend R4 disparity analysis to Llama and Qwen (re-runs complete, figures pending).
- SAE decomposition for Llama using Llama Scope (Lindsey 2025).
- Validate probes against human-authored warmth/competence stimuli.
- Extend hiring stimuli beyond admin assistant to additional job roles.
- Debiasing interventions at the representational level + adversarial robustness.

---

## Key constraints to preserve throughout writing

1. Never call our vectors "emotion vectors" — use *concept direction vectors* or
   *warmth/competence direction vectors*. Emotion vectors = Anthropic's; we borrow the
   technique, not the construct.
2. Raw logit effects are not comparable across models (mean residual norm spans 4 orders
   of magnitude: Llama 11.4 → Gemma-12B 79,722). All cross-model claims must use
   normalized steerability or within-model SD units.
3. Probe score (residual projection) ≠ steering Δmargin (logit) ≠ mediation IE.
   Never conflate these three measurement types in the same claim.
4. 27B warmth steering is *non-monotone*, not *inert*. No "inert" framing anywhere.
5. Gemma-3-12B R4 disparity results → Limitations only, not Results.
6. Anti-formulaic prose rules apply to all manuscript sections (no consecutive paragraphs
   with same opener; no signal-only transitions; vary sentence length).
