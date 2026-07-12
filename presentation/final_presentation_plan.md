# Final Presentation Plan — 9 Minutes

**Course:** Master's seminar (supervisor presentation + grading)
**Audience:** Professor + classmates. ML-literate but not mechanistic interpretability specialists.
**Goal:** Convince the audience that the paper makes a real contribution — not just "we found bias" but "we found *where* it lives and *what form* it takes."
**Constraint:** 9 minutes, no separate discussion. Every slide must earn its place.

---

## The One Sentence That Drives Everything

> "We looked inside four language models to find the social perception structure that might explain hiring discrimination, found it cleanly encoded, and discovered that safety training has not removed the bias — it has reorganized it in a surprising and policy-relevant direction."

Every slide should push this sentence forward.

---

## What to Include vs. Cut

### Must include

| Element | Why it stays |
|---------|--------------|
| The behavioral audit gap ("black box") | Motivates the whole paper in 15 seconds |
| Concept direction schematic | Explains the method without equations |
| Layer emergence / Cohen's d | Shows the representations are real and strong |
| Diverging steering figure | Shows causality: we can flip a hiring decision |
| Disparity comparison (model vs. human) | The headline result — RLHF overcorrection |
| One-sentence takeaway on implications | Gives the audience something to carry out |

### Cut entirely

| Element | Why it goes |
|---------|-------------|
| PCA denoising | One sentence maximum ("we removed a general positivity component"). Details add no value here. |
| SAE / Gemma Scope decomposition | Expert-only depth check. Mention in paper, not presentation. |
| Exact slope values, R², regression tables | "Approximately linear" is sufficient. Show the figure. |
| Bootstrap mediation methodology | Say "we tested whether internal representations carry the group signal into the decision." That is enough. |
| bf16 quantization details | One caveat sentence: "12B results are measurement-limited and excluded from disparity analysis." |
| Model parameter table | Irrelevant for the audience. |
| Layer numbers, d\_model dimensions | Same. |

### Include briefly (one sentence or one visual element)

| Element | How to include it |
|---------|-------------------|
| Steerability paradox | One sentence after the mediation result: "Interestingly, the model that is easiest to steer at the concept level shows no significant mediation path to the decision." |
| Llama/Qwen warmth anti-alignment | Mention as part of the disparity discussion: "Two of the four models have a warmth axis that points in the opposite direction to human ratings, likely due to how safety training restructured it." |
| Mediation forest plot | Show as a supporting visual for the steerability paradox. No need to explain bootstrap methodology. |

---

## Slide-by-Slide Draft

### Slide 1 — Title

**Title:** Probing Warmth and Competence Representations in LLM Hiring Decisions
**Authors:** Emrecan Ulu · Jorge Lastra Cerda · University of Konstanz
**Visual:** Clean, minimal. Maybe a simple icon of a resume and a robot, or just text.
**You say:** Nothing. Let it sit for 5 seconds while the audience settles.

---

### Slide 2 — The Problem (45 seconds)

**Header:** AI is now making hiring decisions — and it discriminates

**Three bullets (short):**
- Half of US organizations use AI in recruiting (SHRM 2025)
- Behavioral audits show name-driven disparities across race and gender
- But they treat the model as a black box: *that* it discriminates, not *how*

**Visual:** A simple diagram — Name → [Black box LLM] → Callback/No Callback — with a question mark over the box.

**You say:** "We know LLMs discriminate in hiring. But every existing study stops at the output. We wanted to go inside the model and find the mechanism."

---

### Slide 3 — The Idea (1 minute)

**Header:** What if social perception is encoded as geometry inside the model?

**One paragraph of text (large font):**
"Language models trained on human text might have internalized the same social perception structure humans use — including warmth and competence."

**Visual:** The concept geometry schematic (`background_concept_geometry.pdf`).
Point to: the two clusters (high/low warmth stories), the blue arrow (the direction vector v), the red dashed arrow (a steering intervention moving a point across the boundary).

**You say:** "We built 200 short stories with a nameless protagonist — 50 depicting high warmth, 50 low warmth, and the same for competence. We ran them through the model, took the internal activations at one layer, and computed the mean difference between conditions. That gives us a direction vector for warmth. If the model has really learned this concept, that direction should generalize."

---

### Slide 4 — Finding 1: The Representations Are Real (1 minute)

**Header:** Warmth and competence are linearly encoded — in all four models

**Two key facts (large font):**
- 100% classification accuracy, including on entirely new situations (topic-holdout)
- All four models — Gemma-3-12B, Gemma-3-27B, Llama-3.1-8B, Qwen3-14B — converge on the same warmth/competence ordering of stories (ρ ≈ 0.76–0.99 across pairs)

**Visual:** Layer emergence figure (`paper_figure2_layer_emergence.pdf`).
Point to: the Cohen's d curves rising as you go deeper into the model, the vertical marker at ~65% depth showing where you chose to extract.

**You say:** "The probe generalizes to new topics it has never seen during training. And when we compare four models from three different companies, they all agree on which stories are warm or cold — even though they were trained independently. That convergence is strong evidence that this reflects something in the training data, not an artifact of one architecture."

---

### Slide 5 — Finding 2: The Representations Are Causal (1.5 minutes)

**Header:** We can steer these directions and flip hiring decisions

**One key fact:**
"Pushing the warmth direction during resume evaluation shifts the callback recommendation — linearly and reliably at Gemma-3-12B."

**Visual:** Diverging steering figure (`paper_figure3_diverging_steering.pdf`).
Point to: the baseline dot (no steering), the line extending left and right (the steerable range), the decision boundary at x=0. Every row crosses the boundary — steering is sufficient to flip the decision for every name.

**You say:** "This is the causal test. We inject the warmth direction directly into the model's activations while it reads a resume, and the hiring recommendation moves. Every single name in the 60-name test set crosses the decision boundary. At Gemma-3-27B the effect is more complex — a small push works but a larger push reverses direction — which tells us the causal pathway is more fragile at larger scale."

**Caveat to mention:** "12B results are measurement-limited and excluded from disparity analysis."

---

### Slide 6 — Finding 3: Model Bias ≠ Human Bias (2 minutes)

**Header:** Safety training has reorganized bias, not removed it

**This is the headline slide. Build it in two steps if you can.**

**Step A — reveal the human benchmark first:**
"Here is what human hiring looks like: White names receive more callbacks than Black names. Males receive slightly more than females."

**Step B — reveal the model results:**
"Here is what the models do."

**Visual:** Disparity comparison figure (`fig18_hiring_disparity.pdf`).
Point to explicitly:
1. The human reference bars (race gap negative = White favored)
2. The 27B bar (race gap positive = Black favored, by +1.18 SD — pointing the opposite direction)
3. The gender bars (27B matches human direction; Llama and Qwen reverse it)

**You say:** "None of the three models with reliable data reproduce the human racial discrimination pattern. All three favor Black-signalling names — the opposite of the historical record. At 27B the gap is enormous: more than one standard deviation in the model's own units. Our interpretation is RLHF overcorrection: safety fine-tuning has corrected racial associations so aggressively that the calibration overshoots the real-world distribution by a large margin. Meanwhile, gender is less consistent across models — 27B matches the human direction, Llama and Qwen reverse it. The model is not a faithful copy of human prejudice. It is a differently biased system."

---

### Slide 7 — Mediation and the Paradox (optional, 30 seconds if time allows)

**Header:** The internal representation mediates the decision — but not where you would expect

**One key fact:**
"Bootstrap mediation shows that the warmth representation carries the racial signal into the hiring decision in Llama — but not in Gemma-3-12B, which is the most concept-steerable model."

**Visual:** Mediation forest plot (`fig19_hiring_mediation_forest.pdf`).
Point to: the Llama race × warmth bar (largest, CI excludes zero), the Gemma bars (all crossing zero).

**You say:** "The model that is easiest to steer via direct intervention is not the model where the representation naturally governs the decision. This suggests that concept-level steerability and causal mediation measure different things."

---

### Slide 8 — Takeaway (30–45 seconds)

**Header:** What this means

**Three bullets:**

- LLMs have internalized the warmth/competence structure of human social perception — it is geometrically real and causally functional
- The bias that emerges in hiring is not inherited discrimination — it is a reorganized, post-RLHF version that can be *larger* and in the *opposite direction* from the human signal
- Behavioral audits alone are not enough: probing internal representations and measuring output disparities capture different things, and you need both

**Visual:** Optional — the paper title again, or a clean summary diagram contrasting "what behavioral audits see" vs "what we see."

**You say:** "The contribution is methodological as well as empirical. We connected mechanistic interpretability to the correspondence study tradition, and the combination reveals something that neither approach would find alone."

---

## Presentation Logistics

**Animation priority (if using PowerPoint/Keynote):**
1. Slide 6 (disparity) — most important to animate. Reveal human benchmark first, then model results one by one.
2. Slide 5 (steering) — consider revealing one name row at a time, then the full figure.
3. All other slides — fine as static.

**Laser pointer / pointing discipline:**
Before every figure, say out loud what the axes are. Then point to the specific bar, line, or element before discussing it. Assume nobody in the room knows where to look.

**Practice target:**
First full run-through will probably be 12–13 minutes. After one rehearsal you should be at 10. After two you should be at 9 or under.

**Figures to include (in order of appearance):**
1. Background concept geometry schematic
2. Layer emergence (paper\_figure2)
3. Diverging steering (paper\_figure3)
4. Disparity comparison (fig18)
5. (Optional) Mediation forest plot (fig19)

---

## Key Sentences to Have Ready

These are things the professor may ask about or that you may need to say under pressure:

- **On why concept stories instead of hiring prompts:** "Using hiring prompts to probe warmth would confound the social perception signal with the hiring decision itself. We needed a clean, identity-neutral stimulus set."
- **On the RLHF interpretation:** "We cannot confirm this is RLHF without ablating the fine-tuning stage, but the reversal direction and the scale are consistent with overcorrection on racial associations during safety training."
- **On the steerability paradox:** "Steering measures causal sensitivity to an external intervention. Mediation measures whether the natural activation propagates to the outcome. The two dissociate — and that is itself an interesting finding about how these models work."
- **On why four models:** "Cross-architecture replication at three independent organizations gives us confidence that the finding is not specific to one training recipe or one architectural choice."
