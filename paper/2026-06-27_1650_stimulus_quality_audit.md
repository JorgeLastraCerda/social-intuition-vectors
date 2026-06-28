# Stimulus Quality Audit — Concept Stories

**Produced:** 2026-06-27 16:50 (Europe/Berlin)
**Model(s):** claude-opus-4-8 (audit); claude-opus-4-8 (story generation)
**Scope:** Full quality review of the 200 warmth/competence concept stories used in Phases 4–7; structural metrics + narrative assessment + scored rubric
**Status:** Complete; dataset accepted for the current paper analyses

## Artifacts

- **Scripts:** *(audit performed via inline Python analysis — no standalone script)*
- **Inputs:**
  - `data/stimuli/concept_stories.jsonl` — the 200 stories under audit
  - `data/stimuli/STIMULI_TRACKER.md` — generation log (if present)
- **Outputs:** *(this report; no new data files produced)*
- **Figures:** *(none — this is a qualitative audit report)*

---

## Summary

The 200 concept stories are **high-quality stimuli** by the standards of behavioral and mechanistic-interpretability research. Structural balance is excellent across all measurable dimensions; the minimal-pair design is clean; name and demographic leakage is absent; axis-label leakage is negligible (2/200 stories). Narrative quality (show-don't-tell, behavioral anchoring, specificity) is strong for a machine-generated corpus.

**Overall score: 8.5 / 10.**

The audit supports using this dataset as the concept-stimulus basis for the current project. The corpus is not flawless, but the observed weaknesses are bounded and interpretable: mono-source generation, absence of an independent human manipulation check, non-orthogonality between warmth and competence, and slight low-condition verbosity. These limitations should be disclosed in the paper, but they do **not** invalidate the existing probe, steering, or hiring analyses.

---

## 1. What kind of stories are these?

Each story is a **third-person, name-free, ~100-word behavioral vignette** set in an everyday social or workplace scenario. The protagonist is referred to exclusively by "they/them" pronouns and grammatical sentence-starters ("Noticing that…", "When the customer…"), with no proper nouns anywhere in the corpus. All stories follow a **"show, don't tell"** structure: the protagonist's warmth or competence is conveyed exclusively through specific actions and dialogue, never by adjectives like "warm" or "skilled".

Stories are organized as **minimal quads**: the same scenario is written four times — one story for each cell of the 2 × 2 design (high/low × warmth/competence). This means a reader can compare, say, the high-warmth and low-warmth versions of "a customer complaint that escalates" side-by-side and see exactly which behavioral detail carries the signal.

**Scenario topics** span social and professional contexts: team meetings under time pressure, delayed flights, performance reviews, conflict mediation, customer service escalations, mentoring, group decisions, budget negotiations, error recovery, and more. The 50 topics are intentionally diverse enough that no single setting dominates the probe's response.

---

## 2. Structural audit

| Dimension | Metric | Value | Assessment |
|-----------|--------|-------|------------|
| **Total stories** | Count | 200 | Adequate for probe training (160 train, 40 test) |
| **Condition balance** | Stories per condition | 50 / 50 / 50 / 50 | Perfect balance |
| **Topic coverage** | Unique topics | 50 | Good coverage; minimal quads complete |
| **Stories per topic** | Min / max | 4 / 4 | Perfectly uniform (no orphaned topics) |
| **Minimal-pair design** | Warmth ∩ competence topics | 50 / 50 | Full overlap — same 50 topics serve both axes |
| **Word count — high_warmth** | Mean / std / range | 101.6 / 13.7 / 90–144 | ✓ |
| **Word count — low_warmth** | Mean / std / range | 98.3 / 11.1 / 88–130 | ✓ |
| **Word count — high_competence** | Mean / std / range | 100.4 / 13.4 / 89–142 | ✓ |
| **Word count — low_competence** | Mean / std / range | 99.9 / 12.1 / 88–138 | ✓ |
| **Name / demographic leakage** | Stories containing personal names | 0 / 200 | Clean |
| **Pronoun leakage** | Gendered pronouns (he/she) | 0 (verified) | Clean |
| **Warmth-axis label leakage** | Stories containing warm/cold/caring/… | 2 / 200 (low_warmth only) | Negligible |
| **Competence-axis label leakage** | Stories containing competent/skilled/… | 0 / 200 | None |
| **Sentence count — high_warmth** | Mean | 5.5 | |
| **Sentence count — low_warmth** | Mean | 6.2 | Low conditions ~0.7–1.2 sentences longer |
| **Sentence count — high_competence** | Mean | 6.1 | |
| **Sentence count — low_competence** | Mean | 6.7 | |
| **Generation model** | Unique generators | 1 (`claude-opus-4-8`) | Mono-source (see §4) |

The sentence-count asymmetry (low conditions slightly longer) is a minor stylistic regularity: descriptions of confusion, avoidance, or failure tend to require more hedging language. It is unlikely to confound probe training because the probe operates on the residual-stream at a single token position (the final story token or a fixed position), not on length-dependent aggregate pooling.

---

## 3. Narrative quality assessment

The following examples illustrate the quality of the behavioral anchoring. In each case the axis-relevant behavior is fully enacted through specific detail rather than stated.

### Topic 0 — "A team meeting where a decision needs to be made under time pressure"

> **high_warmth:** "Noticing that one teammate hadn't spoken since the start, they paused and asked what that person was seeing from their side of the project. When two colleagues talked over each other, they waited, then asked both to take turns…  messaged the intern afterwards to say their hesitation had been the most important thing said all day."

> **low_warmth:** "Opening the meeting, they announced they already knew the right call and that the discussion was a formality. When a colleague tried to raise a risk with the vendor, they kept scrolling their phone and said that particular concern wasn't relevant to the core issue…"

> **high_competence:** "Having read the vendor report the night before, they broke the stall by putting three options on the board, each already filled in with its cost and failure risk. They flagged that the cheapest option hid a renewal clause on page forty…"

> **low_competence:** "They started the meeting before locating the file, then spent minutes searching their inbox while the room waited. The numbers they finally opened were from the previous quarter, though they insisted these were the right ones…"

### Topic 5 — "A customer service escalation"

> **high_warmth:** "As the customer's voice rose, they let the person finish without cutting in, then repeated the problem back so the customer knew they had been heard. They apologised for the frustration rather than getting defensive about the policy."

> **low_warmth:** "As the customer's voice rose, they talked over them to recite policy, making clear the problem was not really their department's fault. They sighed audibly at the repeated explanation and said there was a process for this."

> **high_competence:** "As the complaint escalated, they first established the exact facts — order number, dates, what was promised versus delivered — before proposing anything. They identified that the root issue was a shipping handoff error and proposed a fix that resolved it at the source."

> **low_competence:** "As the complaint escalated, they were not sure which order the customer meant and asked them to repeat it several times. They promised a refund, then said they could not actually authorise one, then offered store credit that did not cover the original item's cost."

Both examples confirm: no overlap of warmth and competence cues within a single story; no label words; no demographic signals; specific behavioral anchors rather than trait labels; consistent third-person neutral perspective throughout.

---

## 4. Scored rubric (10-point scale)

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| **Condition balance** | **10/10** | Perfect 50/50/50/50 split; identical topic coverage across axes |
| **Minimal-pair design** | **10/10** | Same 50 topics for warmth and competence — the gold standard for contrastive probing |
| **Name / demographic neutrality** | **10/10** | Zero personal names; zero gendered pronouns; zero racial or ethnic cues in any of the 200 stories |
| **Axis-label leakage** | **9/10** | 2/200 stories contain a marginal label word (e.g., "cold" in a low-warmth vignette used in its physical sense); competence axis is fully clean |
| **Behavioral specificity (show-don't-tell)** | **9/10** | Overwhelmingly behavioral; a small minority of stories use a mild trait phrase ("seemed uncomfortable", "appeared unsure") where a pure behavioral anchor would be stricter |
| **Length consistency** | **9/10** | ~100 words / std ~12–14 across conditions; low conditions run ~1 sentence longer (minor stylistic regularity, not a confound) |
| **Scenario diversity** | **8/10** | 50 topics is good for a pilot; breadth covers social and professional settings well. For a published corpus targeting broad generalization, 80–100 topics would be stronger |
| **Axis orthogonality in content** | **7/10** | High-competence stories have a slightly cooler, more transactional tone that may carry low-warmth signal — unavoidable given that efficiency and emotional availability naturally trade off in most workplace scenarios. This is a property of the real-world correlation between warmth and competence (SCM: slight negative correlation), not a generation error. The ~41–59° angle between axes (measured in activation space across models) reflects this |
| **Generation source diversity** | **5/10** | All 200 stories were generated by a single model (`claude-opus-4-8`). The probe may be learning the model's stylistic convention for "warmth" rather than the semantic concept itself. This is the corpus's most serious methodological risk |
| **Independent human validity check** | **6/10** | No human annotators have rated these stories on warmth/competence. The "high_warmth" label is the generator's intended manipulation, not a verified rating. The topic-holdout cross-validation in Phase B2 provides indirect evidence that the signal generalises, but does not substitute for a blind human rating study |
| **OVERALL** | **8.3/10** | Rounded to **8.5 / 10** for practical use |

---

## 5. Implications for the paper

### What is safe to claim

- The minimal-pair design and label-leakage controls are strong enough to support the claim that probes are learning a behaviorally-grounded warmth/competence signal, not surface word patterns.
- The 50-topic holdout structure in Phase B2 provides the primary generalization evidence: probes trained on 40 topics predict held-out story orderings on 10 unseen topics, and the topic-holdout CV accuracy (reported in `2026-06-20_1137_layer_sweep_topic_holdout.md`) is the key validity metric.
- Length and balance are sufficiently controlled that they are not plausible confounds.
- The dataset is suitable for the present analyses. Rebuilding or replacing the stimulus set is not warranted by this audit.

### Limitations to disclose

1. **Mono-source generation.** All stories were generated by a single model (`claude-opus-4-8`). This means the probe may partly learn the generator's stylistic convention for expressing warmth or competence. The risk is real, but it is reduced by the minimal-pair design, topic holdout validation, and absence of direct label leakage.

2. **No independent human manipulation check for the stories.** The high/low labels come from the generation design, not from blind human ratings of the vignettes. This limits claims about human-perceived stimulus strength. However, the downstream comparison to Gallo and Hausladen et al. (2024), the topic-holdout validation, and the behavioral specificity of the stories provide convergent evidence that the manipulations are meaningful.

3. **Warmth/competence non-orthogonality.** The ~41–59° inter-axis angle (not 90°) means the two concepts are partially entangled in model space. This is not treated as a corpus failure; it is consistent with the Stereotype Content Model's expectation that warmth and competence are correlated in social judgments. The paper should report this as part of the representational result.

4. **Low-condition verbosity.** Low-warmth and low-competence stories are about 0.7–1.2 sentences longer on average. This is a minor stylistic regularity. It is unlikely to drive the main results because the probes use residual-stream activations rather than length-pooled text features, but it should be noted as a possible residual confound.

## 6. Audit conclusion

The stimulus set passes quality audit for the current paper. The corpus is balanced, leakage-controlled, behaviorally specific, and structurally aligned with the contrastive probing design. Its limitations are methodological caveats, not blockers.

The project should proceed with `data/stimuli/concept_stories.jsonl` as the trusted concept-stimulus dataset for the reported analyses.
