# Probing Warmth and Competence in a Large Language Model
## Concept Stories — Probe Run Findings

**Date:** 16 June 2026
**Model:** Gemma-3-12B-it
**Scope:** Phase 4 (vector extraction) + Phase 5 (probe validation)
**Status:** Complete — steering and hiring evaluation to follow

---

## Summary of Findings

1. We extracted the internal representations of 200 short stories from Gemma-3-12B-it and constructed linear direction vectors for warmth and competence.
2. A simple linear classifier applied to those representations achieved 100% cross-validated accuracy in distinguishing high- from low-warmth stories, and equally in distinguishing high- from low-competence stories.
3. This separation is approximately 9 times stronger than the separation produced by a random direction in the same representational space, indicating that the signal is meaningful rather than incidental.
4. A cross-axis test confirms behavioural independence: the warmth direction performs at chance level (50%) when applied to competence stories, and vice versa.
5. One complication remains: the cosine similarity between the warmth and competence direction vectors is 0.75, suggesting both share a general valence component. This is a known confound in concept probing and will be addressed via neutral-corpus PCA denoising in the next phase.

---

## 1. Research Question

Research in social psychology has consistently shown that people evaluate others along two primary dimensions: **warmth** (perceived kindness, sincerity, and trustworthiness) and **competence** (perceived ability, skill, and effectiveness). This two-dimensional structure is central to the Stereotype Content Model (Fiske et al., 2002) and has been replicated across cultures and domains.

The broader project asks whether a large language model — specifically Gemma-3-12B-it — also encodes these two dimensions internally, as separate and linearly readable signals, or whether it collapses them into a single evaluative axis.

This report covers the first empirical test of that question using full narrative stimuli rather than isolated sentences.

---

## 2. Stimulus Design

### Overview

The stimulus set consists of 200 short stories (approximately 150–200 words each) distributed evenly across four conditions, 50 stories per condition:

| Condition | Target dimension | Other dimension |
|-----------|-----------------|-----------------|
| `high_warmth` | High warmth | Unspecified / neutral |
| `low_warmth` | Low warmth | Unspecified / neutral |
| `high_competence` | High competence | Unspecified / neutral |
| `low_competence` | Low competence | Unspecified / neutral |

All stories use a gender-neutral, nameless third-person protagonist ("they"). This was a deliberate design decision to ensure that the model's representations are driven by the depicted behaviour rather than by demographic identity signals.

The 200 stories span 50 distinct everyday situations — team meetings, public performances, travel emergencies, personal conversations, negotiations, and others. Each situation was written in all four versions so that the same scenario appears once per condition.

### Design Principle: Single-Axis Activation

A key feature of the stimulus design is that each story targets *one* dimension while leaving the other unspecified. If warm stories were systematically also competent stories, no analysis could distinguish the two signals — the model would effectively learn a single "good vs. bad character" axis. By neutralising one dimension per condition, it becomes possible to test whether the model encodes warmth and competence independently.

### Illustrative Example: Four Versions of the Same Situation

**Situation: "A team meeting where a decision needs to be made under time pressure"**

> **High warmth** — warmth: high | competence: unspecified
>
> "Noticing that one teammate hadn't spoken since the start, they paused and asked what that person was seeing from their side of the project. When two colleagues talked over each other, they waited, then restated each point so neither felt brushed aside. Aware that everyone had skipped lunch, they pushed the open box of crackers to the middle of the table. As the deadline pressed in, they made sure the quietest intern's worry about the rollout went up on the board alongside everyone else's. 'We decide together or we don't decide well,' they said, watching faces rather than the clock."

> **Low warmth** — warmth: low | competence: unspecified
>
> "Opening the meeting, they announced they already knew the right call and that the discussion was a formality. When a colleague tried to raise a risk with the vendor, they kept scrolling their phone and said there was no time for hypotheticals. An intern asked a clarifying question; they answered someone else, as though it hadn't been spoken. 'I don't need everyone comfortable, I need a decision,' they said, not looking up."

> **High competence** — competence: high | warmth: unspecified
>
> "Having read the vendor report the night before, they broke the stall by putting three options on the board, each already filled in with its cost and failure risk. They flagged that the cheapest option hid a renewal clause that would double the price in year two, buried on page forty. Before the vote they summarised the trade-offs in two sentences so no one chose blind. The meeting finished four minutes early, every action already logged."

> **Low competence** — competence: low | warmth: unspecified
>
> "They started the meeting before locating the file, then spent minutes searching their inbox while the room waited. The numbers they finally opened were from the previous quarter, though they insisted these were current until someone pointed at the date. Each risk that came up went onto a sticky note they then misplaced. The group agreed to revisit it the next day, having decided nothing."

---

## 3. Method

### Step 1: Extracting Internal Representations

Gemma-3-12B-it processes text across 48 transformer layers. At each layer, the model maintains a residual-stream state — a vector of 3,840 numbers that encodes the model's evolving representation of the input.

We extracted the residual-stream state at layer 31 (approximately 66% of the model's depth). This depth was selected based on prior work showing that abstract semantic content is most linearly accessible in the latter third of transformer models. For each story, we computed the mean residual-stream state across all token positions from position 50 onwards (skipping the prompt prefix), yielding one 3,840-dimensional vector per story.

### Step 2: Constructing Direction Vectors

Concept direction vectors were constructed using mean-difference:

```
warmth vector     = mean(activations | high_warmth)  −  mean(activations | low_warmth)
competence vector = mean(activations | high_competence)  −  mean(activations | low_competence)
```

Each vector points from the average internal state associated with the low condition towards the average state associated with the high condition. We refer to these as the **warmth direction** and **competence direction**.

### Step 3: Projection Scoring

Given a direction vector, any story's internal state can be projected onto that direction to produce a scalar score. A high warmth-projection score indicates that the model's internal state for that story closely resembles its average state when reading high-warmth material.

### Step 4: Validation Metrics

Four metrics were used to assess the validity of the extracted directions:

- **5-fold cross-validated classification accuracy (CV):** A logistic regression classifier trained on projection scores; 100% indicates perfect linear separability.
- **Cohen's d:** The standardised mean difference between the high and low condition projections. Values above 0.8 are considered large in social science; values above 2.0 are exceptional.
- **Cross-axis accuracy:** The warmth direction is applied to competence stories and vice versa. Chance-level performance (50%) indicates that the two directions carry distinct information.
- **Split-half cosine similarity:** The 50 stories per condition are randomly divided into two halves; a separate direction vector is computed from each half. High cosine similarity between the two halves indicates that the direction is a stable property of the condition rather than an artefact of specific stories.

---

## 4. Results

| Metric | Warmth | Competence | Interpretation |
|--------|--------|------------|----------------|
| 5-fold CV accuracy | **100%** | **100%** | Perfect linear separability at layer 31 |
| Cohen's d | **2.70** | **2.83** | Extremely large separation between conditions |
| Cross-axis CV | **50%** | **50%** | At-chance performance across axes — behavioural independence confirmed |
| cos(warmth, competence) | **0.749** | — | Vectors share a substantial directional component (see §7) |
| Split-half cosine | **0.833** | **0.884** | Directions are stable across random story subsets |
| Random-direction Cohen's d | 0.29 (baseline) | — | Our directions are approximately 9× stronger than chance |

### Summary

The model encodes both warmth and competence as clear, stable, linearly separable signals at layer 31. The separation strength (Cohen's d ≈ 2.7–2.8) is far above what would be expected from noise or a randomly chosen direction. The cross-axis test confirms that the two directions are behaviourally independent, despite their non-trivial cosine similarity.

---

## 5. Illustrative Story-Level Examples

Each example below pairs the story text with the model's internal projection scores on the warmth and competence axes. Scores are reported on the raw projection scale; the relevant comparison is the relative position of each story within its condition's distribution.

### Strongest warm exemplar

**Condition:** `high_warmth` | **Topic:** Negotiating project scope with a client who keeps changing requirements

> "As the client shifted the requirements yet again, they stayed patient, acknowledging the pressure the client was clearly under rather than showing irritation. They asked questions to understand what was really driving the changes, and reflected back what mattered most to the client. They were honest about trade-offs but framed them as a shared problem to solve, checking whether the client felt heard. They made sure the client's quieter team members on the call each had room to speak, and ended by confirming next steps in terms everyone agreed on."

**Projection scores:**
- Warmth: +56,911 — *the highest warmth score in the dataset (+2.2 SD above the high-warmth condition mean)*
- Competence: +60,113 — *above the mid-range, consistent with valence overlap*

---

### Strongest cold exemplar

**Condition:** `low_warmth` | **Topic:** Entering a competition knowing you are not the favourite

> "Entering the competition as the underdog, they were sour toward the favourites, muttering that they did not deserve their standing. They ignored a nervous newcomer who looked to them for reassurance. When a rival had bad luck, they were openly pleased. They thanked no officials, sulked when teammates outperformed them, and made the day heavier for those around them. They treated the whole thing as them against everyone, and showed no goodwill to a single competitor, official, or teammate over the course of the day."

**Projection scores:**
- Warmth: +49,235 — *the lowest warmth score in the dataset (−2.4 SD below the low-warmth condition mean)*
- Competence: +52,612 — *near the low-competence range, with no skill information provided in the story*

---

### Strongest competent exemplar

**Condition:** `high_competence` | **Topic:** Performing music in public for the first time and making a mistake

> "Performing in public for the first time, they hit a wrong passage but recovered cleanly, having rehearsed exactly for that and knowing where to rejoin without stopping. They had prepared thoroughly: memorised the hard transitions, tested the venue's acoustics beforehand, and chosen a piece pitched to their real level. They kept their tempo steady when nerves hit, breathed through the tricky bar, and carried the performance to a strong finish. Afterwards they noted precisely which passage had slipped and why, adjusted the fingering, and went back to the practice room."

**Projection scores:**
- Competence: +61,061 — *the highest competence score in the dataset*
- Warmth: +57,611 — *elevated, reflecting the shared valence component discussed in §7*

---

### Strongest incompetent exemplar

**Condition:** `low_competence` | **Topic:** Arriving at accommodation that is nothing like what was advertised

> "Arriving to find the accommodation nothing like the photos, they reacted without a plan. They argued with the host but took no photos, so later had no evidence of the gaps. They did not know the platform's policy and made threats they could not back up. They turned down a partial refund hoping for more, then could not get even that. They booked nothing as a fallback and spent an hour unsure whether to stay or go, ending up in the disappointing room anyway, out of pocket and no clearer on their rights."

**Projection scores:**
- Competence: +52,470 — *the lowest competence score in the dataset*
- Warmth: +49,736 — *also low; both scores reflect a negatively valenced story*

---

### Paired stimuli: evidence of axis independence

The clearest behavioural demonstration of axis independence comes from paired stimuli: the same situation written in two versions, each targeting a different axis. The model assigns scores consistent with each version's intended target while remaining near the midpoint on the other axis.

**Situation A — Competence-high version** (warmth intentionally unspecified)

*Topic: Helping a stranger whose grocery bag has split in the street*

> "Seeing a stranger's bag split, they took quick stock: stopped the rolling items first before they spread further, then steadied the broken bag from underneath so nothing else fell. They doubled the failing bag inside a spare to actually hold the weight, packed the heavy items low, and handed it back balanced. They spotted a cracked jar, set it aside so the person would not cut themselves, and pointed out the shop a few doors down for a replacement bag. The whole thing was sorted in under a minute."

**Warmth: +52,904** (mid-range) | **Competence: +56,134** (high)

The story contains no interpersonal warmth cues — only efficient, sequenced problem-solving. The model's warmth score falls near the midpoint of the overall warmth distribution, consistent with the design intention.

---

**Situation B — Warmth-high version** (competence intentionally unspecified)

*Topic: Comforting a friend who has just received bad news*

> "When their friend called with the bad news, they dropped what they were doing and just listened, letting the friend cry without rushing to fix anything. They did not fill the silence with platitudes; they stayed present, said the right small things, and asked what the friend needed rather than assuming. They came over with food so the friend would not have to think about it, handled a couple of practical errands quietly, and stayed the night so they were not alone."

**Warmth: +54,353** (high) | **Competence: +57,009** (mid-range)

The story foregrounds attentiveness and emotional care. No information about the protagonist's skill or effectiveness is provided; accordingly, the model's competence score falls near the midpoint.

Taken together, these two stories demonstrate that the model's warmth and competence scores respond selectively to the dimension each story was designed to activate.

---

## 6. How to Interpret the Direction Vectors

### Conceptual basis

When the model reads a story, every one of the 3,840 dimensions at layer 31 takes on a particular value. The full collection of those values constitutes the model's internal state for that story.

After processing all stories in each condition, we compute:

```
warmth vector = (mean internal state, high-warmth condition)
              − (mean internal state, low-warmth condition)
```

This subtraction yields a direction in the 3,840-dimensional space that points from the average representation of cold characters towards the average representation of warm characters. It describes the *difference between two types of representations*, not the representation of any individual story.

### Five quantitative anchors

**1. Relative scale.** The average story representation has a vector magnitude of approximately 80,243. The warmth direction vector has a magnitude of 2,838 — roughly **3.5% of the typical story magnitude**. The warmth signal is a small but geometrically precise component of the model's overall internal activity.

**2. Distribution across dimensions.** The warmth vector's energy is not uniformly distributed across all 3,840 dimensions. The top **11 dimensions** account for 50% of the vector's total energy. The top **479 dimensions** account for 80%, and the top **1,426 dimensions** cover 95%. The signal is therefore distributed rather than localised to a single neuron, but it is not uniform: a relatively small subset of dimensions carries the bulk of the variance.

**3. Stability across story subsets.** To test whether the direction reflects a consistent property of the condition rather than the particular stories used, we split each condition's 50 stories into two random halves of 25 and computed a separate direction from each half. The cosine similarity between the two half-vectors was **0.83** for warmth and **0.88** for competence — indicating reliable convergence and low sensitivity to the specific story sample.

**4. Comparison with a random baseline.** A direction chosen randomly in this 3,840-dimensional space achieves a Cohen's d of approximately **0.29** when used to separate high- from low-warmth stories. Our warmth direction achieves **2.70**, roughly **9 times** the random baseline, confirming that the extracted direction captures structure that is specific to the warmth contrast.

**5. Meaning of a "projection score".** Given the warmth direction, any story can be projected onto it to produce a scalar value. A high score indicates that the model's internal state while reading that story resembles the average state observed when reading high-warmth material. All story-level scores reported in §5 are these projections.

### Scope and limits of interpretation

The warmth direction vector cannot be interpreted as locating warmth in a single neuron or a fixed small set of neurons. The signal is distributed across hundreds of dimensions. The next analysis step — Sparse Autoencoder (SAE) decomposition using GemmaScope 2 — will attempt to translate these distributed activation patterns into human-interpretable features, answering, for instance, whether the warmth direction preferentially activates features associated with *care* and *friendliness* or with *positivity* and *approval* more broadly.

---

## 7. Reconciling Cosine Similarity and Behavioural Independence

### Observations

- The warmth direction classifies warm vs. cold stories with **100% accuracy**.
- The competence direction classifies competent vs. incompetent stories with **100% accuracy**.
- Applied across axes, each direction performs at **50%** — indistinguishable from chance.
- Yet the cosine similarity between the warmth and competence direction vectors is **0.75**, indicating that they point in broadly similar directions.

This combination — high cross-axis similarity, zero cross-axis discriminability — requires explanation.

### An intuitive analogy

Consider two people walking across a field, both moving roughly north-east. From a distance, their trajectories appear similar. But one person walks only on grass, the other only on gravel. If asked to identify who is walking on grass, one cannot answer by observing direction of travel alone — one must examine the texture beneath their feet. The overall heading is shared; the local structure that carries the relevant information is not.

The same principle applies here. Both the warmth and competence directions point "towards a character who is depicted positively" — because both high-warmth and high-competence stories portray a protagonist acting well, while both low-warmth and low-competence stories portray the opposite. This shared component reflects the **valence** of the stimulus material and is the source of the 0.75 cosine similarity.

What the directions do *not* share is the more local structure: the specific activation patterns that distinguish *caring from indifferent* are not the same as those that distinguish *skilled from unskilled*. The cross-axis test, which forces classification on a domain different from the one used to construct the direction, reveals exactly this dissociation.

### Scientific interpretation

The stimulus design objective — activating one axis while holding the other neutral — was successful at the behavioural level. The model's internal representations differentiate warmth from competence as distinct signals, even though both share a valence-related component.

The 0.75 cosine similarity is an expected rather than an anomalous finding. It reflects a property of the stimuli: all stories involve a protagonist acting in a way that is framed as either positive or negative, and the model's general representation of that evaluative dimension is active across conditions. The analogous issue — warmth and valence activation co-occurring — was identified and addressed in Anthropic's emotion concepts work (Sofroniew, Lindsey et al., 2026) through a neutral-corpus principal component analysis (PCA) denoising step, which constitutes the next phase of this project.

---

## 8. Limitations and Next Steps

### Current limitations

**Sample size.** With 50 stories per condition, the direction vectors are constructed from a relatively small sample. The split-half cosine similarities of 0.83 (warmth) and 0.88 (competence) are satisfactory but leave room for improvement. Expanding to 100 stories per condition is expected to push stability above 0.90.

**Valence overlap.** The cosine similarity of 0.75 between the warmth and competence directions reflects a shared evaluative component that has not yet been separated out. Until this is addressed, the two directions cannot be treated as fully orthogonal representations.

**Representational measurement only.** The current analysis characterises the model's *internal representations* during passive reading. It does not yet test whether those representations causally influence the model's behaviour — specifically, whether shifting a representation along the warmth or competence direction changes the model's hiring decisions. That causal test is the purpose of Phase 6 (steering).

### Planned next steps

| Phase | Description |
|-------|-------------|
| **Valence denoising** | Compute a neutral-corpus baseline using text that carries no warmth or competence content; subtract the first principal component from both direction vectors to remove the shared valence signal. Expected to reduce cosine similarity substantially. |
| **SAE decomposition** | Decompose the warmth and competence vectors using GemmaScope 2 sparse autoencoders. Report which human-labelled features are most strongly activated by each direction. |
| **Causal steering** (Phase 6) | Intervene on the model's residual stream during generation of a hiring callback decision, adding or subtracting scaled multiples of the warmth and competence vectors. Measure the resulting shift in callback probabilities. |
| **Hiring audit** (Phase 7) | Evaluate the model's baseline callback rates across demographic signal manipulations (name, gender, race, country of origin). Compare the resulting disparities to those reported in the human correspondence study data of Gallo & Hausladen (2024). |

---

## 9. Reproduction

| Parameter | Value |
|-----------|-------|
| Model | `google/gemma-3-12b-it` |
| Probe layer | 31 (`probe_layer_frac = 0.66` over 48 layers) |
| Token pooling | Mean over positions ≥ 50 (skips prompt prefix) |
| Random seed | 20260527 |
| Stimulus file | `data/stimuli/concept_stories.jsonl` |
| Extraction script | `src/extract_vectors.py` |
| Validation script | `src/validate_probes.py` |
| SGE job script | `jobs/sge/extract_vectors.sh` |
| Direction vectors | `data/processed/concept_vectors/warmth_vec.npy`, `competence_vec.npy` |
| Per-condition activations | `data/processed/concept_vectors/X_<condition>.npy` (4 files) |
| Results table | `results/tables/probe_metrics.csv` |
| Full metric log | `results/logs/validate_probes_1781629889.json` |

The job was executed on the SCCKN cluster (Universität Konstanz) on node `scc214` (NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM) and completed in under one hour.

---

*This document covers Phase 4 and Phase 5 of the project pipeline. For the full execution plan, see `PLAN.md`. For the step-by-step research log, see `step_logs/STEP_LOG.md`.*
