# Concept Stimuli — Generation Tracker

Control sheet for building the concept story corpus over several runs. The
authoritative data is `concept_stories.jsonl`; this file logs progress and the
design rules each run must respect.

**Refresh live numbers:**
`python scripts/validate_stimuli.py` (rule pass/fail + counts) and
`python scripts/audit_stimuli.py` (topic coverage + demographic balance).

---

## Target

From `config/config.yaml` (`probing`): 100 topics x 12 stories per topic x 4
conditions = **4,800 stories** = **1,200 per condition**.

| Condition | Target | Depicts (shown, never named) | Off-axis held neutral |
|---|---|---|---|
| high_warmth | 1,200 | genuine care, attentiveness, concern for others | competence |
| low_warmth | 1,200 | emotional distance, self-focus, indifference | competence |
| high_competence | 1,200 | effectiveness, careful thinking, reliable execution | warmth |
| low_competence | 1,200 | disorganisation, poor judgement, repeated errors | warmth |

---

## Current status

_Last updated 2026-06-16 (Run 6). From `validate_stimuli.py` / `audit_stimuli.py`._

**Smoke-test milestone target = 200 stories (50 per condition)** — reached.
(The full-corpus target from config remains 4,800; 200 is the batch built for the
expanded smoke test before tomorrow.)

| Condition | Done | Words (min/mean/max) |
|---|---|---|
| high_warmth | 50 | 90 / 103 / 148 |
| low_warmth | 50 | 90 / 100 / 134 |
| high_competence | 50 | 90 / 102 / 145 |
| low_competence | 50 | 90 / 101 / 144 |
| **Total** | **200** | balanced across conditions |

Protagonists **name-free / demographically neutral**. **50 of 100 topics** covered,
1 story per condition each, spanning all **10 domains** (Workplace 11, Learning 5,
Social 4, Health 4, Community 5, Money 4, Sport 5, Travel 5, Technology 4, Misc 3).
Validator PASS, 0 rule violations; audit shows no demographic skew.

---

## Run log

| Run | Date | Generator | Added | Notes |
|---|---|---|---|---|
| 1 | 2026-06-15 | claude-opus-4-8 (Cowork) | 20 (named) | **Superseded.** Audit caught generator bias (low conditions all male/Anglo). |
| 2 | 2026-06-15 | claude-opus-4-8 (Cowork) | 20 (name-free) | Pilot, 5 domains. D4=neutralise. |
| 3-6 | 2026-06-15/16 | claude-opus-4-8 (Cowork) | +180 (name-free) | Workplace, Learning, Social, Health, Community, Money, Sport, Travel, Technology, Misc. Built corpus to 200 (50/condition, 50 topics) in validated batches; short stories padded to keep lengths balanced. |

> The committed full-run generator (`src/generate_stimuli.py`) uses the Anthropic
> **API** (`claude-haiku-4-5`), separate from a Pro subscription. This 200-story
> corpus was written via Claude in Cowork (no API cost), model recorded in each
> story's `generation_model` field.

---

## Design rules every story must follow

Hard rules (auto-checked by `validate_stimuli.py`, non-zero exit on failure):

1. **Show, don't tell** — quality demonstrated through actions; concept word and
   close variants forbidden (stem list in the script).
2. **Valid record** — required fields, known condition, unique id.
3. **Length band** — ~120-180 words (warns outside 90-200) so the probe cannot
   learn warmth from sentence length.

Soft rules (judgement; spot-check a sample each run):

4. **Hold the off-axis dimension neutral** — when manipulating warmth keep the
   person averagely capable, and vice versa. Reduces warmth-competence
   entanglement inside the model.
5. **Decouple valence where possible** — "low" conditions are naturally more
   negative; soften where you can so the probe captures the trait, not mood.
6. **Behavioural diversity** — vary concrete behaviours and sentence openings so
   the probe does not latch onto a recurring phrase.
7. **Name-free protagonists (D4)** — use "they"/role words, no proper names, so
   the warmth/competence direction carries no gender or origin signal. Names
   enter only at the measurement stages (Fig 1 / steering / benchmark), sourced
   from Carina's validated name set.
8. **One concept per story**, single protagonist, realistic register, past tense.

---

## Demographic balance (protected attributes)

Audited by `audit_stimuli.py` (joins `concept_stories.jsonl` with
`protagonist_metadata.jsonl`). Tracks gender, name-origin cue, and age /
disability / religion cues by condition; warns if a level exceeds 70% of a
condition.

- **Concept stories (current):** name-free, so every cue is "unspecified" /
  "unmarked" — no demographic signal can confound the vector. Audit clean.
- **Where the audit now earns its keep:** profiling the **hiring-stage name set**
  (Carina's 282 names) to show it spans groups, and re-checking any future
  stimuli that reintroduce names.
- `name_roster.csv` (balanced names) is retained in case option D4(b)
  (balance-by-design) is ever revisited.
- **Historical caution (Run 1):** the named pilot was badly skewed (low
  conditions all male / Anglo). Kept as the worked example of generator bias.

---

## Topic coverage

`audit_stimuli.py` reports per-topic counts per condition and how many topics
reached full depth (12/condition). The 100 topics are content *scenarios*
(family, work, travel, sport, ...) across 10 domains — not the treatment. Every
topic is written in all 4 conditions so the concept varies while the scenario is
held constant. The corpus now covers 50 of 100 topics across all 10 domains, 1 story per condition each (0 topics at the full 12/condition depth).

---

## Open research decisions

- **D1 — Fold soft rules into `src/generate_stimuli.py`? (PENDING)** Will update
  the API prompt to enforce name-free protagonists, off-axis-neutral, and
  behavioural diversity once D3 is settled. (Jorge: fold "when we finish deciding".)
- **D2 — Figure 1 design — DECIDED (Jorge):** extract warmth and competence as
  two **separate** probes (one-way door: aggregation is reversible, separation is
  not). In analysis: report each vs its own human rating, AND test the
  correlation between the two probe scores; if strongly positive like Carina's
  data, run PCA to reproduce her PC1/PC2 and correlate against her published PC1.
  Delivers both the apples-to-apples comparison and the separate-dimension story.
- **D3 — Stories per topic (PENDING).** 12/topic is large and strains story
  diversity; consider 2-3/topic across all 100 topics instead, leaning on topic
  variety. Cheaper to generate and check.
- **D4 — Demographics in concept stories — DECIDED (Jorge): NEUTRALISE.** No
  names/gender in concept stories; all protected-attribute manipulation lives in
  the hiring/measurement stage using Carina's validated names. Age/disability/
  religion stay unmarked in concept stories (forcing them risks stereotyping).
