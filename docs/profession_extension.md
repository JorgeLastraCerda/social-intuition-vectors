# Profession Extension — Design Document

**Idea (one line):** stop holding the job fixed. Vary the profession across the
warmth/competence space and test whether the model's hiring bias depends on the
*congruence* between an applicant's stereotyped warmth/competence and the
profession's stereotyped demands — then test it mechanistically with steering.

This converts the contribution from "the model encodes SCM and it leaks into
hiring" to "the model encodes SCM, and it leaks into hiring *as a function of
job–applicant stereotype fit*, which we can manipulate at the representation
level." That interaction is novel and publishable.

---

## 1. Theoretical grounding (the literature it connects to)

- **Stereotype Content Model (SCM).** Fiske et al. (2002): social groups are
  perceived along warmth and competence; the same two axes organise *occupational*
  stereotypes (see §2).
- **Role congruity theory.** Eagly & Karau (2002, *Psychological Review*):
  prejudice arises when a group's stereotyped attributes are incongruent with the
  attributes a role is believed to require. Originally for female leaders;
  generalises to any group×role mismatch.
- **Lack-of-fit model.** Heilman (1983; Heilman & Caleo 2018, *Group Processes &
  Intergroup Relations*): performance/competence expectations are driven by the
  perceived fit between a person's attributes and the job's required attributes.
- **Occupational stereotypes predict real segregation.** He, Kang, Tse & Toh
  (2019, *Journal of Vocational Behavior*): occupations are reliably stereotyped on
  warmth and competence, and those stereotypes predict race and gender segregation
  in the actual workforce — the empirical bridge between occupational SCM and
  hiring outcomes.

**The synthesis our paper would add:** prior work shows (a) groups carry
warmth/competence stereotypes, (b) jobs carry warmth/competence stereotypes, and
(c) mismatch predicts bias — all in *humans*. We test whether an LLM reproduces the
*interaction* internally, and whether steering the warmth/competence representation
moves callbacks more for congruent than incongruent group×job pairings. No prior
work connects role congruity to LLM hiring with interpretability tools.

---

## 2. Databases we can use (real, citable, mostly public)

### For profession warmth/competence ratings (the new axis values)
- **He, Kang, Tse & Toh (2019), *JVB*.** Occupations rated on warmth (warm,
  good-natured, sincere, friendly, well-intentioned, trustworthy) and competence
  (competent, capable, intelligent, efficient, skillful, confident). Author hosts
  PDFs/data (sonia-kang.com). *Primary recommended source — English, US, validated.*
- **Friehs et al. (2022), *Journal of Applied Social Psychology*.** Stereotype
  content of occupational groups in Germany. Useful robustness / cross-cultural
  check, and closer to the Konstanz context.
- **Personnel Review (2022), professionals' warmth/competence of occupations.**
  Recruiter sample; five-cluster solution over the W/C space.

Using *external, published* ratings (not ratings we invent) is what keeps the
profession axis values defensible to reviewers.

### For job characteristics / realistic role descriptions
- **O*NET** (onetcenter.org, U.S. Department of Labor). 900+ occupations, public,
  free, machine-readable. Provides Tasks, Skills, Knowledge, Work Activities, Work
  Context, Job Zone per occupation. Use it to (i) select real occupations, (ii)
  build *realistic, matched* job descriptions for the hiring prompt instead of the
  hand-written Administrative-Assistant template, and (iii) optionally derive a
  data-driven competence proxy (Job Zone / required education).

### For external probe validation (addresses circularity, see audit C1)
- **arXiv:2601.06316** — first sentence-level human-annotated warmth/competence
  dataset. Not a profession source, but a non-AI held-out set to validate the
  direction vectors.

---

## 3. Experimental design

### Core 2×2 logic
Cross **applicant warmth/competence signal** (via name, from Gallo–Hausladen) with
**profession warmth/competence demand** (from He & Kang). Predictions:

- **Baseline disparity:** name-group callback gaps are larger when the group's
  stereotyped profile is *incongruent* with the profession's demand (role congruity).
- **Causal (the mechanistic payoff):** steering the warmth direction shifts
  callbacks *more* for warmth-demanding professions than competence-demanding ones;
  symmetrically for competence. A significant steering×profession-type interaction
  is the headline result.

### Outcome measures (reuse existing pipeline)
- Baseline callback margin per (name, profession) — notebook 07, float32 (audit B1).
- Steering slope per (axis, profession) — notebook 06, **local regime** (audit A1).

---

## 4. Scope plan (three tiers)

### Tier 1 — Lightweight, this paper (target: 6 professions)
Pick 6 occupations at the extremes of the SCM quadrants, by He & Kang ratings:

| Quadrant | Example occupations (pick 1–2) |
|---|---|
| High warmth / high competence | doctor, nurse |
| High competence / low warmth | lawyer, engineer, accountant |
| High warmth / low competence | childcare worker, receptionist |
| Low warmth / low competence | telemarketer, cleaner |

6 professions × 282 names × {baseline + 5 steering strengths × 2 axes} is well
within the existing pipeline's budget. Deliverable: the steering×profession-type
interaction as a new figure. **This is the achievable, deadline-compatible version.**

### Tier 2 — Extended (target: up to 10–12 professions)
2–3 occupations per quadrant for within-quadrant replication and a continuous
(rather than categorical) profession-demand axis, enabling regression of bias on
the *degree* of congruence rather than a 2×2. Do this if Tier 1 lands with time
to spare.

### Tier 3 — Future development / paper #2
Full factorial with O*NET-grounded realistic job descriptions, multiple CV quality
levels, several models, denoised vectors, and a mediation model
(name → warmth/competence probe → callback, moderated by profession demand).
Stake this out explicitly in Future Work so the idea is claimed.

---

## 5. Is it worth it? Honest assessment

- **Upside:** high. It is the single most novel idea in the current backlog, rests
  on established theory (role congruity), uses public validated data (He & Kang,
  O*NET), and exploits the steering pipeline you already built for a test nobody
  has run.
- **Cost:** a real scope expansion under the Jul 10 deadline. Tier 1 is feasible
  *after* the two blockers are cleared (callback margins in float32; local-regime
  steering), because both directly affect whether the profession interaction is
  measurable.
- **Recommendation:** clear audit B1 + A1 → run Tier 1 (6 professions) → if the
  interaction is clean and time remains, extend toward Tier 2; otherwise ship Tier
  1 and write Tier 3 as Future Work.

---

## 6. Concrete next steps

1. Pull He & Kang occupational warmth/competence ratings; select the 6 Tier-1
   occupations and record their W/C coordinates.
2. (Optional but recommended) pull O*NET descriptors for those 6 to build matched,
   realistic job descriptions for the hiring prompt.
3. Parametrise the hiring prompt template by profession (notebook 06 / 07 +
   `src/generate_stimuli.py` `HIRING_PROMPT_TEMPLATE`).
4. Re-run baseline audit (float32) and steering (local regime) per profession.
5. Analyse the steering×profession-demand interaction; build the figure.
6. Write the Methods/Results additions and the role-congruity framing into the
   Introduction/Literature.

---

## Sources
- He, Kang, Tse & Toh (2019), *Journal of Vocational Behavior* — occupational
  warmth/competence ratings: https://www.sonia-kang.com/pdfs/
- Friehs et al. (2022), *J. Applied Social Psychology*: https://onlinelibrary.wiley.com/doi/10.1111/jasp.12872
- Occupational stereotypes (professionals), *Personnel Review* (2022): https://www.emerald.com/pr/article/51/2/603/326391/
- Eagly & Karau (2002), role congruity theory, *Psychological Review*: https://en.wikipedia.org/wiki/Role_congruity_theory
- Heilman & Caleo (2018), lack-of-fit framework: https://journals.sagepub.com/doi/abs/10.1177/1368430218761587
- O*NET database: https://www.onetcenter.org/database.html
- Sentence-level warmth/competence dataset (2026): https://arxiv.org/pdf/2601.06316
