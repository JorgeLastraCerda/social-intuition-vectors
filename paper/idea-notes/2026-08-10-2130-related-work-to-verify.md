# Related Work to Verify Before Submission

- **Timestamp:** 2026-08-10 21:30 Europe/Berlin
- **Raised by:** Jorge
- **Status:** Flagged, not yet acted on. Jorge is contacting Emre separately.
- **Why this exists:** A search run while drafting the Introduction surfaced two papers
  close enough to this work that they affect how the contribution can be stated. Neither
  has been read in full, only abstracts and search summaries, so nothing was added to
  `references.bib` and no manuscript text cites them yet.

---

## 1. Artificial Impressions (highest priority)

- **Title:** Artificial Impressions: Evaluating Large Language Model Behavior Through the
  Lens of Trait Impressions
- **Venue:** EMNLP 2025, main conference
- **Links:** `arXiv:2510.08915`, `aclanthology.org/2025.emnlp-main.981.pdf`

**What the abstract claims.** Linear probes are fit on generated prompts to predict
impressions according to the two-dimensional Stereotype Content Model. The probes are then
used to study the relationship between those impressions and downstream model behavior,
and to examine which prompt features inform them. Reported findings include that models
report impressions inconsistently when asked but that impressions are more consistently
linearly decodable from hidden representations, and that impressions predict response
quality and hedging.

**Why it matters here.** This is the same construct (SCM warmth and competence), the same
general technique (linear decoding from hidden representations), and the same overall move
(relate internal representation to downstream behavior). A reader familiar with it will
expect to see it cited.

**What to check when reading it in full:**

1. How many models, and which families? This determines whether our nine-checkpoint scope
   is a meaningful difference or a marginal one.
2. Do they intervene, or only decode? The abstract describes probing and prediction. If
   there is no activation-steering component, our causal intervention is a genuine
   differentiator.
3. Do they benchmark against any human data, and if so what kind? Our comparison is
   against human ratings and published callback rates for the same applicant names.
4. Is the extraction a fitted probe or a mean-difference direction? We use
   mean-difference, following the emotion-vector method.

**Provisional differentiators, to confirm rather than assume:** causal steering rather
than decoding alone; hiring callback recommendations rather than hedging and response
quality; comparison against a meta-analysis of human correspondence studies.

---

## 2. Fair Outputs, Biased Internals (secondary)

- **Title:** Fair outputs, Biased Internals: Causal Potency and Asymmetry of Latent Bias in
  LLMs for High-Stakes Decisions
- **Link:** `arXiv:2605.15217`

**What the summary claims.** Activation steering is used to test whether representational
divergence is causally relevant to decisions, by adding difference vectors to prompts and
checking whether decisions change predictably. The decision domain appears to be credit
rather than hiring.

**Why it matters here.** The methodological move of "steer the difference vector, observe
whether a high-stakes decision changes" is close to our hiring-steering experiment. The
domain differs, and possibly the construct, but this should be positioned against rather
than ignored.

**What to check:** whether the difference vectors encode a social-perception construct or
a demographic one directly; whether any human benchmark is used; whether the paper reports
non-monotonic or range-dependent effects comparable to ours, since that would be a useful
corroboration rather than a threat.

---

## Consequence already applied to the manuscript

The Introduction was written to avoid any priority claim. It does not say "first," and the
nine-checkpoint scope is presented as scope rather than as a novelty claim. The
contribution is stated as connecting representation-level probing to the
correspondence-study tradition, with cross-model coverage supporting that comparison.

A search specifically for prior work benchmarking model warmth and competence
representations against a meta-analysis of human correspondence studies returned nothing
matching. That remains the most defensible distinctive element, but it is an absence of
evidence from one search session, not a systematic review.

---

## Suggested action

1. Read the Artificial Impressions methods section, answering the four questions above.
2. Decide where to cite it. The natural home is the Background paragraph on prior work,
   next to the existing behavioral-audit citations, framed as prior evidence that SCM
   structure is linearly decodable, with our addition being intervention plus the human
   benchmark.
3. Decide whether the credit-decisions paper warrants a citation or only awareness.
4. If either paper does something we claim as distinctive, adjust the Introduction's
   contribution paragraph accordingly. It was deliberately written so this can be done by
   editing one paragraph.

Both citations would also help the manuscript's citation balance, which currently stands at
28 distinct cited sources.
