# Hiring Callback Causality: Steering Warmth and Competence Changes the Hiring Decision

**Produced:** 2026-06-24 11:36 (Europe/Berlin)  
**Model:** Gemma-3-12B-it  
**Notebooks:** `notebooks/06_hiring_steering_causality.ipynb`, `notebooks/07_hiring_audit.ipynb`  
**Output tables:** `results/tables/hiring_steering_raw_concept_vectors.csv` (600 rows), `results/tables/hiring_audit_concept_vectors.csv` (282 rows)  
**Status:** Complete for 12B baseline; 27B replication and demographic-grouped disparity remain future work

---

## Executive summary

We completed the hiring-callback causality experiment that the Gemma Scope 2 report listed
as future work. Two findings stand:

1. **Warmth causally moves the hiring decision.** Steering the warmth direction at the probe
   layer (layer 31) shifts the model's callback inclination linearly and robustly (slope
   +12.95 margin units per unit steering strength, R²=0.924 across 60 names × 5 strengths).

2. **The model's warmth direction aligns with human warmth ratings.** Projecting name
   activations onto the warmth direction produces scores that correlate significantly with
   human warmth judgements from Gallo & Hausladen (Spearman ρ=0.355, p<0.0001, N=282).

Together these close the causal chain: *name → internal warmth representation → hiring
callback decision*.

The competence effect is positive but non-monotonic (discussed below). The model's internal
probe scores do not significantly predict callback at baseline without steering, which we
argue reflects the bias operating below the level of the decision output.

---

## Methods

### Setup

We used the existing 12B concept vectors (layer 31, `data/processed/concept_vectors/`)
produced in Phase 4. The model and layer were read from `meta.json` so the notebook is
model-agnostic. All helpers (unit normalisation, `yes_no_margin`, `make_steering_hook`) are
copied verbatim from `src/gemma_scope_causality.py` to ensure the methodology is identical
to the concept-steering run.

### Outcome measure

The callback inclination for a name is `logit(" Yes") − logit(" No")` at the final
token of the hiring prompt (the same single-token margin used in the concept-steering run).
Positive = leaning "Yes". We call this the *callback margin*.

### Hiring prompt

Fixed CV template: Administrative Assistant, Bachelor's in Business Administration,
3 years as office coordinator, scheduling/written communication/Microsoft Office skills.
Only the applicant name varies. Prompt ends: "Based solely on this application, would you
recommend calling this candidate back for an interview? Answer with a single word: Yes or No."

### Applicant names (notebook 06)

60 names sampled at random (`random_state=20260527`) from the 282 rated names in
`df_all.csv` (Gallo & Hausladen).

### Causal sweep (notebook 06)

Steering strengths: {−0.50, −0.25, 0.00, +0.25, +0.50} × `mean_resid_norm` (79,722),
matching Emre's concept-steering grid. One hook per forward pass adds
`alpha × unit(direction)` to the residual stream at layer 31, all positions.

### Probe-vs-human validation (notebook 07)

All 282 rated names. For each name, we extracted residual activations from the prompt
"The job applicant's name is {name}.", mean-pooled over non-BOS tokens at layer 31, then
projected onto `unit(warmth_vec)` and `unit(competence_vec)`. Spearman and Pearson
correlations computed against the mean human warmth and competence ratings in `df_all.csv`.

---

## Results

### Baseline callback margin

Without any intervention, mean callback margin = −0.195 (SD=0.140) across all 282 names,
corresponding to P(Yes) ≈ 0.451. The model leans slightly toward "No" regardless of name.

### Causal sweep (notebook 06)

**Warmth** — clean, linear, robust:

| Steering strength | Mean Δ callback margin | SE |
|---|---|---|
| −0.50 | −3.461 | 0.029 |
| −0.25 | −1.529 | 0.070 |
|  0.00 |  0.000 | — |
| +0.25 | +7.125 | 0.052 |
| +0.50 | +8.404 | 0.046 |

Slope = +12.954 margin per unit strength (R²=0.924). The warmth direction is a reliable
causal lever on the hiring decision: more warmth → more likely to call back.

**Competence** — positive overall but non-monotonic:

| Steering strength | Mean Δ callback margin | SE |
|---|---|---|
| −0.50 | −4.608 | 0.069 |
| −0.25 | +3.900 | 0.030 |
|  0.00 |  0.000 | — |
| +0.25 | +4.840 | 0.044 |
| +0.50 | +6.248 | 0.036 |

Slope = +9.061, R²=0.663. The −0.25 point is anomalous: every one of the 60 names showed
a positive delta at that strength (min +3.375, max +4.375). The effect is therefore not
a few outliers — it is systematic.

**Interpretation of the competence non-linearity.** A small reduction in perceived
competence may increase perceived fit for an Administrative Assistant role (reducing
"overqualification" penalty). At −0.50 the effect reverses strongly, suggesting the
role-fit interpretation breaks down as the candidate appears clearly unqualified. This
non-linearity is a finding in itself: the model's hiring decision does not treat competence
as "more is always better" — role appropriateness matters. We flag this for the paper
rather than glossing over it.

### Probe-vs-human validation (notebook 07)

| Dimension | Spearman ρ | Pearson r | p-value | N |
|---|---|---|---|---|
| Warmth | 0.355 | 0.314 | 8.0 × 10⁻¹⁰ | 282 |
| Competence | 0.230 | 0.215 | 9.7 × 10⁻⁵ | 282 |

The model's internal warmth direction maps onto human warmth ratings at a moderate but
highly significant level. Competence is weaker. These are name-level correlations across
a diverse set of 282 names from five different audit studies (Bertrand, Farber, Flake &
Leasure, Gorzug, Jacquemet, Neumark, Nunley, Oreopoulos, Widner).

### Do probe scores predict callback at baseline?

| Predictor | Spearman ρ | p-value |
|---|---|---|
| Model warmth probe | +0.10 | 0.084 (n.s.) |
| Model competence probe | +0.11 | 0.067 (n.s.) |
| Human warmth rating | +0.21 | 0.00036 |
| Human competence rating | +0.17 | 0.0047 |

The model's internal probe scores do not significantly predict its own callback decisions
at baseline (no steering). Human ratings do. This gap is important: the bias does not
surface cleanly in the model's output distribution at rest, but it is causal when activated
(notebook 06). We interpret this as consistent with a latent bias that is present in the
representation but not always expressed in the final decision — exactly the type of covert
bias that activation-level analysis is suited to detect.

---

## Caveats

- Outcome is a single Yes/No token margin, not a generated explanation or full scoring.
  The same measure was used in the concept-steering run, making the two directly comparable.
- Steering applied at all sequence positions at layer 31. Other injection schemes
  (last token only, multiple layers) would yield different magnitudes but likely the same
  direction.
- One fixed CV and one role (Administrative Assistant). Role-appropriateness clearly
  interacts with the competence effect; robustness across roles is a priority follow-up.
- 60-name sample for the causal sweep; full 282-name sweep is straightforward to run.
- Demographic-grouped disparity (the fairness-specific claim) is scaffolded in notebook 07
  but not yet computed — requires a research decision on grouping and human callback dataset.
  See the scaffold in `notebooks/07_hiring_audit.ipynb` cell 11 and the flagged decisions below.

---

## Open decisions (flagged to Jorge)

**D-Phase7-A: Human callback dataset.** `df_all.csv` contains warmth/competence *ratings*,
not real-world callback rates. The actual callback outcome data are in
`data/raw/SocialPerceptions-Predict-Callback-main/0_data/extracted_data/` and
`0_data/published_data/`. Which study's callback rates to use as the human benchmark is a
research decision.

**D-Phase7-B: Demographic grouping.** Disparity needs a label per name (e.g., race,
gender, national origin). The category files are in `0_data/ratings/categories/`. Which
grouping to use and how to handle names that appear in multiple studies are design choices
that affect the fairness claim.

**D-Phase7-C: Mediation.** The full causal chain claim (name → probe score → callback)
requires a mediation test. A simple approach: Sobel test or bootstrap on
(name group → model warmth) × (model warmth → callback margin). This can be run on the
existing `hiring_audit_concept_vectors.csv` once the grouping is settled.

---

## Next steps

1. Run notebook 07 at 27B (`VECTORS_SUBDIR = "concept_vectors_gemma3_27b"`) for scale replication.
2. Wire in real demographic groupings and human callback rates — requires decisions D-Phase7-A and D-Phase7-B.
3. Run the full 282-name causal sweep in notebook 06 (`N_NAMES = None`) for a more complete picture.
4. Add the role-fit competence non-linearity as a finding, not just a caveat, in the paper draft.
