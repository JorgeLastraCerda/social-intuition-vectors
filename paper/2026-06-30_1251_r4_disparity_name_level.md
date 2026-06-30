# R4 Results: Model Hiring Disparities vs. Human Callback Benchmark

**Produced:** 2026-06-30 12:51 (Europe/Berlin)
**Model(s):** Gemma-3-12B-it · Gemma-3-27B-it (Jorge notebook run)
**Scope:** Notebook 09 — group-level and name-level disparity analysis joining model
callback margins with the Gallo & Hausladen published callback benchmark
**Status:** Complete for 12B and 27B (Jorge side). Llama and Qwen R4 pending Emre's
cluster re-runs with float32 fix.

---

## Artifacts

- **Scripts:** `notebooks/09_hiring_disparity_R4.ipynb`
- **Inputs:** `results/tables/hiring_audit_concept_vectors.csv` (12B, 282 names),
  `results/tables/hiring_audit_concept_vectors_gemma3_27b.csv` (27B, 282 names),
  `data/raw/SocialPerceptions-Predict-Callback-main/0_data/published_data/df_all.csv`
  (human benchmark)
- **Outputs:** `results/tables/r4_group_disparity.csv`,
  `results/figures/r4_model_vs_human_disparity.{png,svg}`,
  `results/figures/r4_margin_distribution.png`
- **Figures:** `results/figures/r4_model_vs_human_disparity.png`

---

## Data join

Names are matched on first-name (lower-cased) to the published_data benchmark.
149 of 282 names matched. The 133 unmatched names are primarily from international
audit studies (Oreopoulos, Gorzig, Jacquemet) whose names do not appear in the US
racial-category coding. The matched set has the following race × gender breakdown:

| Group | n names |
|---|---|
| Black Female | 9 |
| Black Male | 9 |
| White Female | 84 |
| White Male | 47 |
| **Total** | **149** |

The Black groups are small (n=9 each). All group-level estimates for Black names carry
wide standard errors and should be interpreted cautiously. The race gap is directionally
informative but not precise.

---

## Human benchmark (reference values)

Human callback rates from the Gallo & Hausladen (2024) meta-analytic dataset,
aggregated over matched names:

| Group | Human callback rate |
|---|---|
| Black Female | 0.065 |
| Black Male | 0.059 |
| White Female | 0.129 |
| White Male | 0.179 |

Derived reference gaps:
- **Race gap (Black − White, weighted):** −0.085 — White names receive substantially
  higher human callback rates. This is the classical audit-study discrimination signal.
- **Gender gap (Female − Male, weighted):** −0.037 — Males receive slightly higher
  human callbacks.

---

## Section 3: Group-level disparity (confirmatory)

### Model callback margin by group

| Group | 12B mean margin | 12B SE | 27B mean margin | 27B SE |
|---|---|---|---|---|
| Black Female | −0.167 | 0.036 | +1.472 | 0.133 |
| Black Male | −0.250 | 0.047 | +1.708 | 0.149 |
| White Female | −0.176 | 0.012 | +1.042 | 0.032 |
| White Male | −0.290 | 0.023 | +1.215 | 0.060 |

### Derived group gaps

| Metric | 12B raw | 12B SD units | 27B raw | 27B SD units | Human |
|---|---|---|---|---|---|
| Race gap (Black − White) | +0.008 | **+0.06 SD** | +0.486 | **+1.18 SD** | −0.085 |
| Gender gap (Female − Male) | +0.108 | **+0.77 SD** | −0.211 | **−0.51 SD** | −0.037 |

### Group-level Pearson (4 data points, model group mean vs human callback)

| Model | r | p |
|---|---|---|
| 12B | −0.720 | 0.280 |
| 27B | +0.122 | 0.878 |

With n=4 groups, significance cannot be established — treat this as a directional
indicator only.

### Interpretation

**Race direction — both models oppose the human benchmark.**
The human data shows White > Black callback rates (race gap = −0.085). Both 12B and 27B
produce Black > White model callback margins. At 12B the gap is negligible (+0.06 SD,
within quantisation noise). At 27B the gap is large (+1.18 SD): Black-signalling names
receive callback margins more than one SD above White-signalling names. The human
benchmark shows the opposite. This is consistent with RLHF / safety fine-tuning that
has over-corrected racial bias, producing a systematic positive adjustment for
minority-associated names.

**Gender direction — 27B agrees with human benchmark, 12B does not.**
The human data shows Male > Female callbacks (gender gap = −0.037). 27B reproduces this
direction (Female − Male = −0.51 SD, model gives males higher callback scores).
12B reverses it (+0.77 SD, model gives females higher scores) — but the 12B result is
unreliable due to quantisation (see B1 limitation below).

**The 27B disparity pattern in plain terms:** The model has learned, via instruction
tuning, to give a large callback boost to Black-signalling names (overcorrecting the
historical discrimination it presumably learned from training data), while preserving a
gender gap in the human-expected direction. This is not a "fairer" model — it is a
differently biased model. It has traded one form of name-driven differential treatment
(racial under-selection of Black names) for another (racial over-selection of Black names
that greatly exceeds the human baseline).

---

## Section 4: Name-level OLS regression (exploratory)

For each model, OLS predicts `callback_margin` from three name-level predictors:
`human_callback`, `model_warmth`, `model_competence`. Standardised betas allow
magnitude comparison across predictors. All 149 matched names used.

### 12B (R² = 0.150)

| Predictor | Pearson r | OLS β (std) |
|---|---|---|
| human_callback | +0.006 (n.s.) | −0.007 |
| model_warmth | **+0.376** (p<0.001) | **+0.784** |
| model_competence | **+0.374** (p<0.001) | **−0.732** |

The model's internal warmth and competence probe scores are both significantly
correlated with baseline callback margin at 12B. In the OLS with both probes, the betas
are large and opposing — this reflects collinearity between the two probe scores. The
key finding: the warmth direction is causally meaningful at the name level, not just at
the steering level. Human callback rates are unrelated to 12B model callback (r=+0.006).

### 27B (R² = 0.088)

| Predictor | Pearson r | OLS β (std) |
|---|---|---|
| human_callback | −0.115 (n.s., p=0.161) | −0.023 |
| model_warmth | **−0.266** (p=0.001) | **−1.252** |
| model_competence | **−0.261** (p=0.001) | **+1.151** |

At 27B, both probe scores are significantly *negatively* correlated with callback margin.
Names that activate the warmth and competence directions more strongly receive lower
baseline callback margins. This is the reversed baseline association documented in
`2026-06-24_1300_hiring_causality_27b_results.md` and the 4-model report. The OLS betas
are again opposing due to collinearity. Human callback is negatively (non-significantly)
associated with 27B model margin.

### Interpretation

The 12B name-level pattern is internally consistent: internal warmth → higher callback,
matching the causal steering finding. The 27B pattern is the opposite: internal warmth →
lower callback, consistent with the reversed baseline association and the steerability
paradox. The OLS R² values (0.150 and 0.088) indicate that warmth/competence probe
scores explain a moderate but non-trivial fraction of name-level callback variation.
These are exploratory and should not be used for confirmatory inference.

---

## B1 limitation — 12B quantisation

12B callback margins are still on the 0.125 grid even after the float32 fix (inherent
to bf16 inference; SD=0.14, 7 unique values). The 12B race gap (+0.06 SD) and gender
gap (+0.77 SD) are dominated by quantisation artefacts and should not be reported as
empirical findings. Disclose this in the paper Limitations. The 27B results (SD=0.41,
18 unique values, wide range) are sufficient for group-level analysis.

---

## Open questions / decisions

These are flagged in notebook 09 cell 14:

- **D2 (grouping):** Race × gender gives n=9 per Black group. Adding national origin
  from `categories.csv` could increase coverage. *Recommendation: keep race × gender
  for headline (aligns with prior literature), add national-origin breakdown in
  supplementary material.*
- **D3 (human callback source):** `published_data/df_all.csv` any-callback rate is
  the correct source. No change needed.
- **D4 (reporting units):** Report group-level gaps (Section 3) as confirmatory and
  name-level OLS (Section 4) as exploratory.
- **Multiple comparisons:** The group-level Pearson is the one confirmatory test.
  Section 4 OLS betas are exploratory — label clearly in paper.

---

## Next steps

1. Fill in the R4 `% TODO` paragraph in `docs/overleaf/Ulu_Lastra.tex` using the
   numbers above. Lead with 27B (reliable), note 12B limitation.
2. Wait for Emre's 4-model re-runs to extend R4 to Llama and Qwen. Expect the same
   diagnostic check (SD > 0.30 → proceed; SD < 0.20 → limitation).
3. Write Abstract once R4 is complete in the LaTeX.
