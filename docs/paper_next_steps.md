# Paper Next Steps — Handoff

**Bar for everything below: publishable in the paper.** Items are ordered by
priority given the Jul 10 deadline. Each has *why it matters*, *concrete steps*,
and *decisions only Jorge can make*.

---

## 0. TL;DR priority order

1. **Fix the 27B steering confound** (saturation regime). Cheap, and it decides
   whether the "scale dissociation" is real or an artifact. Do this first.
2. **R4 — demographic benchmark.** This is the paper's missing headline result
   (the human comparison the title promises). Needs 3 decisions from Jorge.
3. **PCA denoising.** Mostly execution; scripts exist. Strengthens the
   representation result and the steering specificity claim.
4. **Profession / role-congruity extension.** Highest upside, highest scope cost.
   Decide: lightweight version in *this* paper, or paper #2.

---

## 1. Interpret and de-confound the Gemma-3-27B results

### What we see
- Probe→human correlation slightly *higher* at 27B (ρ=0.381/0.283 vs 0.355/0.230).
- Probe→callback correlation *reverses sign* at 27B (−0.16 vs +0.14).
- Baseline P(Yes) = 0.76 at 27B vs 0.45 at 12B; callback margins far more variable.
- Hiring steering ~flat at 27B (warmth slope +1.09, R²=0.026; competence non-monotonic).

### The catch (must address before claiming anything)
Notebook 06 steers hiring with the **broad** range `[-0.5 … +0.5] × mean_resid_norm`.
The earlier concept-level causality work found this range saturates and switched to
the **local** range `[-0.1 … +0.1]`. 27B's baseline is already saturated toward
"Yes" (P=0.76), so a flat steering response is exactly what a saturation artifact
looks like. The "27B is uncontrollable" result is currently confounded with
"27B was steered in the wrong regime."

### Steps
1. Re-run notebook 06 hiring steering at **both** 12B and 27B with the **local**
   regime `[-0.1, -0.05, 0, 0.05, 0.1]` (mirror `gemma_scope_causality.py`).
2. Add a **harder / more neutral hiring prompt** that does not saturate the 27B
   baseline (e.g. a more selective framing, or a weaker candidate profile), so the
   margin has room to move in both directions.
3. Re-estimate slopes and R² per axis per model. Compare to the broad-range run.
4. Decide the claim:
   - If the dissociation **survives** → genuine finding: scale improves
     representation fidelity but weakens the representation→decision causal bridge
     (plausible mechanism: instruction tuning / RLHF reshaping the decision layer).
     Keep the Discussion paragraph as written.
   - If it **disappears** → it was a saturation artifact; soften the paper to
     "causal effect replicates at 27B" and drop the dissociation claim.

### Decision for Jorge
- **D1.** Is the scale-dissociation worth keeping as a headline, contingent on the
  local-regime re-run? (Recommend: yes, run it; let the data decide.)

### Paper status
Currently written into Results + Discussion as a *finding*. Flag in the draft as
**pending the local-regime robustness check** — do not present as settled until step 1–3 done.

---

## 2. R4 — Model vs human callback disparity (the missing headline)

### What it is
The result the title promises: do the model's callback disparities across
demographic groups track the *human* callback disparities from Gallo & Hausladen?
Plus the mechanistic add: does the model's internal warmth/competence probe
*mediate* the name→callback effect?

### What exists already
- `hiring_audit_concept_vectors.csv` (+ 27B): 282 names, each with human
  warmth/competence ratings, model probe scores, model callback margin.
- Notebook 07 §4 is a **scaffold only** — it groups by `study`, which is a
  placeholder, not a demographic grouping.

### What's missing (and why it's a research decision, not a coding one)
- **Demographic group labels per name.** Source: Gallo/Hausladen category files
  under `data/raw/.../0_data/ratings/categories/`.
- **Human callback *rates*** (not rating data). Source: the meta-analysis callback
  outcomes under `0_data/extracted_data/` and `0_data/published_data/`.

### Steps
1. Map each name → demographic group using the category files.
2. Join the human callback rates for each group.
3. Compute, per group: model mean callback margin, human callback rate.
4. **Figure 3:** scatter of model callback disparity vs human callback disparity,
   one point per group, with correlation. (Direct parallel to Gallo & Hausladen's
   own warmth/competence→callback result.)
5. **Mediation:** name → model warmth/competence probe → model callback margin.
   Test whether probe scores statistically mediate the group→callback effect.
   This is the bridge from "bias exists" to "bias runs through the SCM representation."
6. Run at 12B (primary, cleaner regime) and report 27B alongside.

### Decisions for Jorge
- **D2 (grouping).** Group by the protected category each study manipulated
  (race / gender / national origin), harmonised into a small set. *Recommended*
  over grouping by study or by raw name-origin clusters.
- **D3 (human callback dataset).** Which file under `extracted_data` /
  `published_data`, and which callback definition (any callback vs interview-only).
- **D4 (unit of analysis).** Category-level (mirrors Gallo & Hausladen exactly,
  cleanest comparison) vs name-level (more power, less directly comparable).
  *Recommend category-level* for the headline, name-level for the mediation.

### Recommendations (read before starting R4)
- **Fix callback margins first (audit B1).** Recompute logits in float32 — the
  current margins are quantised to a 0.125 grid (bf16 precision), and the disparity
  signal is small enough that the quantum can swamp it. Do not run R4 on bf16 margins.
- **Mind the 12B signal compression (audit B2).** At 12B callback variance is tiny
  (SD≈0.14); the model says "No" to nearly everyone. The disparity test may be
  underpowered there. After the float32 fix, check which model actually carries a
  usable disparity signal and justify the primary-model choice on that — do not
  default to 12B just because its steering is cleaner.
- **Mirror Gallo & Hausladen's own analysis.** Their result is "human W/C ratings
  predict human callback rates." The cleanest model analogue is "does W/C (probe
  *and* human ratings) predict the *model's* callback rates, and how does the slope
  compare to humans?" Frame R4 as that parallel, with mediation as the mechanistic add.
- **Report disparity as an effect size, not just a correlation.** Per-group mean
  margins plus a model-vs-human disparity scatter (one point per group) communicate
  more than a single ρ and match how the benchmark paper presents it.
- **Multiple comparisons (audit D1).** Pre-register/declare the confirmatory tests
  and correct; label the rest exploratory.

### Paper status
Results paragraph 4 is a structured `% TODO`. This is the last piece needed
before the Abstract can be written. See also `docs/robustness_audit.md` (B1, B2, D1)
and `docs/profession_extension.md` (R4 generalises naturally to profession×group).

---

## 3. PCA valence denoising

### Why it matters
Warmth and competence vectors share cos≈0.75 (Gemma), i.e. a common valence
component. Denoising tests whether that entanglement is *just* valence:
- If after denoising the axes separate **and** still steer their own axis but not
  the other → strong "clean, separable SCM dimensions" result.
- If denoising **kills** the steering effect → the causal signal *was* the valence
  component. Also a real, reportable finding.

### What exists
- `scripts/build_neutral_corpus.py`, `src/extract_neutral.py`,
  `src/denoise_vectors.py` — written, synthetic verification passed
  (planted cos 0.906 → 0.071). Pending only because the sandbox can't reach HF;
  it's a **login-node** step.

### Steps
1. Login node: `python scripts/build_neutral_corpus.py` (1,500 Wikipedia intros,
   length-matched, valence-filtered).
2. `qsub jobs/sge/extract_neutral.sh` → neutral activations at each model's probe
   layer (must be done **per model**: each has its own layer/scale).
3. Run `denoise_vectors.py`: PCA on neutral activations, project out top PCs
   covering ≥50% variance from warmth & competence vectors.
4. Report pre/post: cos(W,C), per-axis Cohen's d, cross-axis leak. Refresh Fig 1
   (joint density) and Fig 4 (geometry) with denoised vectors.
5. **Re-run concept-level AND hiring steering with the denoised vectors** and check
   whether axis-specificity improves (does warmth steering stop moving competence?).

### Decisions for Jorge
- **D5 (variance threshold).** Keep the 50% threshold (per Sofroniew et al.) or
  sweep it. *Recommend* fix at 50% for the headline, report a sweep in supplement.
- **D6 (denoised vs raw for the causal story).** Decide whether the paper's primary
  steering results use raw or denoised vectors. *Recommend* report both: raw shows
  the effect exists, denoised shows whether it's axis-specific.

### Paper status
Methods paragraph written; Results paragraph 1 ends with "denoised results
forthcoming." Two `% TODO`s waiting on these numbers.

---

## 4. Profession / role-congruity extension (high upside, scope decision)

### The idea
Currently: vary the name, hold the job fixed (Administrative Assistant). Instead,
vary the **profession** across the SCM quadrants and test whether hiring bias
depends on the **congruence** between the applicant's stereotyped warmth/competence
and the job's demands.

### Why it's publishable
- Connects SCM to **role congruity / lack-of-fit theory** (Eagly & Karau; Heilman)
  — established social-psych frameworks, currently untested in LLM hiring with
  interpretability tools.
- Upgrades the contribution from "SCM leaks into hiring" to "SCM leaks into hiring
  *as a function of job–applicant stereotype fit*."
- Mechanistic test nobody has run: does steering warmth move callbacks **more** for
  warmth-demanding jobs than competence-demanding ones? That is a clean
  interaction prediction the steering pipeline can already measure.

### Lightweight, deadline-feasible version
1. Pick 4–6 professions spanning the SCM quadrants (e.g. caregiver/nurse =
   high-warmth; engineer/financial analyst = high-competence/low-warmth; doctor =
   high-both; plus a low-both anchor). Source profession warmth/competence ratings
   from published occupational-stereotype data so the axis values are external,
   not invented.
2. Re-run the baseline hiring audit (notebook 07) for each profession × the 282 names.
3. Re-run hiring steering (notebook 06, **local regime**) per profession.
4. **Key analysis:** interaction — is the warmth-steering slope larger for
   warmth-congruent jobs? Is baseline disparity larger for incongruent pairings?

### Honest assessment
- Real, novel, and the strongest single idea in this batch.
- But it's a scope expansion under a tight deadline. The lightweight version above
  is achievable *if* the pipeline is stable and PCA + R4 are already done.
- If time is short: ship the current single-profession paper, and make the
  profession × congruence design the explicit **paper #2 / future work** — written
  up prominently so the idea is staked out.

### Decision for Jorge
- **D7.** Lightweight profession extension in *this* paper, or stake it as paper #2?
  Depends on whether 1–3 above land with time to spare. *Recommend* decide after
  R4 and PCA are in hand.

---

## Dependencies / suggested sequence

```
27B local-regime re-run (1)  ─┐
PCA denoising (3) ────────────┼─→ enables clean steering numbers
R4 grouping + human data (2) ─┘
                                   └─→ Abstract can be written
Profession extension (4) ── only if (1)(2)(3) land early
```

Write the **Abstract last**, once R4 exists and the 27B story is settled.
