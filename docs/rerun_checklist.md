# Re-run Checklist — Post Float32 Fix (B1)

**Why these re-runs exist:**  
On 2026-06-30 we discovered that `src/gemma_scope_causality.py::yes_no_margin()`
was computing `logit(Yes) − logit(No)` while the model tensors were still in bf16.
At the logit magnitudes Gemma and Llama use (~5–10), bf16 can only represent
differences in steps of **0.125**, so the result was effectively rounded to one of
7–8 discrete values across all 282 names. Verified: every output CSV that calls
`yes_no_margin()` has this quantisation.

**Fix applied:** `logits[0, -1].float()` before the subtraction, in:
- `src/gemma_scope_causality.py` — the shared source used by all pipeline scripts
- `notebooks/06_hiring_steering_causality.ipynb` — local inline copy
- `notebooks/07_hiring_audit.ipynb` — local inline copy

**Important — deeper cause found after re-runs (2026-06-30):**  
After re-running notebooks 06 and 07 with the float32 fix, callback margins are
**still 100% on the 0.125 grid** for both 12B and 27B. The subtraction fix eliminates
one rounding step, but the root cause is that the model's output logit values are
**bf16-quantized before subtraction**. At these logit magnitudes the Yes-logit and
No-logit for this model naturally differ by multiples of 0.125 — a property of the
model weights, not the subtraction code. A complete fix requires float32 inference
(2× memory; may not be feasible).

**Practical consequence — model-dependent:**
- **27B, Llama, Qwen:** SD of callback margins is large enough (expected > 0.3) that
  group-level disparity gaps of ~0.5 SD will span multiple grid steps and be
  detectable. Re-runs are still needed to confirm.
- **12B (Gemma):** SD = 0.14, only 7 unique values in a 0.75-unit range. Any group gap
  smaller than one grid step (0.125) is invisible. **R4 disparity analysis for 12B
  cannot be made reliable without float32 inference.** Report as a limitation.
- **Margin SD will not dramatically increase after re-runs** — the quantisation is
  inherent to bf16 inference. What changes is that Emre's pipeline now uses the correct
  subtraction precision, which may slightly shift slopes and mediation estimates. The
  critical check is SD and unique-value count, not whether margins become continuous.

**Effect on existing results:**
- Probe-vs-human correlations (R3): **not affected** — those use residual-stream
  projections, not logit differences.
- Concept-level causality (Gemma Scope steering at concept prompts): **affected** —
  re-run if those slopes will be cited precisely.
- Hiring callback margins (R4 and all downstream): **affected for all 4 models**.
  Group-level disparity gaps for models with narrow margin SD (12B) are unreliable.
  Disparity and mediation for 27B, Llama, Qwen should be checked after re-run.
  Mediation indirect effects (Llama IE=+0.190) are large and expected to survive.
- Hiring steering slopes (notebook 06): **affected** — re-run with local regime.

---

## Part 1 — JupyterHub (GPU required)

Run these on the **Full GPU (80GB)** node. Start in the repo root
(`cd /home/jovyan/normalcy-axis`).

### ✅ 1A. Notebook 07 — hiring audit, 12B (DONE 2026-06-30)

Re-ran with float32 fix. Result: still 7 unique values on 0.125 grid (SD=0.14) —
this is inherent to bf16 inference at 12B's logit magnitudes, not a code bug.
**12B R4 disparity numbers remain unreliable; disclose as limitation.**
`results/tables/hiring_audit_concept_vectors.csv` updated and committed.

### ✅ 1B. Notebook 07 — hiring audit, 27B (DONE 2026-06-30)

Re-ran with float32 fix. Result: 18 unique values on 0.125 grid (SD=0.41, range 2.25).
Despite being on the 0.125 grid, the SD is large enough that group gaps of ~0.5 SD
≈ 0.20 raw units = 1.6 grid steps are detectable. **27B R4 analysis can proceed.**
Profession quick-check also ran: role-congruity confirmed (Nurse warmth ρ > Engineer
warmth ρ). `results/tables/hiring_audit_concept_vectors_gemma3_27b.csv` updated.

### ✅ 1C. Notebook 06 — 27B local regime, raw vectors (DONE 2026-06-30)

Raw local-regime results confirmed: warmth Δ=+1.97 at +0.05, Δ=−2.66 at +0.10
(non-monotone). Also ran 12B denoised steering as a nice-to-have (clean linear warmth
response confirmed). CSVs committed.

### ✅ 1D. Notebook 09 — R4 disparity figures (DONE 2026-06-30)

Run complete. Key findings:

**Group-level disparity (149 matched names; Black n=9 per gender, White n=131):**

| Gap | 12B | 27B | Human benchmark |
|---|---|---|---|
| Race (Black − White) | +0.06 SD *(noise)* | **+1.18 SD** | −0.085 (White > Black) |
| Gender (Female − Male) | +0.77 SD *(unreliable)* | **−0.51 SD** | −0.037 (Male > Female) |

- 27B race gap **opposes** the human benchmark: model massively favours Black names
  (+1.18 SD) while humans show White names receive higher callback rates. Consistent
  with RLHF over-correction.
- 27B gender gap **matches** the human direction: model and human both give males
  higher callback scores.
- 12B results are quantisation-dominated (SD=0.14, 7 unique values) — not reportable.

**Name-level OLS (exploratory, 149 names):**
- 12B: model_warmth r=+0.376 → callback (expected direction), R²=0.150
- 27B: model_warmth r=−0.266 → callback (reversed, consistent with steerability paradox), R²=0.088

See `paper/2026-06-30_1251_r4_disparity_name_level.md` for full report.
Outputs committed: `results/tables/r4_group_disparity.csv`, `results/figures/r4_model_vs_human_disparity.{png,svg}`.

---

## ✅ Part 2 — SCCKN Cluster (Emre's pipeline, 4 models) — DONE 2026-07-02

Emre re-ran all four hiring SGE jobs after pulling the float32 fix. SHA-256 checks
confirmed the rewritten files were **byte-identical** to the git-tracked versions:
no new margin values were produced. This confirms that the B1 float32 subtraction
fix does not change result values (as expected: the remaining quantisation is
inherent to bf16 model outputs, not the subtraction).

**Current source of truth:** CSV and JSON files already committed to the repo are
the authoritative post-fix values. No further SCCKN re-runs are required for B1.

**Confirmed outputs (byte-identical to pre-re-run, already committed):**
```
results/tables/hiring_audit_gemma3_{12b,27b,llama31_8b,qwen3_14b}.csv
results/tables/hiring_steering_raw_gemma3_{12b,27b,llama31_8b,qwen3_14b}.csv
results/tables/hiring_disparity_gemma3_{12b,27b,llama31_8b,qwen3_14b}.csv
results/logs/hiring_mediation_gemma3_{12b,27b,llama31_8b,qwen3_14b}.json
```

**Final result check (actual values from committed CSVs):**
- Llama race×warmth mediation IE = +0.190, CI [+0.111, +0.292] — confirmed significant
- 27B race gap = +1.18 SD (Black > White) — direction stable
- Qwen/Llama disparity: see `paper/paper/Ulu_Lastra.tex` Results §Disparities

---

## ✅ Part 3 — Writing — COMPLETE

All R4 paragraph writing is complete in `paper/paper/Ulu_Lastra.tex` (the active
manuscript; note: **not** `docs/overleaf/Ulu_Lastra.tex`, which is stale).
All four models' disparity and mediation results are in the paper.

### Reference for what was written:
1. R4 disparity paragraph: 27B numbers from `paper/2026-06-30_1251_r4_disparity_name_level.md`,
   - Lead with 27B: race gap +1.18 SD (Black > White, **opposes** human direction),
     gender gap −0.51 SD (Female < Male, **matches** human direction).
   - Note 12B as quantisation-limited (limitation, not a finding).
   - Exploratory OLS: 12B warmth r=+0.376, R²=0.150; 27B warmth r=−0.266, R²=0.088.
2. Update 27B steering paragraph: replace "warmth inert" with "warmth non-monotone/
   fragile" (local-regime: Δ=+1.97 at +0.05, collapses to −2.66 at +0.10).
3. Add steerability paradox section (12B most steerable → null mediation; Llama
   least steerable → strongest mediation IE=+0.190).

### After Emre's cluster re-runs:
4. Re-run notebook 09 with Emre's 4-model CSVs to extend R4 to Llama and Qwen.
5. Add Llama/Qwen disparity and mediation numbers to R4 paragraph.
6. Write Abstract (last piece — was blocked on R4, now unblocked for 27B draft).

---

## What does NOT need re-running

- **Probe training / concept-vector extraction** — not affected (no logit subtraction).
- **Probe-vs-human Spearman correlations** — not affected (residual projections).
- **Gemma Scope SAE feature analysis** — not affected.
- **PCA valence denoising** — not affected (done, saved in `concept_vectors_denoised.npz`).
- **Layer sweep** — not affected.
- **Notebooks 06, 07 (Jorge side)** — ✅ done (see Part 1 above).

---

*Last updated: 2026-06-30 (revised: bf16 inference quantisation inherent; Part 1 items marked complete)*
