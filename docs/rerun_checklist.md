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

**Effect on existing results:**
- Probe-vs-human correlations (R3): **not affected** — those use residual-stream
  projections, not logit differences.
- Concept-level causality (Gemma Scope, `src/gemma_scope_causality.py` steering
  at concept prompts): **affected** — re-run if those slopes will be cited precisely.
- Hiring callback margins (R4 and all downstream): **affected for all 4 models**.
  Group-level disparity gaps (e.g. Black−White = −0.013 SD) are smaller than 0.125
  and completely unreliable until fixed. Mediation indirect effects (Llama IE=0.190)
  are large enough to likely survive, but must be reconfirmed.
- Hiring steering slopes (notebook 06): **affected** — steering deltas are differences
  of margins, each of which was on the 0.125 grid. Slopes will shift after the fix.

---

## Part 1 — JupyterHub (GPU required)

Run these on the **Full GPU (80GB)** node. Start in the repo root
(`cd /home/jovyan/normalcy-axis`).

### 1A. Notebook 07 — hiring audit, 12B (float32 fix)

**Why:** Regenerates `results/tables/hiring_audit_concept_vectors.csv` with
correct float32 callback margins. This file feeds notebook 09 (R4 figures).

```
Open: notebooks/07_hiring_audit.ipynb
Set:  VECTORS_SUBDIR = "concept_vectors"    (12B)
Run:  all cells
Check: the B1 diagnostic in notebook 09 should show NO comb pattern
```

### 1B. Notebook 07 — hiring audit, 27B (float32 fix)

**Why:** Same as above for `hiring_audit_concept_vectors_gemma3_27b.csv`.

```
Open: notebooks/07_hiring_audit.ipynb
Set:  VECTORS_SUBDIR = "concept_vectors_gemma3_27b"    (27B)
Run:  all cells
```

### 1C. Notebook 06 — 27B local regime, raw vectors (recover overwritten CSV)

**Why:** The raw local-regime run was overwritten when the denoised run saved to
the same filename. The notebook now uses separate filenames (`_denoised` suffix),
so this is safe to re-run.

```
Open: notebooks/06_hiring_steering_causality.ipynb
Set:  VECTORS_SUBDIR = "concept_vectors_gemma3_27b"
      USE_DENOISED   = False
      (STRENGTHS is already [-0.1, -0.05, 0, 0.05, 0.1])
Run:  all cells
Output: results/tables/hiring_steering_raw_concept_vectors_gemma3_27b.csv
```

Key result to verify: warmth Δ at +0.05 ≈ +1.97, Δ at +0.10 ≈ −2.66
(non-monotone — this is the scale-dissociation finding).

### 1D. Notebook 09 — R4 disparity figures

**Why:** After 1A/1B produce corrected margins, notebook 09 regenerates the
model-vs-human disparity scatter (Figure 3) and mediation regression with real
float32 numbers.

```
Open: notebooks/09_hiring_disparity_R4.ipynb
Run:  all cells
Check: B1 diagnostic at top of notebook should show "float32 OK"
```

---

## Part 2 — SCCKN Cluster (Emre's pipeline, 4 models)

**Why:** Emre's pipeline scripts (`src/hiring_audit.py`, `src/hiring_steering.py`,
`src/hiring_disparity.py`) all call `yes_no_margin()` from
`src/gemma_scope_causality.py`. The fix is already in the source file on the repo —
**Emre just needs to pull and re-submit the SGE jobs.**

These re-runs regenerate the 4-model tables that feed the paper's headline
disparity/mediation results.

### Scripts to re-run (tell Emre):

```bash
# Pull the float32 fix first
git pull

# Then re-submit the 4 hiring jobs
qsub jobs/sge/hiring_gemma3_12b.sh
qsub jobs/sge/hiring_gemma3_27b.sh
qsub jobs/sge/hiring_llama31_8b.sh
qsub jobs/sge/hiring_qwen3_14b.sh
```

**Outputs that will be regenerated (and should be committed after):**
```
results/tables/hiring_audit_gemma3_12b.csv        ← callback margins fixed
results/tables/hiring_audit_gemma3_27b.csv
results/tables/hiring_audit_llama31_8b.csv
results/tables/hiring_audit_qwen3_14b.csv
results/tables/hiring_steering_raw_gemma3_12b.csv ← steering deltas fixed
results/tables/hiring_steering_raw_gemma3_27b.csv
results/tables/hiring_steering_raw_llama31_8b.csv
results/tables/hiring_steering_raw_qwen3_14b.csv
results/tables/hiring_disparity_gemma3_12b.csv    ← group gaps fixed
results/tables/hiring_disparity_gemma3_27b.csv
results/tables/hiring_disparity_llama31_8b.csv
results/tables/hiring_disparity_qwen3_14b.csv
results/logs/hiring_mediation_gemma3_12b.json     ← bootstrap IEs fixed
results/logs/hiring_mediation_gemma3_27b.json
results/logs/hiring_mediation_llama31_8b.json
results/logs/hiring_mediation_qwen3_14b.json
```

**What to check after re-runs:**
- Margin SD should increase (more spread than 7–8 discrete values).
- Llama race×warmth mediation IE ≈ +0.190 — expect this to survive; confirm the
  CI still excludes 0.
- 27B group gaps — previously Black−White = +0.544 raw logit; recheck after fix.
- All four hiring_probe_vs_human logs: the Spearman ρ values should be unchanged
  (those use residual projections, not logit margins).

---

## Part 3 — After all re-runs complete

1. Update the paper (LaTeX) with corrected disparity/mediation numbers.
2. Write the R4 results paragraph (currently a `% TODO` in `Ulu_Lastra.tex`).
3. Write the Abstract (last piece — blocked on R4).

---

## What does NOT need re-running

- **Probe training / concept-vector extraction** — not affected (no logit subtraction).
- **Probe-vs-human Spearman correlations** — not affected (residual projections).
- **Gemma Scope SAE feature analysis** — not affected.
- **PCA valence denoising** — not affected (done, saved in `concept_vectors_denoised.npz`).
- **Layer sweep** — not affected.

---

*Last updated: 2026-06-30*
