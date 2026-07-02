**Produced:** 2026-07-02  
**Model:** claude-sonnet-4-6 (documentation)  
**Scope:** All four models — hiring callback margins throughout Phase 6 and Phase 7  
**Status:** Active limitation — Emre cluster re-runs pending; paper caveat required

## Artifacts

- **Scripts:** `src/gemma_scope_causality.py` (fix at line 81), `src/hiring_audit.py`, `src/hiring_steering.py`, `src/hiring_disparity.py`, `src/dense_steering.py`
- **Inputs:** `results/tables/hiring_audit_gemma3_{12b,27b,llama31_8b,qwen3_14b}.csv` (to be regenerated)
- **Outputs:** All `results/tables/hiring_*.csv`, `results/logs/hiring_mediation_*.json` (to be regenerated)
- **Figures:** `paper/figures/fig17_hiring_steering_callback.{png,pdf}`, `fig18_hiring_disparity.{png,pdf}`, `fig19_hiring_mediation_forest.{png,pdf}` (to be regenerated)

---

# IMPORTANT LIMITATION: bf16 Quantisation of Callback Margins (Bug B1)

## What was discovered

On 2026-06-30, Jorge Lastra audited all hiring callback CSVs and found that **every output margin falls on a 0.125 grid** — only 7–8 distinct values across all 282 names. The root cause: `yes_no_margin()` in `src/gemma_scope_causality.py` computed `logit(Yes) − logit(No)` while the model tensors were still in **bf16** (BFloat16 format). At the logit magnitudes these models produce (~5–10), bf16 precision is too coarse to distinguish small differences — the subtraction result is rounded to the nearest multiple of 0.125.

This affected **every pipeline script and notebook that calls `yes_no_margin()`**, which is the sole source of callback margin values throughout Phase 6 and Phase 7.

## Code fix applied (partial)

**Fix (line 81, `src/gemma_scope_causality.py`):**

```python
# Before (buggy):
return float((logits[0, -1][yes_id] - logits[0, -1][no_id]).item())

# After (B1 fix):
next_token_logits = logits[0, -1].float()
return float((next_token_logits[yes_id] - next_token_logits[no_id]).item())
```

The `.float()` cast removes one rounding step: the subtraction itself now runs in float32. All four pipeline scripts import `yes_no_margin` from the shared source file, so they inherit the fix automatically — no per-script edits required.

The same fix was applied inline in `notebooks/06_hiring_steering_causality.ipynb` and `notebooks/07_hiring_audit.ipynb`.

## Why the fix is partial — not a complete solution

After applying the fix and re-running notebooks 06 and 07, **margins are still on the 0.125 grid**. The `.float()` cast eliminates rounding in the subtraction step, but the operands themselves (the individual Yes-logit and No-logit values) are produced by bf16 inference and are already coarse before they reach the subtraction. Two bf16-quantised numbers differing by a multiple of 0.125 still produce a multiple of 0.125 after float32 subtraction.

**A complete fix requires float32 inference** (`torch_dtype=torch.float32` at model load time), which approximately doubles GPU memory usage. This is likely infeasible at 27B on available cluster nodes.

## Model-specific impact

| Model | Margin SD | Unique values | Consequence |
|---|---|---|---|
| **Gemma-3-12B** | 0.14 | 7 | **R4 disparity unreliable.** Group gaps of ~0.5 SD ≈ 0.07 raw units are below one grid step (0.125). Cannot distinguish real group differences from quantisation noise. |
| **Gemma-3-27B** | 0.41 | 18 | **R4 disparity usable.** Group gaps of ~0.5 SD ≈ 0.20 units = 1.6 grid steps are detectable. |
| **Llama-3.1-8B** | expected > 0.30 | TBD | Likely usable — confirm SD after re-runs. |
| **Qwen3-14B** | expected > 0.30 | TBD | Likely usable — confirm SD after re-runs. |

## Which results are affected

**Affected (all outputs using callback margins):**

- `results/tables/hiring_audit_*.csv` — baseline callback margins per name
- `results/tables/hiring_steering_raw_*.csv` — per-name steering delta-margins
- `results/tables/hiring_disparity_*.csv` — group-level disparity gaps
- `results/logs/hiring_mediation_*.json` — bootstrap indirect effects
- `results/tables/hiring_audit_concept_vectors*.csv` — notebook 07 outputs
- `paper/2026-06-24_1136_hiring_causality_results.md` — 12B Phase 6+7
- `paper/2026-06-24_1300_hiring_causality_27b_results.md` — 27B Phase 6+7
- `paper/2026-06-27_1446_dense_steering_4model.md` — dense steering (all models)
- `paper/2026-06-27_1541_hiring_phase7_4model.md` — 4-model Phase 7 consolidated
- `paper/2026-06-30_1251_r4_disparity_name_level.md` — R4 disparity (12B and 27B)

**Not affected (no logit subtraction):**

- Probe training, concept-vector extraction
- Probe-vs-human Spearman ρ correlations (residual projections, not logits)
- Gemma Scope SAE feature analysis
- Layer sweep results
- PCA valence denoising (notebook 08)

## Re-runs required

Jorge's JupyterHub re-runs (notebooks 06 and 07) are **complete** as of 2026-06-30 — see `docs/rerun_checklist.md` Part 1.

Emre's cluster re-runs (**Part 2**) are **pending**. After pulling the repo, submit:

```bash
qsub jobs/sge/hiring_gemma3_12b.sh
qsub jobs/sge/hiring_gemma3_27b.sh
qsub jobs/sge/hiring_llama31_8b.sh
qsub jobs/sge/hiring_qwen3_14b.sh
```

Then re-run `notebooks/09_hiring_disparity_R4.ipynb` with the updated CSVs to regenerate R4 figures and mediation results.

**After each re-run, verify:**

```python
import pandas as pd
for model in ["gemma3_12b", "gemma3_27b", "llama31_8b", "qwen3_14b"]:
    df = pd.read_csv(f"results/tables/hiring_audit_{model}.csv")
    vals = df["callback_margin"].dropna()
    on_grid = ((vals * 8).round() / 8 == vals.round(3)).mean()
    print(f"{model}: unique={vals.nunique()}, SD={vals.std():.3f}, on_grid={on_grid:.0%}")
```

Expected: SD > 0.30 for 27B, Llama, Qwen. SD for 12B expected to remain ≈ 0.14 (inherent to bf16 inference at 12B's logit scale).

## Required paper disclosures

The following caveats must appear in the paper:

1. **Methods — Measurement section:** Callback margins are computed as `logit(Yes) − logit(No)` via float32 cast from bf16 inference. Due to bf16 quantisation at the logit level, margins remain on a 0.125 grid. This is an inherent property of bf16 inference at the logit magnitudes produced by these models, not a rounding error in post-processing.

2. **Gemma-3-12B disparity:** The 12B model produces only 7 unique margin values (SD = 0.14). Group-level callback disparities for 12B are below the resolution threshold (one grid step = 0.125) and are not reportable as reliable empirical findings. They are reported as a methodological limitation, not as evidence of the presence or absence of disparity. Float32 inference would be required for reliable 12B disparity measurement.

3. **Recommended framing:** Lead disparity results with 27B, Llama, and Qwen (SD sufficient for detection); relegate 12B to a footnote or limitations paragraph citing quantisation.
