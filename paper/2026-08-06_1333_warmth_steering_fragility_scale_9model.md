# Warmth Steering Fragility Across Nine Models — Is It a Scale Effect or a Gemma Effect?

**Produced:** 2026-08-06 13:33 (Europe/Berlin)
**Model(s):** Gemma-3-12B-it · Gemma-3-27B-it · Llama-3.1-8B-Instruct · Gemma-4-12B-it · Gemma-4-26B-A4B-it · Gemma-4-31B-it · Qwen3-14B · Qwen3.6-27B · Qwen3.6-35B-A3B
**Scope:** Cross-model analysis of the Limitations item "Fragile causal effect at scale" (`paper/paper/Ulu_Lastra.tex`), extending the original two-model observation (Gemma-3-12B vs. Gemma-3-27B) to all nine models using hiring-steering data already on disk. No new GPU inference; this is a re-analysis of existing tables.
**Status:** Complete

## Artifacts

- **Scripts:**
  - `paper/figures/fig_warmth_steering_fragility_9model.py` (new; produces the figure below and the console dose-response dump this report's numbers are drawn from)
  - `src/hiring_steering.py` (original steering-sweep script; not re-run here)
  - `src/summarize_hiring_steering.py` (produced the seven `*_local.csv` summary files consumed here; not re-run here)
- **Inputs:**
  - `results/tables/hiring_steering_llama31_8b_local.csv`
  - `results/tables/hiring_steering_qwen3_14b_local.csv`
  - `results/tables/hiring_steering_gemma4_12b_local.csv`
  - `results/tables/hiring_steering_gemma4_26b_a4b_local.csv`
  - `results/tables/hiring_steering_gemma4_31b_local.csv`
  - `results/tables/hiring_steering_qwen36_27b_local.csv`
  - `results/tables/hiring_steering_qwen36_35b_a3b_local.csv`
  - `results/tables/hiring_steering_raw_concept_vectors_gemma3_27b.csv` (per-name; aggregated to means here)
  - `results/tables/hiring_steering_raw_gemma3_12b.csv` (per-name, broad regime only; aggregated to means here)
  - Provenance: `results/logs/hiring_steering_summary_{llama31_8b,qwen3_14b,gemma4_12b,gemma4_26b_a4b,gemma4_31b,qwen36_27b,qwen36_35b_a3b}_local.json`, `results/logs/hiring_steering_gemma3_12b.json`, `results/logs/hiring_steering_gemma3_27b.json`
- **Outputs:** none (this report reads existing tables; no new CSV is produced)
- **Figures:**
  - `paper/figures/fig_warmth_steering_fragility_9model.png`
  - `paper/figures/fig_warmth_steering_fragility_9model.pdf`

## Summary

The manuscript's Limitations list originally observed fragility (a non-monotone,
sign-reversing causal response to warmth steering) by comparing exactly two
models: Gemma-3-12B, which responds strongly and monotonically, against
Gemma-3-27B, which responds positively at low strength and then reverses sign
at higher strength. Framed only over those two models, this reads as a generic
"it breaks down at larger scale" pattern.

Re-examining the same question with all nine models available (using the
nine-model local-regime hiring-steering sweep, already on disk from earlier
runs) does not support that generic framing. Fragility is real, but it is not
a function of parameter count: the two largest non-Gemma checkpoints in this
study, Qwen3.6-27B and Qwen3.6-35B-A3B, both respond monotonically to warmth
steering across the full local-regime grid, the same clean shape shown by the
smaller Llama-3.1-8B and Qwen3-14B. Every model that shows fragility, on
either the warmth or the competence axis, is a Gemma checkpoint. The corrected
finding is: **fragility is concentrated in the Gemma family, not caused by
model size in general.**

## Dose-response classification, all nine models

A model's warmth dose-response is classified **monotone** here if the mean
change in callback margin, $\Delta$, is non-decreasing across the full
strength grid from the most negative to the most positive value tested (to
within the roughly 0.01-0.02 logit-unit noise floor set by the bf16
callback-margin quantization documented elsewhere in Limitations). It is
classified **non-monotone / fragile** if $\Delta$ reverses direction
somewhere inside that grid, most visibly when positive steering strength
produces a *smaller* or *negative* $\Delta$ than a lower positive strength.

| Model | Regime | Warmth | Competence |
|---|---|---|---|
| Gemma-3-12B | broad ($\pm0.25,\pm0.50$) | monotone | **non-monotone** |
| Gemma-3-27B | local ($\pm0.05,\pm0.10$) | **non-monotone** | **non-monotone** |
| Llama-3.1-8B | local | monotone | monotone |
| Gemma-4-12B | local | monotone | monotone |
| Gemma-4-26B-A4B | local | monotone (near-inert) | **non-monotone** |
| Gemma-4-31B | local | **non-monotone** | **non-monotone** |
| Qwen3-14B | local | monotone | monotone (near-flat) |
| Qwen3.6-27B | local | monotone | monotone |
| Qwen3.6-35B-A3B | local | monotone | monotone |

Four of the five Gemma checkpoints (Gemma-3-12B, Gemma-3-27B, Gemma-4-26B-A4B,
Gemma-4-31B) show non-monotone behavior on at least one axis; only Gemma-4-12B
stays clean on both. All four non-Gemma checkpoints (Llama-3.1-8B, Qwen3-14B,
Qwen3.6-27B, Qwen3.6-35B-A3B) are monotone on both axes, with no exceptions.
Size alone does not predict the pattern within the Gemma family either:
Gemma-4-12B (12B parameters) is clean, while Gemma-3-12B (also roughly 12B
parameters) already shows competence fragility, so the split tracks model
family more cleanly than it tracks parameter count.

Warmth dose-response values (mean $\Delta$ callback margin per strength) are
plotted in `paper/figures/fig_warmth_steering_fragility_9model.png`, reproduced
below in table form for the record:

| Model | $\Delta$ at lowest $\alpha$ | $\Delta$ at second-lowest $\alpha$ | $\Delta$ at second-highest $\alpha$ | $\Delta$ at highest $\alpha$ |
|---|---|---|---|---|
| Gemma-3-12B (broad) | $-3.472$ | $-1.557$ | $+7.079$ | $+8.351$ |
| Gemma-3-27B | $-0.765$ | $-1.285$ | $+1.973$ | $-2.658$ |
| Llama-3.1-8B | $-0.340$ | $-0.195$ | $+0.218$ | $+0.469$ |
| Gemma-4-12B | $-2.988$ | $-1.336$ | $+0.703$ | $+1.049$ |
| Gemma-4-26B-A4B | $-0.254$ | $-0.127$ | $+0.018$ | $+0.062$ |
| Gemma-4-31B | $-1.137$ | $-0.341$ | $-0.065$ | $-0.442$ |
| Qwen3-14B | $-0.565$ | $-0.298$ | $+0.221$ | $+0.423$ |
| Qwen3.6-27B | $-1.202$ | $-0.642$ | $+0.633$ | $+1.196$ |
| Qwen3.6-35B-A3B | $-0.738$ | $-0.435$ | $+0.419$ | $+0.967$ |

## Interpretation

Reading the fragility pattern as Gemma-specific rather than scale-specific
connects to two things already reported elsewhere in this manuscript. First,
the Supplementary's nine-model mediation extension already notes that both
Gemma-3 checkpoints show no significant hiring-level mediation path on either
axis, while Gemma-4 breaks that pattern with several significant paths; a
representation that is causally unstable under direct steering, as found
here, is a plausible mechanistic companion to a representation that fails to
mediate the downstream decision at all. Second, the open architectural
question already flagged in Limitations, an elevated cosine similarity
between the warmth and competence directions specific to Gemma-3 relative to
Qwen3 and Llama-3.1, offers one candidate explanation for why Gemma
checkpoints in particular would show cross-axis interference under steering:
if the two directions are less geometrically separable in Gemma's residual
stream, amplifying one is more likely to perturb machinery the other axis
also depends on, producing the reversals observed here. This report does not
test that mechanism directly; it only establishes that the fragility pattern
correlates with model family, which narrows where such a mechanistic account
should be sought.

The practical implication for the paper's causal claim is a re-scoping, not a
retraction: "the warmth steering effect is fragile at larger scale" should
read as "the warmth steering effect is fragile in larger Gemma checkpoints,"
since the two largest checkpoints actually tested outside the Gemma family
(Qwen3.6-27B, Qwen3.6-35B-A3B) show no such fragility.

## Caveats

- **Regime mismatch for Gemma-3-12B.** Gemma-3-12B has no local-regime
  ($\pm0.05,\pm0.10$) hiring-steering run; only the broad-regime
  ($\pm0.25,\pm0.50$) sweep exists. Its row in both tables above is comparable
  to the other eight in shape and sign, not in absolute magnitude or strength
  scale, since it is drawn from wider steering coefficients on a different
  grid.
- **Cross-model magnitudes are not comparable.** Callback-margin logit scales
  differ across models (already documented in Limitations for the disparity
  analysis), so the $\Delta$ values in the tables above should be read for
  sign and shape within a model's own row, not compared in absolute size
  across rows.
- **bf16 quantization.** All callback margins are subject to the 0.125-unit
  quantization grid documented in Limitations ("Callback-margin
  quantization"); the near-inert Gemma-4-26B-A4B warmth axis and the
  near-flat Qwen3-14B competence positive side sit close enough to zero that
  some of their apparent shape may reflect this floor rather than a
  meaningful null response.
- **Raw vectors only.** Like the rest of the disparity/mediation pipeline,
  this analysis uses only raw dense steering vectors; whether denoised
  vectors show the same family-clustered fragility is untested (tracked
  separately in Limitations, "Raw vectors only in the disparity and
  mediation pipeline").
- **n = 5 Gemma vs. n = 4 non-Gemma checkpoints.** The family split rests on
  a small number of models per family; it is a suggestive pattern within this
  specific nine-model set, not a statistically tested claim about model
  families in general.
