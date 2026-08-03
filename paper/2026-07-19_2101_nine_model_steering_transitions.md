# Nine-Model Steering Transition Synthesis

- **Produced:** 2026-07-19 21:01 Europe/Berlin
- **Model:** Nine Gemma, Qwen, and Llama checkpoints
- **Scope:** Empirical warmth and competence callback transitions on the shared 60-name panel
- **Status:** Complete; warmth integrated in the main text and competence in the appendix

## Artifacts

- **Scripts:** `paper/figures/_steering_transition_flow_common.py`, `paper/figures/paper_figure4_hiring_warmth_transitions.py`, `paper/figures/supp_figure1_hiring_competence_transitions.py`
- **Inputs:** `results/tables/hiring_steering_raw_concept_vectors.csv`, `results/tables/hiring_steering_raw_concept_vectors_gemma3_27b.csv`, `results/tables/hiring_steering_raw_llama31_8b.csv`, `results/tables/hiring_steering_raw_qwen3_14b.csv`, `results/tables/hiring_steering_raw_gemma4_{12b,26b_a4b,31b}_local.csv`, `results/tables/hiring_steering_raw_qwen36_{27b,35b_a3b}_local.csv`
- **Outputs:** `results/tables/hiring_steering_transition_summary_9model.csv`
- **Figures:** `paper/figures/paper_figure4_hiring_warmth_transitions.{png,pdf}`, `paper/figures/supp_figure1_hiring_competence_transitions.{png,pdf}`

## Data contract

The synthesis uses the same 60 applications for every model. Positive steering
is evaluated at `+0.10 × mean residual norm` for Gemma-3, Gemma-4, and Qwen3.6.
Llama-3.1-8B and Qwen3-14B use their available broad-regime endpoint of `+0.50`.
The figure is therefore a categorical transition comparison, not a matched
cross-model effect-size comparison.

Every source table is validated for required columns, 60 unique names at
baseline and endpoint, and exact name-set agreement. The generator recomputes
all mean margins and the full 3×3 No/Tie/Yes transition matrix. Frozen expected
means and transition counts prevent silent input drift.

## Findings

Warmth steering moves the mean callback decision from No to Yes for
Gemma-3-12B and Llama-3.1-8B. Gemma-3-27B instead reverses from Yes to No at
`+0.10`, reproducing its non-monotone local response. The other six models keep
a Yes mean decision. Their unchanged categories conceal heterogeneous margin
effects, which range from `-0.442` at Gemma-4-31B to `+1.196` at Qwen3.6-27B.

Competence steering produces 22 strict No-to-Yes transitions at Gemma-3-12B,
60 Yes-to-No transitions at Gemma-3-27B, and 35 No-to-Yes transitions at
Llama-3.1-8B. The remaining six models keep all 60 applications in the Yes
category, although the mean margin falls at Gemma-4-26B-A4B, Gemma-4-31B, and
Qwen3-14B.

## Visual interpretation

The center lane is labeled `SAME INPUT`, not `NEUTRAL`, because it represents
the shared application before the counterfactual conditions separate. Gray
dashed paths end at the empirical baseline category, while blue paths end at the
steered category. When both conditions occupy the same lane, their small
vertical separation follows the sign of the observed margin change. Lane
identity, endpoint values, and transition counts are empirical; connector shape
and within-lane offsets remain schematic.

## Verification

The data and rendering scripts passed regression checks for all 18 model-axis
combinations. Ruff, whitespace, embedded-font, and PDF text checks passed.
Standalone 180-dpi renders and the final manuscript pages showed no clipping or
overlap. The 18-page LaTeX build has no undefined references, overfull boxes, or
compilation errors.
