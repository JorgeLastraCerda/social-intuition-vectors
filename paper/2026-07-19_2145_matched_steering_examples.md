# Matched Steering Examples and Complete Transition Census

- **Produced:** 2026-07-19 21:45 Europe/Berlin
- **Model:** Nine Gemma, Qwen, and Llama checkpoints
- **Scope:** Main-text example selection and complete positive-endpoint appendix table
- **Status:** Complete

## Artifacts

- **Scripts:** `paper/figures/_steering_transition_flow_common.py`, `paper/figures/paper_figure4_hiring_bidirectional_examples.py`, `paper/figures/generate_figures.py`
- **Inputs:** `results/tables/hiring_steering_transition_summary_9model.csv` and its nine source hiring-steering tables
- **Outputs:** `results/tables/hiring_steering_transition_summary_9model.tex`, `paper/paper/Ulu_Lastra.pdf`
- **Figures:** `paper/figures/paper_figure4_hiring_bidirectional_examples.png`, `paper/figures/paper_figure4_hiring_bidirectional_examples.pdf`

## Selection contract

The main-text figure uses Gemma-3-12B and Gemma-3-27B for both warmth and
competence. This is a matched illustrative comparison rather than a prevalence
estimate: model family, the 60 applications, and the `+0.10 × mean residual
norm` endpoint are held constant while the observed mean transition direction
changes from No-to-Yes at 12B to Yes-to-No at 27B.

The complete result is a nine-row appendix table covering all 18 positive
model-axis endpoints. Each row reports the baseline margin and decision,
steered endpoint and decision, mean margin change, and every nonzero
No/Tie/Yes transition. Transition counts sum to 60 within every cell. Llama-3.1
and Qwen3 use their available `+0.50` endpoint; the other seven models use
`+0.10`, so raw effects are not treated as matched cross-model effect sizes.

## Findings

Positive concept steering does not have a fixed sign in the hiring callback
readout. Gemma-3-12B crosses from a negative to a positive mean callback margin
on both axes, whereas Gemma-3-27B crosses in the opposite direction. Six models
remain Yes-to-Yes for every name despite margin changes ranging from negative to
positive. Category preservation therefore reflects both downstream direction
alignment and distance from the decision boundary, not merely intervention
strength.

The manuscript now states that concept directions are constructed independently
as high-minus-low activation contrasts and are not optimized to increase the
callback margin. It also distinguishes empirical endpoints and transition counts
from schematic connector geometry. The earlier nine-panel warmth and competence
figures remain in the artifact inventory for provenance but are no longer
included in the active manuscript.

## Verification

The generator validated all frozen means, exact name-set agreement, and the full
3-by-3 transition matrix for each of the 18 conditions. Python lint, whitespace,
embedded-font, PDF text, standalone figure, appendix table, and manuscript-page
checks passed. The 18-page manuscript build completed without undefined
references, overfull boxes, or compilation errors.
