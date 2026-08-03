# Steering Flip Pilot Designs

- **Produced:** 2026-07-19 17:06 Europe/Berlin
- **Model:** Gemma-3-12B-it
- **Scope:** Three alternative paper visualizations of a warmth-steering callback flip
- **Status:** Pilot set complete; no design selected or added to the manuscript

## Artifacts

- **Scripts:** `paper/figures/_pilot_steering_flip_common.py`, `paper/figures/pilot_steering_flip_deflection.py`, `paper/figures/pilot_steering_flip_boundary.py`, `paper/figures/pilot_steering_flip_split_flow.py`, `paper/figures/pilot_steering_flip_contact_sheet.py`
- **Inputs:** `results/tables/hiring_steering_raw_concept_vectors.csv`
- **Figures:** `paper/figures/pilot_steering_flip_deflection.{png,pdf}`, `paper/figures/pilot_steering_flip_boundary.{png,pdf}`, `paper/figures/pilot_steering_flip_split_flow.{png,pdf}`, `paper/figures/pilot_steering_flip_contact_sheet.{png,pdf}`

## Empirical anchor

All three pilots visualize the same Gemma-3-12B warmth intervention. At steering strength
`+0.05 × mean residual norm`, the mean callback margin moves from `-0.19375` without
steering to `+0.98333` with steering. Of the 60 names, 54 move strictly from No to Yes;
the remaining six move from an exact zero-margin tie to Yes.

The scripts recover these quantities directly from the raw name-level table and stop if
the two conditions do not contain the same 60 names or if the frozen pilot statistics no
longer match. Orange identifies the warmth intervention, blue identifies the resulting
Yes state, and gray identifies the unsteered counterfactual. Meaning is also encoded by
line style, labels, and geometry rather than color alone.

## Alternatives

1. **Deflected causal flow.** A common application path reaches the intervention point,
   after which the dashed unsteered branch continues to No while the solid steered branch
   bends to Yes. This option most closely matches the intended causal narrative.
2. **Decision-boundary bend.** Empirical mean margins are placed on a numerical axis, and
   the steering connector crosses the zero-margin decision boundary. This option gives the
   clearest quantitative reading.
3. **Aggregate split-flow.** Width-encoded flows show 54 baseline No cases and six ties
   entering the intervention and ending as 60 Yes cases. This option emphasizes replication
   across names.

## Interpretation constraint

The connectors summarize a controlled comparison between unsteered and steered forward
passes. They are not measurements of a token-wise or layer-wise hidden-state trajectory.
Only the endpoint margins, steering strength, and transition counts are empirical.

## Verification

The PNGs and vector PDFs were regenerated from the raw table. Poppler rendering at 180 dpi
showed no clipped or overlapping labels, and `pdffonts` confirmed embedded DejaVu Sans
TrueType fonts. The manuscript source and current manuscript figure selection were left
unchanged pending a user choice among the pilots.
