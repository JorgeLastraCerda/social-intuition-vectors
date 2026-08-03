# Steering Decision-Boundary Refinements

- **Produced:** 2026-07-19 17:21 Europe/Berlin
- **Model:** Gemma-3-12B-it
- **Scope:** Three refinements of the selected decision-boundary steering pilot
- **Status:** Pilot comparison complete; no design selected or added to the manuscript

## Artifacts

- **Scripts:** `paper/figures/_pilot_steering_flip_common.py`, `paper/figures/pilot_steering_boundary_kink.py`, `paper/figures/pilot_steering_boundary_hinge.py`, `paper/figures/pilot_steering_boundary_vector_addition.py`, `paper/figures/pilot_steering_boundary_refinements.py`
- **Inputs:** `results/tables/hiring_steering_raw_concept_vectors.csv`
- **Figures:** `paper/figures/pilot_steering_boundary_kink.{png,pdf}`, `paper/figures/pilot_steering_boundary_hinge.{png,pdf}`, `paper/figures/pilot_steering_boundary_vector_addition.{png,pdf}`, `paper/figures/pilot_steering_boundary_refinements.{png,pdf}`

## Empirical anchor

Each design uses the same Gemma-3-12B warmth intervention as the first pilot set.
The mean callback margin changes from `-0.19375` without steering to `+0.98333`
at `+0.05 × mean residual norm`. Fifty-four of 60 names move strictly from No to
Yes, while six move from an exact tie to Yes.

Horizontal positions encode the measured callback margins, and the dashed vertical
line at zero is the observed Yes/No decision boundary. Vertical placement, connector
curvature, and arrow angles are schematic. The steering intervention occurs at the
probe layer, not at the zero-margin boundary.

## Design alternatives

1. **Sharp counterfactual kink.** A gray unsteered direction terminates at the
   baseline No point. The orange intervention enters there, a dashed counterfactual
   remains on the No side, and the blue path turns across the boundary to Yes.
2. **Decision-boundary hinge.** The steered connector is split at zero, with an
   explicit hinge labeled `decision flips here`. The intervention arrow lands before
   the boundary so the figure does not imply that steering is applied at zero margin.
3. **Vector addition.** A gray baseline vector and orange steering contribution are
   arranged tip-to-tail, while a blue resultant reaches the measured Yes endpoint.
   The steering contribution is labeled with the empirical mean change, `+1.177`.

## Verification

All scripts passed Ruff and the repository whitespace check. The standalone PDFs and
the comparison PDF use embedded DejaVu Sans fonts. Poppler rendering at 180 dpi found
no clipped or overlapping labels. Existing pilot figures and the manuscript remain
unchanged pending selection.
