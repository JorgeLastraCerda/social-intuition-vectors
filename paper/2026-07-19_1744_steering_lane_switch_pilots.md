# Steering Lane-Switch Pilots

- **Produced:** 2026-07-19 17:44 Europe/Berlin
- **Model:** Gemma-3-12B-it
- **Scope:** Three schematic Neutral-to-No/Yes lane-switch visualizations for warmth steering
- **Status:** Pilot comparison complete; no design selected or added to the manuscript

## Artifacts

- **Scripts:** `paper/figures/_pilot_steering_flip_common.py`, `paper/figures/pilot_steering_lane_switch_diagonal.py`, `paper/figures/pilot_steering_lane_switch_step.py`, `paper/figures/pilot_steering_lane_switch_smooth.py`, `paper/figures/pilot_steering_lane_switch_variants.py`
- **Inputs:** `results/tables/hiring_steering_raw_concept_vectors.csv`
- **Figures:** `paper/figures/pilot_steering_lane_switch_diagonal.{png,pdf}`, `paper/figures/pilot_steering_lane_switch_step.{png,pdf}`, `paper/figures/pilot_steering_lane_switch_smooth.{png,pdf}`, `paper/figures/pilot_steering_lane_switch_variants.{png,pdf}`

## Empirical anchor

All three pilots show the same Gemma-3-12B warmth intervention at
`+0.05 × mean residual norm`. The mean callback margin changes from `-0.19375`
without steering to `+0.98333` with steering. Fifty-four of 60 names move
strictly from No to Yes, and six move from an exact tie to Yes.

The endpoint labels, intervention strength, and transition counts are empirical.
The horizontal process direction, lane positions, switch location, and connector
geometry are schematic. They communicate a counterfactual decision process, not
a measured residual-stream trajectory.

## Design alternatives

1. **Diagonal lane switch.** The shared application flow reaches a neutral gate,
   then the blue steered route makes a sharp diagonal move to the Yes lane. A
   dashed gray counterfactual moves to the No lane. Both routes settle into exact
   horizontal flows before their endpoints.
2. **Right-angle switch.** The steering gate produces an immediate orthogonal
   change from Neutral to Yes, while the no-steering counterfactual drops to No.
   This is the most literal visual rendering of a discrete intervention.
3. **Smooth lane switch.** Short S-shaped connectors move from the gate to the
   two outcome lanes before becoming horizontal. This version presents the same
   causal contrast with a softer visual transition.

## Interpretation constraint

The common neutral segment represents the same application before the displayed
counterfactual split. The solid blue branch is the observed steered condition.
The dashed gray branch is the corresponding unsteered outcome. The orange gate
marks the intervention conceptually, while the actual intervention is applied at
the probe layer during model execution.

## Verification

The shared loader enforces the frozen 60-name data contract and recomputes the
means and transition counts from the canonical input table. Ruff, the repository
whitespace check, PDF text extraction, and embedded-font checks passed. A 180-dpi
Poppler render of the comparison PDF showed no clipping or label overlap. The
manuscript remains unchanged while the visual grammar is under selection.
