"""Generate the sharp counterfactual-kink steering pilot."""

from _pilot_steering_flip_common import draw_boundary_kink, run_single


if __name__ == "__main__":
    run_single(draw_boundary_kink, "pilot_steering_boundary_kink", "boundary kink")
