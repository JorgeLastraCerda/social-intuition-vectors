"""Generate the decision-boundary hinge steering pilot."""

from _pilot_steering_flip_common import draw_boundary_hinge, run_single


if __name__ == "__main__":
    run_single(draw_boundary_hinge, "pilot_steering_boundary_hinge", "boundary hinge")
