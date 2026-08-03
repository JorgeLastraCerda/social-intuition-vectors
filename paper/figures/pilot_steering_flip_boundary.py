"""Generate the decision-boundary steering pilot."""

from _pilot_steering_flip_common import draw_boundary, run_single


if __name__ == "__main__":
    run_single(draw_boundary, "pilot_steering_flip_boundary", "decision-boundary")
