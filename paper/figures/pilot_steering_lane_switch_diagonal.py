"""Generate the diagonal lane-switch steering pilot."""

from _pilot_steering_flip_common import draw_lane_switch_diagonal, run_single


if __name__ == "__main__":
    run_single(
        draw_lane_switch_diagonal,
        "pilot_steering_lane_switch_diagonal",
        "diagonal lane switch",
    )
