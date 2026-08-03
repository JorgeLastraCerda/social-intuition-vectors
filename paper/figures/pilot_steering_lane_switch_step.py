"""Generate the right-angle lane-switch steering pilot."""

from _pilot_steering_flip_common import draw_lane_switch_step, run_single


if __name__ == "__main__":
    run_single(
        draw_lane_switch_step,
        "pilot_steering_lane_switch_step",
        "right-angle lane switch",
    )
