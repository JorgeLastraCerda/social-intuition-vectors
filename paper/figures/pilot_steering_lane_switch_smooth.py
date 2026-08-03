"""Generate the smooth lane-switch steering pilot."""

from _pilot_steering_flip_common import draw_lane_switch_smooth, run_single


if __name__ == "__main__":
    run_single(
        draw_lane_switch_smooth,
        "pilot_steering_lane_switch_smooth",
        "smooth lane switch",
    )
