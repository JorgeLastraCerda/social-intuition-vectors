"""Generate the aggregate split-flow steering pilot."""

from _pilot_steering_flip_common import draw_split_flow, run_single


if __name__ == "__main__":
    run_single(draw_split_flow, "pilot_steering_flip_split_flow", "aggregate split-flow")
