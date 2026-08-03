"""Generate the deflected causal-flow steering pilot."""

from _pilot_steering_flip_common import draw_deflection, run_single


if __name__ == "__main__":
    run_single(draw_deflection, "pilot_steering_flip_deflection", "deflected causal-flow")
