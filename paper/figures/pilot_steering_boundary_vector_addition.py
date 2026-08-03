"""Generate the vector-addition steering pilot."""

from _pilot_steering_flip_common import draw_boundary_vector_addition, run_single


if __name__ == "__main__":
    run_single(
        draw_boundary_vector_addition,
        "pilot_steering_boundary_vector_addition",
        "boundary vector addition",
    )
