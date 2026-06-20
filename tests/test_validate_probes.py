import numpy as np

from src.validate_probes import projected_cv_accuracy


def test_projected_cv_accuracy_is_invariant_to_large_offsets_and_scales():
    rng = np.random.default_rng(20260527)
    high = rng.normal(loc=1.5, scale=0.5, size=(50, 1))
    low = rng.normal(loc=-1.5, scale=0.5, size=(50, 1))
    direction = np.array([1.0])

    baseline = projected_cv_accuracy(high, low, direction, seed=20260527)
    shifted_high = high * 80_000 + 60_000
    shifted_low = low * 80_000 + 60_000
    shifted = projected_cv_accuracy(
        shifted_high,
        shifted_low,
        direction,
        seed=20260527,
    )

    assert baseline == shifted
    assert baseline > 0.95
