import numpy as np

from src.validate_probes import (
    direction_topic_holdout_cv,
    projected_cv_accuracy,
    topic_cross_axis_transfer_cv,
)


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


def test_direction_topic_holdout_rebuilds_a_generalizing_direction():
    rng = np.random.default_rng(7)
    groups = np.arange(20)
    high = rng.normal(loc=2.0, scale=0.2, size=(20, 3))
    low = rng.normal(loc=-2.0, scale=0.2, size=(20, 3))

    mean, std, folds = direction_topic_holdout_cv(
        high, low, groups, groups, seed=11, n_splits=5
    )

    assert mean == 1.0
    assert std == 0.0
    assert folds == [1.0] * 5


def test_topic_cross_axis_transfer_does_not_recalibrate_on_target():
    rng = np.random.default_rng(17)
    groups = np.arange(20)
    source_high = rng.normal(loc=2.0, scale=0.15, size=(20, 1))
    source_low = rng.normal(loc=-2.0, scale=0.15, size=(20, 1))
    # Same within-target separation, but shifted beyond the source decision boundary.
    target_high = rng.normal(loc=12.0, scale=0.15, size=(20, 1))
    target_low = rng.normal(loc=8.0, scale=0.15, size=(20, 1))

    calibrated = projected_cv_accuracy(target_high, target_low, np.array([1.0]), seed=11)
    transfer_mean, _, _ = topic_cross_axis_transfer_cv(
        source_high,
        source_low,
        target_high,
        target_low,
        groups,
        groups,
        groups,
        groups,
        seed=11,
        n_splits=5,
    )

    assert calibrated == 1.0
    assert transfer_mean == 0.5
