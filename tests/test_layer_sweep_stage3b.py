from __future__ import annotations

import numpy as np

from src.layer_sweep import (
    paired_topic_bootstrap_curves,
    sweep_metrics_at_layer,
    sweep_stage3b_metrics_at_layer,
)


CONDITIONS = (
    "high_warmth",
    "low_warmth",
    "high_competence",
    "low_competence",
)


def synthetic_activations() -> tuple[np.ndarray, dict[str, list[int]], dict[str, np.ndarray]]:
    rng = np.random.default_rng(7)
    n_layers, n_topics, d_model = 3, 10, 8
    buckets: dict[str, list[int]] = {}
    groups: dict[str, np.ndarray] = {}
    acts = np.zeros((n_layers, len(CONDITIONS) * n_topics, d_model), dtype=np.float32)
    for condition_index, condition in enumerate(CONDITIONS):
        start = condition_index * n_topics
        buckets[condition] = list(range(start, start + n_topics))
        groups[condition] = np.arange(n_topics, dtype=np.int64)
        values = rng.normal(0.0, 0.2, size=(n_layers, n_topics, d_model))
        if condition == "high_warmth":
            values[:, :, 0] += np.array([0.5, 1.0, 2.0])[:, None]
        elif condition == "low_warmth":
            values[:, :, 0] -= np.array([0.5, 1.0, 2.0])[:, None]
        elif condition == "high_competence":
            values[:, :, 1] += np.array([0.4, 0.8, 1.5])[:, None]
        else:
            values[:, :, 1] -= np.array([0.4, 0.8, 1.5])[:, None]
        acts[:, start:start + n_topics, :] = values
    return acts, buckets, groups


def test_stage3b_preserves_legacy_metrics_and_adds_five_fold_scores() -> None:
    acts, buckets, groups = synthetic_activations()
    legacy = sweep_metrics_at_layer(1, acts, buckets, groups, 3, 1)
    enhanced, folds = sweep_stage3b_metrics_at_layer(
        1, acts, buckets, groups, 3, 1, seed=20260527
    )
    for key, value in legacy.items():
        assert enhanced[key] == value
    assert set(folds) == {
        "warmth_direction_topic_cv",
        "comp_direction_topic_cv",
        "warmth_to_comp_topic_transfer",
        "comp_to_warmth_topic_transfer",
    }
    assert all(len(scores) == 5 for scores in folds.values())
    assert all(0.0 <= enhanced[key] <= 1.0 for key in folds)


def test_paired_topic_bootstrap_is_seeded_and_returns_layer_bands() -> None:
    acts, buckets, groups = synthetic_activations()
    kwargs = dict(
        n_bootstrap=40,
        seed=20260527,
        batch_size=7,
        device="cpu",
    )
    first = paired_topic_bootstrap_curves(acts, buckets, groups, **kwargs)
    second = paired_topic_bootstrap_curves(acts, buckets, groups, **kwargs)
    assert first == second
    assert first["n_topics"] == 10
    for metric in ("warmth_cohens_d", "comp_cohens_d", "cos_wc"):
        low = np.asarray(first["bands"][metric]["low"])
        high = np.asarray(first["bands"][metric]["high"])
        probabilities = np.asarray(first["peaks"][metric]["layer_probabilities"])
        assert low.shape == high.shape == (3,)
        assert np.all(low <= high)
        assert probabilities.shape == (3,)
        assert np.isclose(probabilities.sum(), 1.0)


def test_paired_topic_bootstrap_rejects_unmatched_topic_sets() -> None:
    acts, buckets, groups = synthetic_activations()
    groups["low_competence"] = groups["low_competence"] + 1
    try:
        paired_topic_bootstrap_curves(
            acts, buckets, groups,
            n_bootstrap=10, seed=1, batch_size=5, device="cpu",
        )
    except ValueError as exc:
        assert "identical topic sets" in str(exc)
    else:
        raise AssertionError("unmatched topic sets should fail")
