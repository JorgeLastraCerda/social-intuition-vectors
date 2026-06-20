import numpy as np
import torch

from src.gemma_scope_causality import (
    candidate_token_id,
    make_error_preserving_ablation_hook,
    summarize_baseline,
    yes_no_margin,
)
from src.gemma_scope_utils import (
    decompose_feature_axes,
    smallest_energy_feature_set,
)
from src.match_gemma_scope_features import normalized_profiles


class FakeModel:
    def to_tokens(self, text, prepend_bos):
        if text == " Yes":
            return torch.tensor([[3]])
        if text == " No":
            return torch.tensor([[4]])
        return torch.tensor([[1, 2]])

    def run_with_hooks(self, tokens, **kwargs):
        logits = torch.zeros(1, tokens.shape[1], 6)
        logits[0, -1, 3] = 2.5
        logits[0, -1, 4] = -0.5
        return logits


class IdentitySAE:
    W_dec = torch.eye(3)

    def encode(self, values):
        return values

    def decode(self, values):
        return values


def test_feature_axis_decomposition_is_exact():
    warmth = np.array([2.0, -3.0, 1.0, 0.0])
    competence = np.array([1.0, -5.0, -2.0, 0.0])
    vectors = decompose_feature_axes(warmth, competence)

    np.testing.assert_allclose(
        vectors["shared"] + vectors["warmth_specific"],
        warmth,
    )
    np.testing.assert_allclose(
        vectors["shared"] + vectors["competence_specific"],
        competence,
    )
    np.testing.assert_allclose(vectors["shared"], [1.0, -3.0, 0.0, 0.0])


def test_smallest_energy_feature_set_reaches_requested_fraction():
    selected = smallest_energy_feature_set(np.array([3.0, 2.0, 1.0]), 0.75)
    np.testing.assert_array_equal(selected, np.array([0, 1]))


def test_yes_no_margin_uses_single_next_token_forward():
    model = FakeModel()
    assert candidate_token_id(model, " Yes") == 3
    assert yes_no_margin(model, "prompt", "blocks.0.hook_resid_post") == 3.0


def test_error_preserving_ablation_removes_selected_linear_feature():
    hook = make_error_preserving_ablation_hook(
        IdentitySAE(),
        np.array([1], dtype=np.int64),
    )
    residual = torch.tensor([[[1.0, 2.0, 3.0]]])
    result = hook(residual, None)
    torch.testing.assert_close(result, torch.tensor([[[1.0, 0.0, 3.0]]]))


def test_baseline_summary_reports_accuracy_and_paired_gap():
    rows = []
    for topic in (0, 1):
        rows.extend(
            [
                {
                    "mode": "baseline",
                    "axis": "warmth",
                    "story_id": f"high-{topic}",
                    "topic_idx": topic,
                    "label": 1,
                    "margin": 2.0,
                },
                {
                    "mode": "baseline",
                    "axis": "warmth",
                    "story_id": f"low-{topic}",
                    "topic_idx": topic,
                    "label": 0,
                    "margin": -1.0,
                },
            ]
        )
    rows.extend(
        {**row, "axis": "competence", "story_id": f"c-{row['story_id']}"}
        for row in list(rows)
    )
    summary = summarize_baseline(rows, seed=7)
    accuracy = next(
        row
        for row in summary
        if row["axis"] == "warmth" and row["direction"] == "accuracy"
    )
    gap = next(
        row
        for row in summary
        if row["axis"] == "warmth"
        and row["direction"] == "high_low_margin_gap"
    )
    assert accuracy["effect"] == 1.0
    assert gap["effect"] == 3.0


def test_normalized_profiles_center_each_feature():
    from scipy import sparse

    matrix = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [2.0, 1.0],
                [3.0, 0.0],
            ]
        )
    )
    profiles = normalized_profiles(matrix, np.array([0, 1]))
    np.testing.assert_allclose(profiles.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(profiles, axis=1), 1.0)
