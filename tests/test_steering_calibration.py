from __future__ import annotations

import numpy as np
import pytest
import torch
import subprocess
from pathlib import Path

from src.steering_calibration import (
    NormDiagnostics,
    calibrated_alpha,
    descriptive_null_metrics,
    directional_sd,
    intervene_tensor,
    paired_topic_difference_ci,
    standardized_shift,
)

ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "jobs/sge/submit_calibrated_steering_pilot.sh"
RUNNER = ROOT / "jobs/sge/calibrated_steering_run.sh"


def test_sd_matching_equalizes_standardized_shift() -> None:
    target_alpha = calibrated_alpha(
        strength=0.1,
        mean_residual_norm=100.0,
        target_direction_sd=2.0,
        direction_sd=2.0,
        control_scale="sd_matched",
    )
    random_alpha = calibrated_alpha(
        strength=0.1,
        mean_residual_norm=100.0,
        target_direction_sd=2.0,
        direction_sd=0.05,
        control_scale="sd_matched",
    )
    assert target_alpha == pytest.approx(10.0)
    assert random_alpha == pytest.approx(0.25)
    assert standardized_shift(target_alpha, 2.0) == pytest.approx(
        standardized_shift(random_alpha, 0.05)
    )


def test_target_alpha_reproduces_legacy_magnitude() -> None:
    for strength in (-0.1, 0.0, 0.1):
        assert calibrated_alpha(
            strength=strength,
            mean_residual_norm=73.0,
            target_direction_sd=1.7,
            direction_sd=1.7,
            control_scale="sd_matched",
        ) == pytest.approx(strength * 73.0)


def test_directional_sd_uses_sample_projection_sd() -> None:
    matrix = np.asarray([[1.0, 4.0], [2.0, 3.0], [4.0, 2.0]])
    assert directional_sd(matrix, np.asarray([1.0, 0.0])) == pytest.approx(
        np.std(matrix[:, 0], ddof=1)
    )


def test_norm_preserving_intervention_preserves_each_token_norm() -> None:
    residual = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])
    diagnostics = NormDiagnostics()
    changed = intervene_tensor(
        residual,
        torch.tensor([1.0, 0.0]),
        2.0,
        "norm_preserving",
        diagnostics,
    )
    assert torch.allclose(
        residual.norm(dim=-1), changed.norm(dim=-1), atol=1e-6, rtol=1e-6
    )
    assert diagnostics.max_relative_norm_drift < 1e-6


def test_descriptive_null_metrics_do_not_emit_pass_fail() -> None:
    metrics = descriptive_null_metrics(np.asarray([-2.0, 0.0, 1.0]), 0.5)
    assert set(metrics) == {
        "signed_percentile",
        "absolute_percentile",
        "random_median",
        "target_minus_random_median",
    }
    assert metrics["target_minus_random_median"] == pytest.approx(0.5)


def test_paired_topic_bootstrap_uses_within_topic_random_median() -> None:
    rows = []
    for topic, target in ((1, 2.0), (2, 4.0)):
        for direction, value in (
            ("warmth", target),
            ("random_000", 0.0),
            ("random_001", 2.0),
        ):
            rows.append(
                {
                    "mode": "steering",
                    "axis": "warmth",
                    "intervention": "additive",
                    "strength": 0.1,
                    "topic_idx": topic,
                    "direction": direction,
                    "delta_margin": value,
                }
            )
    estimate, low, high = paired_topic_difference_ci(
        rows,
        judgment_axis="warmth",
        steering_axis="warmth",
        intervention="additive",
        endpoint_strength=0.1,
        seed=42,
        n_boot=500,
    )
    assert estimate == pytest.approx(2.0)
    assert low <= estimate <= high


def test_pilot_jobs_are_independent_rtx6000_and_disable_full282() -> None:
    submitter = SUBMITTER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "-hold_jid" not in submitter
    assert "git pull" not in runner
    assert "sync_outputs.sh" not in runner
    assert 'if "RTX PRO 6000" not in name' in runner
    for model in ("gemma3_12b", "gemma4_12b", "qwen36_27b"):
        result = subprocess.run(
            ["bash", str(SUBMITTER), "--model", model, "--dry-run"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "independent=1" in result.stdout
        assert "user_held=1" in result.stdout
        assert "full282=disabled" in result.stdout
        assert "RTX_PRO_6000" in result.stdout
