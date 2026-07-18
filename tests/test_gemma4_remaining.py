from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.dense_steering import empirical_null_rows, orthogonal_random_directions
from src.summarize_hiring_steering import gate_reasons, summarize_frame
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "jobs/sge/submit_gemma4_remaining.sh"
RUNNER = ROOT / "jobs/sge/gemma4_remaining_run.sh"


def test_gemma4_configs_are_pinned_and_hardware_specific() -> None:
    expected = {
        "12b": ("12ace6d648d72bd41519e140f1185f34d38c7e3d", 48, 3840),
        "26b_a4b": ("01e5b3ee840d3a9e0b0b493c593e85398a30ef75", 30, 2816),
        "31b": ("b9ea41a2887d8607f594846523f94c6cc75ac8a4", 60, 5376),
    }
    for short, (revision, layers, width) in expected.items():
        cfg = load_config(ROOT / f"config/gemma4_{short}.yaml")
        assert cfg.model.name.startswith("google/gemma-4-")
        assert cfg.model.revision == revision
        assert cfg.model.backend == "transformer-bridge"
        assert (cfg.smoke.expected_layers, cfg.smoke.expected_d_model) == (
            layers,
            width,
        )
        assert cfg.probing.seed == 20260527
        assert cfg.neutral.n_texts == 1500


def test_random_controls_are_deterministic_unit_and_orthogonal_to_both_axes() -> None:
    warmth = np.array([1.0, 0.0, 1.0, 0.0])
    competence = np.array([0.0, 1.0, 1.0, 0.0])
    first = orthogonal_random_directions(warmth, competence, n_directions=50, seed=7)
    second = orthogonal_random_directions(warmth, competence, n_directions=50, seed=7)
    assert len(first) == 50
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
        np.testing.assert_allclose(np.linalg.norm(left), 1.0, atol=1e-6)
        np.testing.assert_allclose(left @ warmth, 0.0, atol=1e-6)
        np.testing.assert_allclose(left @ competence, 0.0, atol=1e-6)


def _hiring_frame(*, nonlinear: bool = False) -> pd.DataFrame:
    rows = []
    strengths = [-0.1, -0.05, 0.0, 0.05, 0.1]
    for axis in ("warmth", "competence"):
        for strength in strengths:
            effect = strength
            if nonlinear and strength == 0.1:
                effect = -0.1
            for index in range(60):
                rows.append(
                    {
                        "axis": axis,
                        "strength": strength,
                        "name": f"name-{index}",
                        "margin": index / 8,
                        "delta": effect,
                    }
                )
    return pd.DataFrame(rows)


def test_hiring_summary_is_seeded_and_preserves_legacy_grid_diagnostic() -> None:
    left, diagnostics_left = summarize_frame(_hiring_frame(), seed=20260527, n_boot=100)
    right, diagnostics_right = summarize_frame(
        _hiring_frame(), seed=20260527, n_boot=100
    )
    pd.testing.assert_frame_equal(left, right)
    assert diagnostics_left == diagnostics_right
    assert len(left) == 10
    assert diagnostics_left["margin_diagnostic"]["fraction_on_0.125_grid"] == 1.0
    for axis in ("warmth", "competence"):
        assert diagnostics_left["axes"][axis]["r_squared"] == 1.0
        assert diagnostics_left["axes"][axis]["monotone_non_decreasing"] is True


def test_conditional_282_gate_fires_for_nonmonotone_or_uncertain_runs() -> None:
    _, stable = summarize_frame(_hiring_frame(), seed=1, n_boot=100)
    _, unstable = summarize_frame(_hiring_frame(nonlinear=True), seed=1, n_boot=100)
    reasons = gate_reasons(
        {"local": unstable, "broad": stable, "denoised_local": stable}
    )
    assert any("non_monotone" in reason for reason in reasons)
    assert any("r_squared_below_0.50" in reason for reason in reasons)


def test_empirical_null_summary_uses_fifty_controls() -> None:
    rows = []
    for axis in ("warmth", "competence"):
        other = "competence" if axis == "warmth" else "warmth"
        for direction, effect in ((axis, 1.0), (other, 0.5)):
            for strength in (-0.1, 0.0, 0.1):
                rows.append(
                    {
                        "mode": "steering",
                        "axis": axis,
                        "direction": direction,
                        "strength": strength,
                        "effect": effect * strength,
                    }
                )
        for random_id in range(50):
            for strength in (-0.1, 0.0, 0.1):
                rows.append(
                    {
                        "mode": "steering",
                        "axis": axis,
                        "direction": f"random_{random_id:03d}",
                        "strength": strength,
                        "effect": 0.01 * strength,
                    }
                )
    null = empirical_null_rows(rows)
    assert len(null) == 4
    assert {row["n_random_directions"] for row in null} == {50}
    assert {row["empirical_two_sided_p"] for row in null} == {1 / 51}


def test_submitter_dry_runs_all_models_without_scheduler_chaining() -> None:
    text = SUBMITTER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "-hold_jid" not in text
    assert "git pull" not in runner
    assert "sync_outputs.sh" not in runner
    for model in ("12b", "26b_a4b", "31b"):
        env = {**os.environ, "RUN_ID": "test-run"}
        result = subprocess.run(
            ["bash", str(SUBMITTER), "--model", model, "--run", "smoke", "--dry-run"],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "independent=1" in result.stdout
        assert "user_held=1" in result.stdout
    result = subprocess.run(
        [
            "bash",
            str(SUBMITTER),
            "--model",
            "12b",
            "--run",
            "hiring_local",
            "--full282",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "full282=1" in result.stdout
