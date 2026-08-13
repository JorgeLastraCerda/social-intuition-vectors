from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from src.build_paper_probe_tables import (
    CALIBRATED_LABELS,
    DISPLAY_NAME,
    HIRING_STEERING_BROAD_CSV,
    MODEL_ORDER,
    STEERING_DENSE_CSV,
    _ols_slope_r2,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def test_disparity_gap_table_has_declared_units_and_nine_models() -> None:
    text = (TABLES / "hiring_disparity_gaps_9model.tex").read_text()
    assert "Model (SD)" in text
    assert "Human (pp)" in text
    assert "\\label{tab:disparity_gaps}" in text
    data_rows = [line for line in text.splitlines() if re.match(r"(?:G|L|Q)[0-9]", line)]
    assert len(data_rows) == 9
    assert all("+1.2" in line and "-3.6" in line for line in data_rows)


def test_signal_control_verdicts_match_source_intervals() -> None:
    verdicts: dict[tuple[str, str], bool] = {}
    for label in MODEL_ORDER:
        df = pd.read_csv(TABLES / STEERING_DENSE_CSV[label])
        for axis in ("warmth", "competence"):
            if label in CALIBRATED_LABELS:
                sub = df[(df["axis"] == axis) & (df["intervention"] == "additive")]
                target = float(sub[(sub["direction"] == axis) & (sub["strength"] == 0.1)]["effect"].iloc[0])
                random = sub[sub["direction"].str.startswith("random_") & (sub["strength"] == 0.1)]["effect"]
                mean, radius = float(random.mean()), float(1.96 * random.std(ddof=1))
                lo, hi = mean - radius, mean + radius
            else:
                sub = df[(df["mode"] == "steering") & (df["axis"] == axis) & (df["strength"] == 0.1)]
                target = float(sub[sub["direction"] == "raw_dense"]["effect"].iloc[0])
                row = sub[sub["direction"] == "random"].iloc[0]
                lo, hi = float(row["ci_low"]), float(row["ci_high"])
            verdicts[(label, axis)] = target < lo or target > hi
    assert sum(verdicts.values()) == 12
    assert not any(verdicts[(label, axis)] for label in ("gemma4_12b", "gemma4_26b_a4b", "gemma4_31b") for axis in ("warmth", "competence"))


def test_broad_slope_endpoint_sign_disagreements_are_exact() -> None:
    found = set()
    for label in MODEL_ORDER:
        df = pd.read_csv(TABLES / HIRING_STEERING_BROAD_CSV[label])
        for axis in ("warmth", "competence"):
            means = df[df["axis"] == axis].groupby("strength")["delta"].mean().sort_index()
            slope, _, _ = _ols_slope_r2(list(means.index.astype(float)), list(means.values))
            if slope * float(means.loc[0.5]) < 0:
                found.add((label, axis))
    assert found == {
        ("gemma3_27b", "warmth"), ("gemma3_27b", "competence"),
        ("gemma4_12b", "warmth"), ("gemma4_12b", "competence"),
        ("gemma4_26b_a4b", "competence"),
        ("gemma4_31b", "warmth"), ("gemma4_31b", "competence"),
        ("qwen36_35b_a3b", "competence"),
    }


def test_local_transition_summary_uses_point_one_for_all_models() -> None:
    summary = pd.read_csv(TABLES / "hiring_steering_transition_summary_9model.csv")
    assert len(summary) == 18
    assert set(summary["strength"]) == {0.1}
    assert set(summary["n_names"]) == {60}
    llama = summary[summary["model"] == "Llama-3.1-8B"]
    qwen = summary[summary["model"] == "Qwen3-14B"]
    assert set(llama["baseline_decision"]) == {"No"}
    assert set(llama["steered_decision"]) == {"No"}
    assert set(qwen["baseline_decision"]) == {"Yes"}
    assert set(qwen["steered_decision"]) == {"Yes"}
    assert qwen.set_index("axis").loc["competence", "delta_mean"] == 0.010416666667


def test_appendix_pivots_do_not_repeat_human_benchmark_per_model() -> None:
    marginal = (TABLES / "hiring_disparity_marginal_9model.tex").read_text()
    crossed = (TABLES / "hiring_disparity_race_gender_9model.tex").read_text()
    assert marginal.count("Human callback") == 1
    assert crossed.count("Human callback") == 1
    for label in MODEL_ORDER:
        assert marginal.count(DISPLAY_NAME[label]) == 2
        assert crossed.count(DISPLAY_NAME[label]) == 2
