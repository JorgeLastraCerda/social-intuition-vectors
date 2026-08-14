from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.build_paper_probe_tables import (
    CALIBRATED_LABELS,
    DISPLAY_NAME,
    HIRING_STEERING_BROAD_CSV,
    MODEL_ORDER,
    STEERING_DENSE_CSV,
    _ols_slope_r2,
    _pooled_standardized_mean_difference,
)
from src.hiring_disparity import load_callback_disparity_frame
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"


def test_disparity_gap_table_has_declared_units_and_nine_models() -> None:
    text = (TABLES / "hiring_disparity_gaps_9model.tex").read_text()
    assert "Model $d$" in text
    assert "Human $d$" in text
    assert "pooled within-group SD" in text
    assert "\\label{tab:disparity_gaps}" in text
    data_rows = [line for line in text.splitlines() if re.match(r"(?:G|L|Q)[0-9]", line)]
    assert len(data_rows) == 9
    assert all("+0.15" in line and "-0.47" in line for line in data_rows)


def test_disparity_gaps_recompute_from_the_canonical_matched_population() -> None:
    cfg = load_config(ROOT / "config" / "config.yaml")
    merged = load_callback_disparity_frame(
        TABLES / "hiring_audit_gemma3_27b.csv", Path(cfg.paths.raw_data)
    )
    assert len(merged.loc[merged["race"] == "Black"]) == 47
    assert len(merged.loc[merged["race"] == "White"]) == 180
    assert len(merged.loc[merged["gender"] == "Female"]) == 154
    assert len(merged.loc[merged["gender"] == "Male"]) == 115

    def independent_d(value: str, group: str, positive: str, negative: str) -> float:
        x = merged.loc[merged[group] == positive, value].dropna().astype(float)
        y = merged.loc[merged[group] == negative, value].dropna().astype(float)
        pooled = (
            ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1))
            / (len(x) + len(y) - 2)
        ) ** 0.5
        return float((x.mean() - y.mean()) / pooled)

    assert independent_d("human_callback", "race", "Black", "White") == pytest.approx(0.152, abs=0.001)
    assert independent_d("human_callback", "gender", "Female", "Male") == pytest.approx(-0.474, abs=0.001)
    assert independent_d("callback_margin", "race", "Black", "White") == pytest.approx(1.441, abs=0.001)
    assert independent_d("callback_margin", "gender", "Female", "Male") == pytest.approx(-0.457, abs=0.001)


def test_pooled_standardized_gap_rejects_zero_variance() -> None:
    frame = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [1.0, 1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="pooled within-group SD"):
        _pooled_standardized_mean_difference(frame, "value", "group", "A", "B")


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


def test_ablation_caption_reports_scale_specific_shared_feature_necessity() -> None:
    text = (TABLES / "gemma_scope_ablation.tex").read_text()
    assert "only in Gemma-3-27B" in text
    assert "increases both gaps in Gemma-3-12B" in text
    assert "three of the four rows" not in text


def test_callback_precision_summary_matches_canonical_282_name_audits() -> None:
    expected = {
        "gemma3_12b": (0.15, 8),
        "llama31_8b": (0.12, 12),
        "gemma3_27b": (0.43, 20),
        "qwen3_14b": (0.35, 17),
    }
    for label, (expected_sd, expected_unique) in expected.items():
        margins = pd.read_csv(TABLES / f"hiring_audit_{label}.csv")["callback_margin"]
        assert len(margins) == 282
        assert round(float(margins.std(ddof=1)), 2) == expected_sd
        assert margins.nunique() == expected_unique

    gemma3 = pd.read_csv(TABLES / "hiring_audit_gemma3_12b.csv")["callback_margin"]
    gemma4 = pd.read_csv(TABLES / "hiring_audit_gemma4_12b.csv")["callback_margin"]
    assert ((gemma3 / 0.125).round() - gemma3 / 0.125).abs().max() < 1e-9
    assert ((gemma4 / 0.125).round() - gemma4 / 0.125).abs().max() > 1e-6


def test_gemma4_competence_endpoints_round_slightly_below_zero() -> None:
    frame = pd.read_csv(TABLES / "concept_steerability_normalized_9model.csv")
    endpoints = frame[
        (frame["axis"] == "competence")
        & (frame["strength"] == 0.1)
        & frame["model"].isin(["Gemma-4-26B-A4B", "Gemma-4-31B"])
    ].set_index("model")["normalized_steerability"]
    assert round(float(endpoints["Gemma-4-26B-A4B"]), 3) == -0.002
    assert round(float(endpoints["Gemma-4-31B"]), 3) == -0.001


def test_competence_human_alignment_has_two_small_nonsignificant_negatives() -> None:
    negative = {}
    positive = {}
    for label in MODEL_ORDER:
        report = json.loads((LOGS / f"hiring_probe_vs_human_{label}.json").read_text())
        row = next(item for item in report["correlations"] if item["pair"] == "competence")
        target = negative if row["spearman_rho"] < 0 else positive
        target[label] = (float(row["spearman_rho"]), float(row["spearman_p"]))

    assert set(negative) == {"llama31_8b", "gemma4_26b_a4b"}
    assert all(abs(rho) < 0.06 and p > 0.05 for rho, p in negative.values())
    assert len(positive) == 7
    assert all(rho > 0 and p < 0.05 for rho, p in positive.values())
