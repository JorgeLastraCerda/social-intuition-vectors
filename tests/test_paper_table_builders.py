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
MANUSCRIPT = ROOT / "paper" / "paper" / "Ulu_Lastra.tex"
MEDIATION_TABLE_BUILDER = ROOT / "src" / "build_paper_mediation_table.py"


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


def test_specificity_table_bolds_every_row_maximum() -> None:
    text = (TABLES / "concept_direction_specificity.tex").read_text()
    bold_effects = re.findall(r"\\textbf\{([+-]\d+\.\d{2})\}", text)
    assert bold_effects == ["+3.88", "+3.69", "+2.86", "+4.36"]

    data_rows = [line for line in text.splitlines() if re.match(r"G3-(?:12|27)B &", line)]
    assert len(data_rows) == 4
    assert all(row.count(r"\textbf{") == 1 for row in data_rows)


def test_broad_slope_r2_threshold_uses_raw_values_and_three_decimals() -> None:
    high_r2 = set()
    for label in MODEL_ORDER:
        df = pd.read_csv(TABLES / HIRING_STEERING_BROAD_CSV[label])
        for axis in ("warmth", "competence"):
            means = df[df["axis"] == axis].groupby("strength")["delta"].mean().sort_index()
            _, _, r2 = _ols_slope_r2(list(means.index.astype(float)), list(means.values))
            if r2 >= 0.8:
                high_r2.add((label, axis))

    assert high_r2 == {
        ("gemma3_12b", "warmth"),
        ("llama31_8b", "warmth"),
        ("llama31_8b", "competence"),
        ("gemma4_26b_a4b", "warmth"),
        ("qwen36_27b", "warmth"),
        ("qwen36_27b", "competence"),
        ("qwen36_35b_a3b", "warmth"),
    }
    text = (TABLES / "hiring_steering_slopes_9model.tex").read_text()
    qwen_warmth = next(line for line in text.splitlines() if line.startswith("Q3-14B & Warmth"))
    assert " & 0.799 & " in qwen_warmth
    assert r"\textbf{0.799}" not in qwen_warmth
    assert r"R^2\ge0.800" in text


def test_warmth_human_alignment_has_three_significant_negatives() -> None:
    significant_negative = set()
    for label in MODEL_ORDER:
        report = json.loads((LOGS / f"hiring_probe_vs_human_{label}.json").read_text())
        row = next(item for item in report["correlations"] if item["pair"] == "warmth")
        if float(row["spearman_rho"]) < 0 and float(row["spearman_p"]) < 0.05:
            significant_negative.add(label)

    assert significant_negative == {"llama31_8b", "gemma4_26b_a4b", "qwen3_14b"}
    gemma4_12b = json.loads((LOGS / "hiring_probe_vs_human_gemma4_12b.json").read_text())
    warmth = next(item for item in gemma4_12b["correlations"] if item["pair"] == "warmth")
    assert float(warmth["spearman_rho"]) == pytest.approx(0.009, abs=0.001)
    assert float(warmth["spearman_p"]) > 0.05


def test_mediation_language_matches_available_unadjusted_intervals() -> None:
    n_excluding_zero = 0
    n_total = 0
    for label in MODEL_ORDER:
        report = json.loads((LOGS / f"hiring_mediation_{label}.json").read_text())
        n_total += len(report["mediation"])
        n_excluding_zero += sum(bool(row["significant_95"]) for row in report["mediation"])
    assert (n_excluding_zero, n_total) == (14, 36)

    active_text = "\n".join(
        path.read_text()
        for path in (MANUSCRIPT, MEDIATION_TABLE_BUILDER, TABLES / "mediation_9model.tex")
    )
    assert "Bonferroni" not in active_text
    assert "survives correction" not in active_text
    generated = (TABLES / "mediation_9model.tex").read_text()
    assert "14 of 36 intervals exclude zero at this unadjusted threshold" in generated
    assert "No multiplicity-adjusted inference is reported" in generated
    assert "exploratory" in generated


def test_manuscript_closeout_wording_is_arithmetically_consistent() -> None:
    text = " ".join(MANUSCRIPT.read_text().split())
    assert "summarizes three of the five single-model checks in four reported statistics" in text
    assert "collects four of the five single-model checks" not in text
    assert text.count(
        "1,500 length-matched introductory passages drawn from distinct Wikipedia articles"
    ) == 3
    assert "both Qwen3.6-27B axes; and Qwen3.6-35B-A3B warmth" in text
    assert "Significant negative name-level warmth alignment in three models" in text


def test_manuscript_human_alignment_counts_match_source_logs() -> None:
    counts = {"competence": {"sig_pos": 0, "other": 0}, "warmth": {"sig_pos": 0, "other": 0}}
    for label in MODEL_ORDER:
        report = json.loads((LOGS / f"hiring_probe_vs_human_{label}.json").read_text())
        for axis in counts:
            row = next(item for item in report["correlations"] if item["pair"] == axis)
            positive_and_significant = (
                float(row["spearman_rho"]) > 0 and float(row["spearman_p"]) < 0.05
            )
            counts[axis]["sig_pos" if positive_and_significant else "other"] += 1

    # Competence is positive and significant in 7 of 9; the other 2 are negative
    # and nonsignificant. Warmth is negative or nonsignificant in 4 of 9.
    assert counts["competence"] == {"sig_pos": 7, "other": 2}
    assert counts["warmth"] == {"sig_pos": 5, "other": 4}

    text = " ".join(MANUSCRIPT.read_text().split())
    assert "Competence tracks human ratings in seven of the nine checkpoints" in text
    assert (
        "positive and significant in seven of the nine, with the remaining two "
        "small, negative, and nonsignificant"
    ) in text
    assert "Competence tracks human ratings in every checkpoint" not in text
    assert "positive in all nine, reaching significance in all but one" not in text


def test_manuscript_avoids_unqualified_encoding_claims() -> None:
    text = " ".join(MANUSCRIPT.read_text().split())
    assert "appear to be encoded in all nine models" not in text
    assert "The structure of social perception appears to be present" not in text
    assert (
        "Warmth- and competence-associated contrastive directions are recoverable "
        "in all nine models"
    ) in text
    assert "A direction that separates our warmth and competence stories exists" in text


def _active_table_sources() -> list[tuple[Path, str]]:
    manuscript = MANUSCRIPT.read_text()
    sources = [(MANUSCRIPT, manuscript)]
    included = re.findall(r"\\input\{\\tabledir/([^}]+\.tex)\}", manuscript)
    sources.extend((TABLES / name, (TABLES / name).read_text()) for name in included)
    return sources


def test_every_active_table_caption_is_below_its_table() -> None:
    table_blocks: list[tuple[Path, str]] = []
    longtable_blocks: list[tuple[Path, str]] = []
    for path, text in _active_table_sources():
        table_blocks.extend(
            (path, match.group(0))
            for match in re.finditer(
                r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, flags=re.DOTALL
            )
        )
        longtable_blocks.extend(
            (path, match.group(0))
            for match in re.finditer(
                r"\\begin\{longtable\}.*?\\end\{longtable\}", text, flags=re.DOTALL
            )
        )

    assert len(table_blocks) + len(longtable_blocks) == 22
    for path, block in table_blocks:
        assert block.count(r"\caption{") == 1, path
        assert block.count(r"\label{tab:") == 1, path
        assert block.index(r"\caption{") > block.rindex(r"\end{tabular}"), path
        assert block.index(r"\label{tab:") > block.index(r"\caption{"), path

    for path, block in longtable_blocks:
        assert block.count(r"\caption{") == 1, path
        assert block.count(r"\label{tab:") == 1, path
        last_data_row = max(
            block.rfind(line)
            for line in block.splitlines()
            if line.rstrip().endswith(r"\\")
            and r"\caption{" not in line
            and r"\label{" not in line
        )
        assert block.index(r"\caption{") > last_data_row, path
        assert block.index(r"\label{tab:") > block.index(r"\caption{"), path


def test_active_table_labels_and_references_are_complete_and_unique() -> None:
    combined = "\n".join(text for _, text in _active_table_sources())
    labels = re.findall(r"\\label\{(tab:[^}]+)\}", combined)
    references = re.findall(r"\\(?:auto)?ref\{(tab:[^}]+)\}", combined)

    assert len(labels) == 22
    assert len(labels) == len(set(labels))
    assert set(references) <= set(labels)
    assert "tab:disparity_race_gender" in labels
    assert r"\begin{table}[H]" in (TABLES / "concept_saturation.tex").read_text()
    assert r"\autoref{tab:disparity_marginal} in the main text" not in combined
    assert "census reported in the main text" not in combined
