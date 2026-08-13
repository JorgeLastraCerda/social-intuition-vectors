"""Shared data contract and drawing code for nine-model steering transitions."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath

from style import apply as apply_paper_style


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO_ROOT / "results/tables"
OUT_DIR = REPO_ROOT / "paper/figures"
SUMMARY_PATH = TABLE_DIR / "hiring_steering_transition_summary_9model.csv"
SUMMARY_TEX_PATH = TABLE_DIR / "hiring_steering_transition_summary_9model.tex"

CHARCOAL = "#2F343B"
MUTED = "#6B7280"
LIGHT_GRAY = "#D7DCE2"
WHITE = "#FFFFFF"
STEERED_BLUE = "#0072B2"
WARMTH_GATE = "#D55E00"
COMPETENCE_GATE = "#7D6608"
TITLE_SUFFIX = "#262626"

FAMILY_COLORS = {
    "Gemma-3": "#003B73",
    "Gemma-4": "#0077B6",
    "Llama-3.1": "#C74600",
    "Qwen3": "#5E2A84",
    "Qwen3.6": "#B01872",
}

TRANSITIONS = tuple(
    (source, target)
    for source in ("No", "Tie", "Yes")
    for target in ("No", "Tie", "Yes")
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    family: str
    filename: str
    strength: float


MODEL_ROWS = (
    (
        ModelSpec("Gemma-3-12B", "Gemma-3", "hiring_steering_raw_concept_vectors.csv", 0.10),
        ModelSpec(
            "Gemma-3-27B",
            "Gemma-3",
            "hiring_steering_raw_concept_vectors_gemma3_27b.csv",
            0.10,
        ),
        ModelSpec("Llama-3.1-8B", "Llama-3.1", "hiring_steering_raw_llama31_8b_local.csv", 0.10),
    ),
    (
        ModelSpec("Gemma-4-12B", "Gemma-4", "hiring_steering_raw_gemma4_12b_local.csv", 0.10),
        ModelSpec(
            "Gemma-4-26B-A4B",
            "Gemma-4",
            "hiring_steering_raw_gemma4_26b_a4b_local.csv",
            0.10,
        ),
        ModelSpec("Gemma-4-31B", "Gemma-4", "hiring_steering_raw_gemma4_31b_local.csv", 0.10),
    ),
    (
        ModelSpec("Qwen3-14B", "Qwen3", "hiring_steering_raw_qwen3_14b_local.csv", 0.10),
        ModelSpec("Qwen3.6-27B", "Qwen3.6", "hiring_steering_raw_qwen36_27b_local.csv", 0.10),
        ModelSpec(
            "Qwen3.6-35B-A3B",
            "Qwen3.6",
            "hiring_steering_raw_qwen36_35b_a3b_local.csv",
            0.10,
        ),
    ),
)
MODEL_SPECS = tuple(spec for row in MODEL_ROWS for spec in row)


@dataclass(frozen=True)
class TransitionSummary:
    spec: ModelSpec
    axis: str
    n_names: int
    baseline_mean: float
    steered_mean: float
    transitions: Counter[tuple[str, str]]

    @property
    def delta(self) -> float:
        return self.steered_mean - self.baseline_mean

    @property
    def baseline_decision(self) -> str:
        return classify(self.baseline_mean)

    @property
    def steered_decision(self) -> str:
        return classify(self.steered_mean)


EXPECTED_MEANS = {
    ("Gemma-3-12B", "warmth"): (-0.19375, 2.175),
    ("Gemma-3-12B", "competence"): (-0.19375, 0.0375),
    ("Gemma-3-27B", "warmth"): (1.16875, -1.4895833333333333),
    ("Gemma-3-27B", "competence"): (1.16875, -1.5270833333333333),
    ("Llama-3.1-8B", "warmth"): (-2.0635416666666666, -1.5947916666666666),
    ("Llama-3.1-8B", "competence"): (-2.0635416666666666, -1.7385416666666667),
    ("Gemma-4-12B", "warmth"): (17.187886555989582, 18.236637369791666),
    ("Gemma-4-12B", "competence"): (17.187886555989582, 18.7146484375),
    ("Gemma-4-26B-A4B", "warmth"): (21.244791666666668, 21.307291666666668),
    ("Gemma-4-26B-A4B", "competence"): (21.244791666666668, 20.864583333333332),
    ("Gemma-4-31B", "warmth"): (25.614884440104166, 25.173177083333332),
    ("Gemma-4-31B", "competence"): (25.614884440104166, 24.800651041666665),
    ("Qwen3-14B", "warmth"): (1.6729166666666666, 2.095833333333333),
    ("Qwen3-14B", "competence"): (1.6729166666666666, 1.6833333333333333),
    ("Qwen3.6-27B", "warmth"): (5.345833333333333, 6.541666666666667),
    ("Qwen3.6-27B", "competence"): (5.345833333333333, 5.879166666666666),
    ("Qwen3.6-35B-A3B", "warmth"): (3.03125, 3.997916666666667),
    ("Qwen3.6-35B-A3B", "competence"): (3.03125, 3.464583333333333),
}


def _expected_counts() -> dict[tuple[str, str], Counter[tuple[str, str]]]:
    expected: dict[tuple[str, str], Counter[tuple[str, str]]] = {}
    for axis in ("warmth", "competence"):
        expected[("Gemma-3-27B", axis)] = Counter({("Yes", "No"): 60})
    expected[("Gemma-3-12B", "warmth")] = Counter(
        {("No", "Yes"): 54, ("Tie", "Yes"): 6}
    )
    expected[("Gemma-3-12B", "competence")] = Counter(
        {
            ("No", "No"): 19,
            ("No", "Tie"): 13,
            ("No", "Yes"): 22,
            ("Tie", "No"): 1,
            ("Tie", "Tie"): 2,
            ("Tie", "Yes"): 3,
        }
    )
    expected[("Llama-3.1-8B", "warmth")] = Counter({("No", "No"): 60})
    expected[("Llama-3.1-8B", "competence")] = Counter({("No", "No"): 60})
    unchanged_models = (
        "Gemma-4-12B",
        "Gemma-4-26B-A4B",
        "Gemma-4-31B",
        "Qwen3-14B",
        "Qwen3.6-27B",
        "Qwen3.6-35B-A3B",
    )
    for model in unchanged_models:
        for axis in ("warmth", "competence"):
            expected[(model, axis)] = Counter({("Yes", "Yes"): 60})
    return expected


EXPECTED_COUNTS = _expected_counts()


def configure_style() -> None:
    apply_paper_style()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def classify(value: float) -> str:
    if value > 0:
        return "Yes"
    if value < 0:
        return "No"
    return "Tie"


def load_summary(spec: ModelSpec, axis: str) -> TransitionSummary:
    path = TABLE_DIR / spec.filename
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"axis", "strength", "name", "margin"}
    if not rows or not required.issubset(rows[0]):
        missing = required.difference(rows[0] if rows else {})
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    selected = [row for row in rows if row["axis"] == axis]
    baseline: dict[str, float] = {}
    steered: dict[str, float] = {}
    for row in selected:
        strength = float(row["strength"])
        target = None
        if math.isclose(strength, 0.0, abs_tol=1e-12):
            target = baseline
        elif math.isclose(strength, spec.strength, abs_tol=1e-12):
            target = steered
        if target is None:
            continue
        name = row["name"]
        if name in target:
            raise ValueError(f"{path}: duplicate {axis} row for {name!r} at {strength:+.2f}")
        target[name] = float(row["margin"])

    if len(baseline) != 60 or len(steered) != 60:
        raise ValueError(
            f"{path}: expected 60 names at baseline and {spec.strength:+.2f}; "
            f"found {len(baseline)} and {len(steered)}"
        )
    if baseline.keys() != steered.keys():
        raise ValueError(f"{path}: baseline and steered name sets differ")

    transitions = Counter(
        (classify(baseline[name]), classify(steered[name])) for name in baseline
    )
    summary = TransitionSummary(
        spec=spec,
        axis=axis,
        n_names=len(baseline),
        baseline_mean=sum(baseline.values()) / len(baseline),
        steered_mean=sum(steered.values()) / len(steered),
        transitions=transitions,
    )
    _validate_expected(summary)
    return summary


def _validate_expected(summary: TransitionSummary) -> None:
    key = (summary.spec.label, summary.axis)
    expected_baseline, expected_steered = EXPECTED_MEANS[key]
    if not math.isclose(summary.baseline_mean, expected_baseline, abs_tol=1e-12):
        raise ValueError(f"{key}: unexpected baseline mean {summary.baseline_mean}")
    if not math.isclose(summary.steered_mean, expected_steered, abs_tol=1e-12):
        raise ValueError(f"{key}: unexpected steered mean {summary.steered_mean}")
    if summary.transitions != EXPECTED_COUNTS[key]:
        raise ValueError(f"{key}: unexpected transitions {dict(summary.transitions)}")
    if sum(summary.transitions.values()) != summary.n_names:
        raise ValueError(f"{key}: transition counts do not sum to {summary.n_names}")


def build_summaries() -> list[TransitionSummary]:
    return [
        load_summary(spec, axis)
        for spec in MODEL_SPECS
        for axis in ("warmth", "competence")
    ]


def write_summary_csv(summaries: list[TransitionSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transition_columns = [f"{source.lower()}_to_{target.lower()}" for source, target in TRANSITIONS]
    fieldnames = [
        "model",
        "family",
        "axis",
        "strength",
        "n_names",
        "baseline_mean",
        "steered_mean",
        "delta_mean",
        "baseline_decision",
        "steered_decision",
        *transition_columns,
        "input_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            row: dict[str, str | int | float] = {
                "model": summary.spec.label,
                "family": summary.spec.family,
                "axis": summary.axis,
                "strength": f"{summary.spec.strength:.2f}",
                "n_names": summary.n_names,
                "baseline_mean": f"{summary.baseline_mean:.12f}",
                "steered_mean": f"{summary.steered_mean:.12f}",
                "delta_mean": f"{summary.delta:.12f}",
                "baseline_decision": summary.baseline_decision,
                "steered_decision": summary.steered_decision,
                "input_path": f"results/tables/{summary.spec.filename}",
            }
            row.update(
                {
                    column: summary.transitions[(source, target)]
                    for column, (source, target) in zip(transition_columns, TRANSITIONS)
                }
            )
            writer.writerow(row)


def _decision_code(decision: str) -> str:
    return {"No": "N", "Tie": "T", "Yes": "Y"}[decision]


def _margin_with_decision(value: float) -> str:
    return f"{value:+.3f} ({_decision_code(classify(value))})"


def _transition_table_cell(summary: TransitionSummary) -> str:
    parts = [
        f"{summary.transitions[(source, target)]} {source[0]}$\\to${target[0]}"
        for source, target in TRANSITIONS
        if summary.transitions[(source, target)]
    ]
    if len(parts) <= 3:
        return "; ".join(parts)
    midpoint = math.ceil(len(parts) / 2)
    first = ", ".join(parts[:midpoint])
    second = ", ".join(parts[midpoint:])
    return f"\\shortstack{{{first}\\\\{second}}}"


def write_summary_tex(summaries: list[TransitionSummary], path: Path) -> None:
    by_key = {(summary.spec.label, summary.axis): summary for summary in summaries}
    lines = [
        "% Generated by paper_figure4_hiring_bidirectional_examples.py.",
        "% Source: results/tables/hiring_steering_transition_summary_9model.csv.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{\\textbf{Complete positive-endpoint callback transitions.} "
        "Mean baseline and steered callback margins are followed by their decision "
        "category in parentheses. Transition cells list every nonzero individual-name "
        "transition among No (N), Tie (T), and Yes (Y); each cell sums to 60.}",
        "\\label{tab:hiring_transition_census}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{tabular}{@{}lcccccccc@{}}",
        "\\toprule",
        "& & & \\multicolumn{3}{c}{Warmth} & \\multicolumn{3}{c}{Competence} \\\\",
        "\\cmidrule(lr){4-6} \\cmidrule(l){7-9}",
        "Model & $\\alpha$ & Baseline & Endpoint & $\\Delta$ margin & Transitions "
        "& Endpoint & $\\Delta$ margin & Transitions \\\\",
        "\\midrule",
    ]
    for spec in MODEL_SPECS:
        warmth = by_key[(spec.label, "warmth")]
        competence = by_key[(spec.label, "competence")]
        if not math.isclose(warmth.baseline_mean, competence.baseline_mean, abs_tol=1e-12):
            raise ValueError(f"{spec.label}: warmth and competence baselines differ")
        lines.append(
            " & ".join(
                (
                    spec.label,
                    f"{spec.strength:+.2f}",
                    _margin_with_decision(warmth.baseline_mean),
                    _margin_with_decision(warmth.steered_mean),
                    f"{warmth.delta:+.3f}",
                    _transition_table_cell(warmth),
                    _margin_with_decision(competence.steered_mean),
                    f"{competence.delta:+.3f}",
                    _transition_table_cell(competence),
                )
            )
            + " \\\\"
        )
    lines.extend(
        (
            "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {path}")


def _add_family_title(ax: Axes, spec: ModelSpec) -> None:
    suffix = spec.label[len(spec.family) :]
    family_text = TextArea(
        spec.family,
        textprops={"color": FAMILY_COLORS[spec.family], "fontsize": 8.4},
    )
    suffix_text = TextArea(
        suffix,
        textprops={"color": TITLE_SUFFIX, "fontsize": 8.4},
    )
    title_box = HPacker(children=[family_text, suffix_text], align="baseline", pad=0, sep=0)
    ax.add_artist(
        AnchoredOffsetbox(
            loc="lower center",
            child=title_box,
            bbox_to_anchor=(0.5, 1.015),
            bbox_transform=ax.transAxes,
            frameon=False,
            borderpad=0,
            pad=0,
        )
    )


def _smooth_connector(
    ax: Axes,
    start: tuple[float, float],
    join: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    dashed: bool,
) -> None:
    sx, sy = start
    jx, jy = join
    path = MplPath(
        [(sx, sy), (sx + 0.07, sy), (jx - 0.07, jy), (jx, jy)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    linestyle = (0, (4, 3)) if dashed else "solid"
    linewidth = 1.55 if dashed else 2.25
    zorder = 3 if dashed else 5
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            capstyle="round",
            zorder=zorder,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            join,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            zorder=zorder,
        )
    )


def _transition_annotation(summary: TransitionSummary) -> str:
    strict = [
        (source, target, summary.transitions[(source, target)])
        for source, target in (("No", "Yes"), ("Yes", "No"))
        if summary.transitions[(source, target)]
    ]
    unchanged = sum(summary.transitions[(label, label)] for label in ("No", "Tie", "Yes"))
    tie_involved = sum(
        count
        for (source, target), count in summary.transitions.items()
        if source != target and "Tie" in (source, target)
    )
    parts = [f"{count} {source[0]}→{target[0]}" for source, target, count in strict]
    if unchanged:
        parts.append(f"{unchanged} unchanged")
    if tie_involved:
        parts.append(f"{tie_involved} tie-involved")
    return " · ".join(parts)


def _decision_y(decision: str) -> float:
    if decision == "Yes":
        return 0.72
    if decision == "No":
        return 0.24
    return 0.48


def draw_panel(ax: Axes, summary: TransitionSummary, show_lane_labels: bool = True) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _add_family_title(ax, summary.spec)

    gate_color = WARMTH_GATE if summary.axis == "warmth" else COMPETENCE_GATE
    # When the lane labels are hidden, there is no need to reserve the wide
    # left gutter that fits "SAME\nINPUT" — pull the whole diagram left and
    # stretch it into that freed space instead of leaving it blank.
    grid_start = 0.18 if show_lane_labels else 0.02
    start_x = 0.22 if show_lane_labels else 0.05
    end_x = 0.84
    gate_x = start_x + (end_x - start_x) * 0.387097
    join_x = start_x + (end_x - start_x) * 0.725806

    yes_y, input_y, no_y = 0.72, 0.48, 0.24
    for label, y in (("YES", yes_y), ("SAME\nINPUT", input_y), ("NO", no_y)):
        ax.plot((grid_start, 0.96), (y, y), color=LIGHT_GRAY, linewidth=0.8, zorder=0)
        if show_lane_labels:
            ax.text(0.015, y, label, ha="left", va="center", fontsize=6.2, color=MUTED)
    ax.add_patch(
        FancyArrowPatch(
            (start_x, input_y),
            (gate_x, input_y),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.8,
            color=CHARCOAL,
            zorder=4,
        )
    )
    ax.plot(start_x, input_y, marker="o", markersize=3.6, color=CHARCOAL, zorder=6)

    baseline_y = _decision_y(summary.baseline_decision)
    steered_y = _decision_y(summary.steered_decision)
    same_lane = baseline_y == steered_y
    if same_lane:
        direction = 1 if summary.delta >= 0 else -1
        baseline_y -= direction * 0.028
        steered_y += direction * 0.028

    _smooth_connector(
        ax,
        (gate_x, input_y),
        (join_x, baseline_y),
        (end_x, baseline_y),
        MUTED,
        dashed=True,
    )
    _smooth_connector(
        ax,
        (gate_x, input_y),
        (join_x, steered_y),
        (end_x, steered_y),
        STEERED_BLUE,
        dashed=False,
    )

    ax.add_patch(
        FancyArrowPatch(
            (gate_x, 0.89),
            (gate_x, input_y + 0.02),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.8,
            color=gate_color,
            zorder=7,
        )
    )
    axis_short = "W" if summary.axis == "warmth" else "C"
    ax.text(
        gate_x - 0.015,
        0.88,
        f"+{axis_short}  α={summary.spec.strength:+.2f}",
        ha="right",
        va="top",
        fontsize=6.5,
        color=gate_color,
    )
    ax.plot(
        gate_x,
        input_y,
        marker="o",
        markersize=6.2,
        markerfacecolor=WHITE,
        markeredgecolor=gate_color,
        markeredgewidth=1.7,
        zorder=9,
    )

    baseline_offset = 0.047 if same_lane and baseline_y > steered_y else -0.047
    steered_offset = -0.047 if same_lane and steered_y < baseline_y else 0.047
    for y, color, prefix, value, offset in (
        (baseline_y, MUTED, "B", summary.baseline_mean, baseline_offset),
        (steered_y, STEERED_BLUE, "S", summary.steered_mean, steered_offset),
    ):
        ax.plot(
            end_x,
            y,
            marker="o",
            markersize=5.8,
            markerfacecolor=WHITE,
            markeredgecolor=color,
            markeredgewidth=1.6,
            zorder=9,
        )
        ax.text(
            0.975,
            y + offset,
            f"{prefix} {value:+.3f}",
            ha="right",
            va="bottom" if offset > 0 else "top",
            fontsize=6.3,
            color=color,
        )

    ax.text(
        0.98,
        0.91,
        f"Δmargin {summary.delta:+.3f}",
        ha="right",
        va="top",
        fontsize=6.3,
        color=CHARCOAL,
    )
    ax.text(
        0.56,
        0.045,
        _transition_annotation(summary),
        ha="center",
        va="center",
        fontsize=6.25,
        color=CHARCOAL,
    )


def create_figure(summaries: list[TransitionSummary], axis: str) -> Figure:
    by_key = {(summary.spec.label, summary.axis): summary for summary in summaries}
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 5.8), facecolor=WHITE)
    for row_index, row in enumerate(MODEL_ROWS):
        for column_index, spec in enumerate(row):
            draw_panel(axes[row_index, column_index], by_key[(spec.label, axis)])

    gate_color = WARMTH_GATE if axis == "warmth" else COMPETENCE_GATE
    handles = [
        Line2D([0], [0], color=MUTED, linewidth=1.6, linestyle=(0, (4, 3))),
        Line2D([0], [0], color=STEERED_BLUE, linewidth=2.2),
        Line2D([0], [0], color=gate_color, marker="v", linewidth=1.7, markersize=5),
    ]
    fig.legend(
        handles,
        ("Baseline", "Steered", f"+{axis.capitalize()} intervention"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.6,
        fontsize=7.4,
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.925, bottom=0.025, wspace=0.12, hspace=0.40)
    return fig


def create_matched_figure(summaries: list[TransitionSummary]) -> Figure:
    by_key = {(summary.spec.label, summary.axis): summary for summary in summaries}
    selected_models = ("Gemma-3-12B", "Gemma-3-27B")
    selected_axes = ("warmth", "competence")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 3.7), facecolor=WHITE)
    for row_index, axis in enumerate(selected_axes):
        for column_index, model in enumerate(selected_models):
            draw_panel(
                axes[row_index, column_index],
                by_key[(model, axis)],
                show_lane_labels=(column_index == 0),
            )

    handles = [
        Line2D([0], [0], color=MUTED, linewidth=1.6, linestyle=(0, (4, 3))),
        Line2D([0], [0], color=STEERED_BLUE, linewidth=2.2),
        Line2D(
            [0],
            [0],
            color=WARMTH_GATE,
            marker="v",
            linewidth=1.7,
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            color=COMPETENCE_GATE,
            marker="v",
            linewidth=1.7,
            markersize=5,
        ),
    ]
    fig.legend(
        handles,
        ("Baseline", "Steered", "+Warmth", "+Competence"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
        fontsize=7.4,
    )
    fig.subplots_adjust(
        left=0.035,
        right=0.99,
        top=0.89,
        bottom=0.03,
        wspace=0.02,
        hspace=0.24,
    )
    return fig


def save_figure(fig: Figure, out_dir: Path, basename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = out_dir / f"{basename}.{extension}"
        fig.savefig(path, format=extension, bbox_inches="tight", facecolor=WHITE)
        print(f"saved {path}")
    plt.close(fig)


def run_figure(axis: str, basename: str) -> None:
    parser = argparse.ArgumentParser(description=f"Generate the nine-model {axis} transition figure.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()
    configure_style()
    summaries = build_summaries()
    write_summary_csv(summaries, args.summary_out)
    save_figure(create_figure(summaries, axis), args.out_dir, basename)


def run_matched_figure(basename: str) -> None:
    parser = argparse.ArgumentParser(
        description="Generate matched bidirectional hiring-transition examples."
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--table-out", type=Path, default=SUMMARY_TEX_PATH)
    args = parser.parse_args()
    configure_style()
    summaries = build_summaries()
    write_summary_csv(summaries, args.summary_out)
    write_summary_tex(summaries, args.table_out)
    save_figure(create_matched_figure(summaries), args.out_dir, basename)
