"""Shared data validation and drawing helpers for steering-flip pilot figures."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "results/tables/hiring_steering_raw_concept_vectors.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "paper/figures"

CHARCOAL = "#2F343B"
MUTED = "#6B7280"
LIGHT_GRAY = "#D7DCE2"
PALE_GRAY = "#F2F4F6"
WARMTH = "#D55E00"
YES_BLUE = "#0072B2"
PALE_BLUE = "#E8F3F8"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class PilotData:
    model_label: str
    axis_label: str
    strength: float
    baseline_mean: float
    steered_mean: float
    no_to_yes: int
    tie_to_yes: int
    n_names: int


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _classify(value: float) -> str:
    if value > 0:
        return "Yes"
    if value < 0:
        return "No"
    return "Tie"


def load_pilot_data(path: Path) -> PilotData:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {"axis", "strength", "name", "margin", "delta"}
    if not rows or not required.issubset(rows[0]):
        missing = required.difference(rows[0] if rows else {})
        raise ValueError(f"Missing required steering columns: {sorted(missing)}")

    by_strength: dict[float, dict[str, float]] = {0.0: {}, 0.05: {}}
    for row in rows:
        if row["axis"] != "warmth":
            continue
        strength = float(row["strength"])
        matched = next((s for s in by_strength if math.isclose(strength, s, abs_tol=1e-9)), None)
        if matched is None:
            continue
        name = row["name"]
        if name in by_strength[matched]:
            raise ValueError(f"Duplicate warmth row for {name!r} at strength {matched:+.2f}")
        by_strength[matched][name] = float(row["margin"])

    baseline = by_strength[0.0]
    steered = by_strength[0.05]
    if len(baseline) != 60 or len(steered) != 60:
        raise ValueError(
            f"Pilot requires 60 names at each condition; found {len(baseline)} and {len(steered)}"
        )
    if baseline.keys() != steered.keys():
        raise ValueError("Baseline and steered conditions contain different names")

    transitions: dict[tuple[str, str], int] = {}
    for name in baseline:
        key = (_classify(baseline[name]), _classify(steered[name]))
        transitions[key] = transitions.get(key, 0) + 1

    data = PilotData(
        model_label="Gemma-3-12B",
        axis_label="Warmth",
        strength=0.05,
        baseline_mean=sum(baseline.values()) / len(baseline),
        steered_mean=sum(steered.values()) / len(steered),
        no_to_yes=transitions.get(("No", "Yes"), 0),
        tie_to_yes=transitions.get(("Tie", "Yes"), 0),
        n_names=len(baseline),
    )

    expected = {
        "baseline_mean": -0.19375,
        "steered_mean": 0.9833333333333333,
        "no_to_yes": 54,
        "tie_to_yes": 6,
    }
    if not math.isclose(data.baseline_mean, expected["baseline_mean"], abs_tol=1e-12):
        raise ValueError(f"Unexpected baseline mean: {data.baseline_mean}")
    if not math.isclose(data.steered_mean, expected["steered_mean"], abs_tol=1e-12):
        raise ValueError(f"Unexpected steered mean: {data.steered_mean}")
    if data.no_to_yes != expected["no_to_yes"] or data.tie_to_yes != expected["tie_to_yes"]:
        raise ValueError(
            "Unexpected transition counts: "
            f"No→Yes={data.no_to_yes}, tie→Yes={data.tie_to_yes}"
        )
    return data


def _header(ax: Axes, data: PilotData, variant: str, panel: str | None) -> None:
    prefix = f"{panel}   " if panel else ""
    ax.text(
        0.01,
        0.97,
        f"{prefix}{variant}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="normal",
        color=CHARCOAL,
        zorder=20,
        bbox=dict(facecolor=WHITE, edgecolor="none", pad=1.5),
    )
    ax.text(
        0.99,
        0.97,
        f"{data.model_label} · {data.axis_label}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=MUTED,
        zorder=20,
        bbox=dict(facecolor=WHITE, edgecolor="none", pad=1.5),
    )


def _endpoint(ax: Axes, x: float, y: float, color: str, label: str, value: float) -> None:
    ax.add_patch(Circle((x, y), 0.035, facecolor=WHITE, edgecolor=color, linewidth=2.3, zorder=5))
    ax.text(x, y + 0.10, label, ha="center", va="bottom", fontsize=12, color=CHARCOAL)
    ax.text(
        x,
        y - 0.105,
        f"mean margin {value:+.3f}",
        ha="center",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )


def draw_deflection(ax: Axes, data: PilotData, panel: str | None = None) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _header(ax, data, "Deflected causal flow", panel)

    start = (0.08, 0.48)
    gate = (0.43, 0.48)
    no_end = (0.88, 0.22)
    yes_end = (0.88, 0.65)

    ax.text(start[0], start[1] + 0.11, "Same application", ha="center", color=CHARCOAL)
    ax.add_patch(
        FancyArrowPatch(
            start,
            gate,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=2.0,
            color=CHARCOAL,
        )
    )
    ax.add_patch(Circle(gate, 0.024, facecolor=WHITE, edgecolor=WARMTH, linewidth=2.2, zorder=6))

    ax.add_patch(
        FancyArrowPatch(
            gate,
            no_end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.8,
            linestyle=(0, (4, 3)),
            color=MUTED,
            connectionstyle="arc3,rad=-0.12",
        )
    )
    ax.text(0.66, 0.31, "No steering", ha="center", va="top", fontsize=8.5, color=MUTED)

    ax.add_patch(
        FancyArrowPatch(
            gate,
            yes_end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=3.0,
            color=YES_BLUE,
            connectionstyle="arc3,rad=0.14",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (gate[0], 0.83),
            (gate[0], gate[1] + 0.035),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.2,
            color=WARMTH,
        )
    )
    ax.text(
        gate[0] + 0.015,
        0.82,
        f"+ warmth steering\nα = {data.strength:+.2f} × mean residual norm",
        ha="left",
        va="top",
        fontsize=8.5,
        color=CHARCOAL,
    )

    _endpoint(ax, *no_end, MUTED, "No", data.baseline_mean)
    _endpoint(ax, *yes_end, YES_BLUE, "Yes", data.steered_mean)
    ax.text(
        0.48,
        0.045,
        f"{data.no_to_yes}/{data.n_names} strict No→Yes flips  ·  "
        f"{data.tie_to_yes} ties→Yes",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=CHARCOAL,
    )


def draw_boundary(ax: Axes, data: PilotData, panel: str | None = None) -> None:
    ax.set_xlim(-0.55, 1.30)
    ax.set_ylim(-0.50, 0.92)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(CHARCOAL)
    ax.set_yticks([])
    ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis="x", colors=MUTED, length=3)
    ax.set_xlabel("Callback margin  (logit Yes − logit No)", color=CHARCOAL, labelpad=3)
    ax.axvspan(-0.55, 0, color=PALE_GRAY, zorder=0)
    ax.axvspan(0, 1.30, color=PALE_BLUE, zorder=0)
    ax.axvline(0, color=CHARCOAL, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    _header(ax, data, "Decision-boundary bend", panel)

    ax.text(-0.48, 0.60, "No", ha="left", va="center", fontsize=11, color=CHARCOAL)
    ax.text(0.01, -0.39, "decision boundary", ha="left", va="center", fontsize=8, color=MUTED)

    start = (data.baseline_mean, 0.02)
    end = (data.steered_mean, 0.38)
    ax.plot(*start, marker="o", markersize=8, markerfacecolor=WHITE,
            markeredgecolor=MUTED, markeredgewidth=2.0, zorder=5)
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=3.0,
            color=YES_BLUE,
            connectionstyle="arc3,rad=-0.20",
            zorder=4,
        )
    )
    ax.plot(*end, marker="o", markersize=8, markerfacecolor=WHITE,
            markeredgecolor=YES_BLUE, markeredgewidth=2.2, zorder=5)

    injection_x = -0.02
    ax.add_patch(
        FancyArrowPatch(
            (injection_x, 0.78),
            (injection_x, 0.25),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.2,
            color=WARMTH,
            zorder=6,
        )
    )
    ax.text(
        injection_x + 0.035,
        0.76,
        f"+ warmth steering\nα = {data.strength:+.2f} × mean residual norm",
        ha="left",
        va="top",
        fontsize=8.5,
        color=CHARCOAL,
    )
    ax.annotate(
        f"Unsteered\n{data.baseline_mean:+.3f}",
        xy=start,
        xytext=(-0.43, 0.19),
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
        arrowprops=dict(arrowstyle="-", color=LIGHT_GRAY, linewidth=0.9),
    )
    ax.annotate(
        f"Yes · Steered\n{data.steered_mean:+.3f}",
        xy=end,
        xytext=(1.20, 0.43),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=CHARCOAL,
        arrowprops=dict(arrowstyle="-", color=YES_BLUE, linewidth=0.9),
    )
    ax.text(
        0.60,
        -0.27,
        f"{data.no_to_yes}/{data.n_names} strict flips  ·  {data.tie_to_yes} ties→Yes",
        ha="center",
        va="center",
        fontsize=8.5,
        color=CHARCOAL,
    )


def _boundary_variant_canvas(
    ax: Axes,
    data: PilotData,
    variant: str,
    panel: str | None,
) -> None:
    ax.set_xlim(-0.55, 1.30)
    ax.set_ylim(-0.50, 0.92)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(CHARCOAL)
    ax.set_yticks([])
    ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis="x", colors=MUTED, length=3)
    ax.set_xlabel("Callback margin  (logit Yes − logit No)", color=CHARCOAL, labelpad=3)
    ax.axvspan(-0.55, 0, color=PALE_GRAY, zorder=0)
    ax.axvspan(0, 1.30, color=PALE_BLUE, zorder=0)
    ax.axvline(0, color=CHARCOAL, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    _header(ax, data, variant, panel)
    ax.text(0.01, -0.39, "decision boundary", ha="left", va="center", fontsize=8, color=MUTED)


def _transition_summary(ax: Axes, data: PilotData, x: float = 0.60) -> None:
    ax.text(
        x,
        -0.27,
        f"{data.no_to_yes}/{data.n_names} strict No→Yes flips  ·  "
        f"{data.tie_to_yes} ties→Yes",
        ha="center",
        va="center",
        fontsize=8.5,
        color=CHARCOAL,
    )


def draw_boundary_kink(ax: Axes, data: PilotData, panel: str | None = None) -> None:
    """Sharp schematic break from the baseline No endpoint to the steered Yes endpoint."""
    _boundary_variant_canvas(ax, data, "Sharp counterfactual kink", panel)
    baseline = (data.baseline_mean, 0.02)
    steered = (data.steered_mean, 0.34)

    ax.add_patch(
        FancyArrowPatch(
            (0.28, 0.52),
            baseline,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.1,
            color=MUTED,
            zorder=3,
        )
    )
    ax.text(0.27, 0.45, "Unsteered direction", ha="center", va="bottom",
            fontsize=8.5, color=MUTED)
    ax.add_patch(
        FancyArrowPatch(
            baseline,
            (-0.47, -0.08),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.6,
            linestyle=(0, (4, 3)),
            color=MUTED,
            zorder=2,
        )
    )
    ax.text(-0.46, -0.15, "No without steering", ha="left", va="top",
            fontsize=8.1, color=MUTED)

    ax.add_patch(
        FancyArrowPatch(
            baseline,
            steered,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=3.0,
            color=YES_BLUE,
            connectionstyle="arc3,rad=-0.12",
            zorder=5,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (data.baseline_mean, 0.80),
            (data.baseline_mean, 0.10),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.2,
            color=WARMTH,
            zorder=6,
        )
    )
    ax.text(
        -0.46,
        0.67,
        f"+ warmth steering\nα = {data.strength:+.2f} × mean residual norm",
        ha="left",
        va="top",
        fontsize=8.5,
        color=CHARCOAL,
    )

    ax.plot(*baseline, marker="o", markersize=9, markerfacecolor=WHITE,
            markeredgecolor=MUTED, markeredgewidth=2.2, zorder=8)
    ax.plot(*steered, marker="o", markersize=9, markerfacecolor=WHITE,
            markeredgecolor=YES_BLUE, markeredgewidth=2.2, zorder=8)
    ax.text(-0.43, 0.12, f"No · baseline\n{data.baseline_mean:+.3f}", ha="left",
            va="bottom", fontsize=8.5, color=CHARCOAL)
    ax.text(1.18, 0.43, f"Yes · steered\n{data.steered_mean:+.3f}", ha="right",
            va="bottom", fontsize=8.5, color=CHARCOAL)
    _transition_summary(ax, data)


def draw_boundary_hinge(ax: Axes, data: PilotData, panel: str | None = None) -> None:
    """Split the observed margin shift at the zero-margin decision boundary."""
    _boundary_variant_canvas(ax, data, "Decision-boundary hinge", panel)
    baseline = (data.baseline_mean, 0.08)
    hinge = (0.0, 0.20)
    steered = (data.steered_mean, 0.38)

    ax.add_patch(
        FancyArrowPatch(
            baseline,
            (-0.48, -0.03),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.7,
            linestyle=(0, (4, 3)),
            color=MUTED,
            zorder=2,
        )
    )
    ax.text(-0.47, -0.11, "No without steering", ha="left", va="top",
            fontsize=8.1, color=MUTED)

    ax.add_patch(
        FancyArrowPatch(
            baseline,
            hinge,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=3.0,
            color=YES_BLUE,
            zorder=4,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            hinge,
            steered,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=3.0,
            color=YES_BLUE,
            connectionstyle="arc3,rad=-0.08",
            zorder=4,
        )
    )
    intervention_x = (data.baseline_mean + 0.0) / 2.0
    ax.add_patch(
        FancyArrowPatch(
            (intervention_x, 0.79),
            (intervention_x, 0.25),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.2,
            color=WARMTH,
            zorder=6,
        )
    )
    ax.text(
        intervention_x + 0.035,
        0.77,
        f"+ warmth steering\nα = {data.strength:+.2f} × mean residual norm",
        ha="left",
        va="top",
        fontsize=8.5,
        color=CHARCOAL,
    )

    ax.plot(*baseline, marker="o", markersize=8, markerfacecolor=WHITE,
            markeredgecolor=MUTED, markeredgewidth=2.1, zorder=7)
    ax.plot(*hinge, marker="o", markersize=10, markerfacecolor=WHITE,
            markeredgecolor=WARMTH, markeredgewidth=2.4, zorder=7)
    ax.plot(*steered, marker="o", markersize=8, markerfacecolor=WHITE,
            markeredgecolor=YES_BLUE, markeredgewidth=2.1, zorder=7)
    ax.text(-0.43, 0.17, f"No · baseline\n{data.baseline_mean:+.3f}", ha="left",
            va="bottom", fontsize=8.5, color=CHARCOAL)
    ax.text(0.03, -0.02, "decision flips here", ha="left", va="top",
            fontsize=8.2, color=WARMTH)
    ax.text(1.18, 0.47, f"Yes · steered\n{data.steered_mean:+.3f}", ha="right",
            va="bottom", fontsize=8.5, color=CHARCOAL)
    _transition_summary(ax, data)


def draw_boundary_vector_addition(
    ax: Axes,
    data: PilotData,
    panel: str | None = None,
) -> None:
    """Tip-to-tail decomposition of baseline, steering contribution, and outcome."""
    _boundary_variant_canvas(ax, data, "Vector addition", panel)
    origin = (0.0, -0.02)
    baseline_tip = (data.baseline_mean, 0.29)
    steered_tip = (data.steered_mean, 0.43)
    delta = data.steered_mean - data.baseline_mean

    ax.add_patch(
        FancyArrowPatch(
            origin,
            baseline_tip,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.2,
            color=MUTED,
            zorder=4,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            baseline_tip,
            steered_tip,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=2.7,
            color=WARMTH,
            connectionstyle="arc3,rad=-0.05",
            zorder=5,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            origin,
            steered_tip,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=3.0,
            color=YES_BLUE,
            connectionstyle="arc3,rad=-0.08",
            zorder=3,
        )
    )

    ax.plot(*origin, marker="o", markersize=5, color=CHARCOAL, zorder=7)
    ax.plot(*baseline_tip, marker="o", markersize=8, markerfacecolor=WHITE,
            markeredgecolor=MUTED, markeredgewidth=2.1, zorder=7)
    ax.plot(*steered_tip, marker="o", markersize=8, markerfacecolor=WHITE,
            markeredgecolor=YES_BLUE, markeredgewidth=2.1, zorder=7)
    ax.text(-0.45, 0.36, f"Baseline vector\nNo · {data.baseline_mean:+.3f}", ha="left",
            va="bottom", fontsize=8.5, color=MUTED)
    ax.text(0.45, 0.63, f"Warmth contribution\nΔmargin {delta:+.3f}", ha="center",
            va="bottom", fontsize=8.5, color=WARMTH)
    ax.text(1.18, 0.46, f"Resultant\nYes · {data.steered_mean:+.3f}", ha="right",
            va="bottom", fontsize=8.5, color=CHARCOAL)
    ax.text(0.56, -0.26, "horizontal positions empirical  ·  angles schematic",
            ha="center", va="center", fontsize=8.1, color=MUTED)


def _flow_band(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    width: float,
    color: str,
    alpha: float,
) -> None:
    sx, sy = start
    ex, ey = end
    vertices = [
        (sx, sy),
        (sx + 0.18, sy),
        (ex - 0.18, ey),
        (ex, ey),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    ax.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            facecolor="none",
            edgecolor=color,
            linewidth=width,
            alpha=alpha,
            capstyle="round",
            zorder=2,
        )
    )


def _box(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    edge: str,
    fill: str,
    label: str,
    detail: str,
) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.6,
            zorder=4,
        )
    )
    ax.text(x + width / 2, y + height * 0.62, label, ha="center", va="center",
            fontsize=10, color=CHARCOAL, zorder=5)
    ax.text(x + width / 2, y + height * 0.28, detail, ha="center", va="center",
            fontsize=8.2, color=MUTED, zorder=5)


def draw_split_flow(ax: Axes, data: PilotData, panel: str | None = None) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _header(ax, data, "Aggregate split-flow", panel)

    left_x, gate_x, right_x = 0.06, 0.43, 0.80
    box_w, box_h = 0.15, 0.18
    _box(ax, (left_x, 0.50), box_w, box_h, MUTED, PALE_GRAY, "No", "54 names")
    _box(ax, (left_x, 0.20), box_w, box_h, LIGHT_GRAY, WHITE, "Tie", "6 names")
    _box(ax, (right_x, 0.42), box_w, 0.24, YES_BLUE, PALE_BLUE, "Yes", "60 names")

    gate_y = 0.54
    ax.add_patch(Circle((gate_x + 0.075, gate_y), 0.038, facecolor=WHITE,
                        edgecolor=WARMTH, linewidth=2.4, zorder=5))
    _flow_band(ax, (left_x + box_w, 0.59), (gate_x + 0.04, gate_y), 18, MUTED, 0.40)
    _flow_band(ax, (left_x + box_w, 0.29), (gate_x + 0.04, gate_y), 5, LIGHT_GRAY, 0.95)
    _flow_band(ax, (gate_x + 0.11, gate_y), (right_x, 0.54), 21, YES_BLUE, 0.55)
    ax.add_patch(
        FancyArrowPatch(
            (0.76, 0.54),
            (right_x + 0.005, 0.54),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.6,
            color=YES_BLUE,
            zorder=3,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (gate_x + 0.075, 0.86),
            (gate_x + 0.075, gate_y + 0.045),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.2,
            color=WARMTH,
            zorder=6,
        )
    )
    ax.text(
        gate_x + 0.095,
        0.84,
        f"+ warmth steering\nα = {data.strength:+.2f} × mean residual norm",
        ha="left",
        va="top",
        fontsize=8.5,
        color=CHARCOAL,
    )
    ax.text(0.135, 0.10, f"Unsteered mean {data.baseline_mean:+.3f}", ha="center",
            va="center", fontsize=8.2, color=MUTED)
    ax.text(0.875, 0.29, f"Steered mean {data.steered_mean:+.3f}", ha="center",
            va="center", fontsize=8.2, color=MUTED)
    ax.text(0.50, 0.12, f"{data.no_to_yes} No→Yes  ·  {data.tie_to_yes} tie→Yes",
            ha="center", va="center", fontsize=8.8, color=CHARCOAL)


def _lane_canvas(
    ax: Axes,
    data: PilotData,
    variant: str,
    panel: str | None,
) -> tuple[float, float, float, float, float]:
    """Create the shared schematic three-lane decision canvas."""
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1)
    ax.axis("off")
    _header(ax, data, variant, panel)

    yes_y, neutral_y, no_y = 0.72, 0.48, 0.24
    start_x, gate_x, end_x = 0.15, 0.42, 0.88
    for label, y in (("YES", yes_y), ("NEUTRAL", neutral_y), ("NO", no_y)):
        ax.plot((0.10, 0.95), (y, y), color=LIGHT_GRAY, linewidth=1.0, zorder=0)
        ax.text(
            0.02,
            y,
            label,
            ha="left",
            va="center",
            fontsize=8.2,
            color=MUTED,
        )

    ax.add_patch(
        FancyArrowPatch(
            (start_x, neutral_y),
            (gate_x, neutral_y),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=2.4,
            color=CHARCOAL,
            zorder=3,
        )
    )
    ax.plot(start_x, neutral_y, marker="o", markersize=5.5, color=CHARCOAL, zorder=5)
    ax.text(
        start_x,
        neutral_y + 0.07,
        "Same application",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=CHARCOAL,
    )
    return yes_y, neutral_y, no_y, gate_x, end_x


def _finish_lane_canvas(
    ax: Axes,
    data: PilotData,
    yes_y: float,
    neutral_y: float,
    no_y: float,
    gate_x: float,
    end_x: float,
) -> None:
    """Add the shared intervention, endpoints, and empirical annotation."""
    ax.add_patch(
        FancyArrowPatch(
            (gate_x, 0.86),
            (gate_x, neutral_y + 0.018),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.3,
            color=WARMTH,
            zorder=7,
        )
    )
    ax.text(
        gate_x - 0.02,
        0.87,
        f"+ warmth steering\nα = {data.strength:+.2f} × mean residual norm",
        ha="right",
        va="top",
        fontsize=8.4,
        color=WARMTH,
        linespacing=1.15,
        zorder=8,
    )

    ax.plot(
        gate_x,
        neutral_y,
        marker="o",
        markersize=9,
        markerfacecolor=WHITE,
        markeredgecolor=WARMTH,
        markeredgewidth=2.3,
        zorder=9,
    )
    ax.plot(
        end_x,
        yes_y,
        marker="o",
        markersize=9,
        markerfacecolor=WHITE,
        markeredgecolor=YES_BLUE,
        markeredgewidth=2.2,
        zorder=9,
    )
    ax.plot(
        end_x,
        no_y,
        marker="o",
        markersize=9,
        markerfacecolor=WHITE,
        markeredgecolor=MUTED,
        markeredgewidth=2.2,
        zorder=9,
    )
    ax.text(
        0.95,
        yes_y + 0.07,
        f"Yes · steered  {data.steered_mean:+.3f}",
        ha="right",
        va="bottom",
        fontsize=8.6,
        color=YES_BLUE,
    )
    ax.text(
        0.95,
        no_y - 0.065,
        f"No · baseline  {data.baseline_mean:+.3f}",
        ha="right",
        va="top",
        fontsize=8.6,
        color=MUTED,
    )
    ax.text(
        0.53,
        -0.015,
        f"{data.no_to_yes}/{data.n_names} strict No→Yes flips  ·  "
        f"{data.tie_to_yes} ties→Yes",
        ha="center",
        va="center",
        fontsize=8.3,
        color=CHARCOAL,
    )


def _lane_horizontal_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    dashed: bool = False,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.8 if not dashed else 2.0,
            linestyle=(0, (4, 3)) if dashed else "solid",
            color=color,
            zorder=5 if not dashed else 3,
        )
    )


def draw_lane_switch_diagonal(
    ax: Axes,
    data: PilotData,
    panel: str | None = None,
) -> None:
    """Show a short diagonal switch followed by horizontal decision flow."""
    yes_y, neutral_y, no_y, gate_x, end_x = _lane_canvas(
        ax, data, "Diagonal lane switch", panel
    )
    join_x = 0.61
    ax.plot(
        (gate_x, join_x),
        (neutral_y, yes_y),
        color=YES_BLUE,
        linewidth=2.8,
        solid_capstyle="round",
        zorder=5,
    )
    ax.plot(
        (gate_x, join_x),
        (neutral_y, no_y),
        color=MUTED,
        linewidth=2.0,
        linestyle=(0, (4, 3)),
        solid_capstyle="round",
        zorder=3,
    )
    _lane_horizontal_arrow(ax, (join_x, yes_y), (end_x, yes_y), YES_BLUE)
    _lane_horizontal_arrow(ax, (join_x, no_y), (end_x, no_y), MUTED, dashed=True)
    _finish_lane_canvas(ax, data, yes_y, neutral_y, no_y, gate_x, end_x)


def draw_lane_switch_step(
    ax: Axes,
    data: PilotData,
    panel: str | None = None,
) -> None:
    """Show an immediate right-angle switch at the steering gate."""
    yes_y, neutral_y, no_y, gate_x, end_x = _lane_canvas(
        ax, data, "Right-angle switch", panel
    )
    join_x = gate_x + 0.055
    ax.plot(
        (gate_x, join_x, join_x),
        (neutral_y, neutral_y, yes_y),
        color=YES_BLUE,
        linewidth=2.8,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=5,
    )
    ax.plot(
        (gate_x, join_x, join_x),
        (neutral_y, neutral_y, no_y),
        color=MUTED,
        linewidth=2.0,
        linestyle=(0, (4, 3)),
        solid_capstyle="round",
        zorder=3,
    )
    _lane_horizontal_arrow(ax, (join_x, yes_y), (end_x, yes_y), YES_BLUE)
    _lane_horizontal_arrow(ax, (join_x, no_y), (end_x, no_y), MUTED, dashed=True)
    _finish_lane_canvas(ax, data, yes_y, neutral_y, no_y, gate_x, end_x)


def _short_lane_curve(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    dashed: bool = False,
) -> None:
    sx, sy = start
    ex, ey = end
    path = MplPath(
        [(sx, sy), (sx + 0.07, sy), (ex - 0.07, ey), (ex, ey)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=2.8 if not dashed else 2.0,
            linestyle=(0, (4, 3)) if dashed else "solid",
            capstyle="round",
            zorder=5 if not dashed else 3,
        )
    )


def draw_lane_switch_smooth(
    ax: Axes,
    data: PilotData,
    panel: str | None = None,
) -> None:
    """Show a compact smooth deflection that settles onto a decision lane."""
    yes_y, neutral_y, no_y, gate_x, end_x = _lane_canvas(
        ax, data, "Smooth lane switch", panel
    )
    join_x = 0.64
    _short_lane_curve(ax, (gate_x, neutral_y), (join_x, yes_y), YES_BLUE)
    _short_lane_curve(ax, (gate_x, neutral_y), (join_x, no_y), MUTED, dashed=True)
    _lane_horizontal_arrow(ax, (join_x, yes_y), (end_x, yes_y), YES_BLUE)
    _lane_horizontal_arrow(ax, (join_x, no_y), (end_x, no_y), MUTED, dashed=True)
    _finish_lane_canvas(ax, data, yes_y, neutral_y, no_y, gate_x, end_x)


DrawFunction = Callable[[Axes, PilotData, str | None], None]


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def save_figure(fig: Figure, out_dir: Path, basename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{basename}.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight", facecolor=WHITE)
        print(f"saved {path}")
    plt.close(fig)


def run_single(draw: DrawFunction, basename: str, variant: str) -> None:
    args = _parser(f"Generate the {variant} steering-flip pilot.").parse_args()
    configure_style()
    data = load_pilot_data(args.input)
    fig, ax = plt.subplots(figsize=(7.0, 2.25), facecolor=WHITE)
    draw(ax, data, None)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.97, bottom=0.08)
    save_figure(fig, args.out_dir, basename)


def run_contact_sheet() -> None:
    args = _parser("Generate the steering-flip pilot contact sheet.").parse_args()
    configure_style()
    data = load_pilot_data(args.input)
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.8), facecolor=WHITE)
    draw_deflection(axes[0], data, "A")
    draw_boundary(axes[1], data, "B")
    draw_split_flow(axes[2], data, "C")
    fig.subplots_adjust(left=0.045, right=0.985, top=0.985, bottom=0.045, hspace=0.34)
    save_figure(fig, args.out_dir, "pilot_steering_flip_contact_sheet")


def run_boundary_refinement_contact_sheet() -> None:
    args = _parser("Generate the refined decision-boundary pilot contact sheet.").parse_args()
    configure_style()
    data = load_pilot_data(args.input)
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.8), facecolor=WHITE)
    draw_boundary_kink(axes[0], data, "A")
    draw_boundary_hinge(axes[1], data, "B")
    draw_boundary_vector_addition(axes[2], data, "C")
    fig.subplots_adjust(left=0.045, right=0.985, top=0.985, bottom=0.045, hspace=0.34)
    save_figure(fig, args.out_dir, "pilot_steering_boundary_refinements")


def run_lane_switch_contact_sheet() -> None:
    args = _parser("Generate the schematic lane-switch pilot contact sheet.").parse_args()
    configure_style()
    data = load_pilot_data(args.input)
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.8), facecolor=WHITE)
    draw_lane_switch_diagonal(axes[0], data, "A")
    draw_lane_switch_step(axes[1], data, "B")
    draw_lane_switch_smooth(axes[2], data, "C")
    fig.subplots_adjust(left=0.045, right=0.985, top=0.985, bottom=0.045, hspace=0.34)
    save_figure(fig, args.out_dir, "pilot_steering_lane_switch_variants")
