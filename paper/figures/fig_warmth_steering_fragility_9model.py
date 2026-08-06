"""
paper/figures/fig_warmth_steering_fragility_9model.py

Nine-model small-multiples grid of local-regime warmth dose-response (mean
change in callback margin vs. steering strength), supporting the findings
report on Limitation "Fragile causal effect at scale"
(paper/2026-08-06_1333_warmth_steering_fragility_scale_9model.md).

Real measured data, not synthetic. Sources (repo-relative to paper/figures/):
  ../../results/tables/hiring_steering_{label}_local.csv           (7 models,
      already-summarized mean delta per strength)
  ../../results/tables/hiring_steering_raw_concept_vectors_gemma3_27b.csv
      (Gemma-3-27B, per-name; aggregated to a mean here)
  ../../results/tables/hiring_steering_raw_gemma3_12b.csv
      (Gemma-3-12B, per-name, BROAD regime only: strengths are
      {-0.5,-0.25,0,0.25,0.5}, not the {-0.1,-0.05,0,0.05,0.1} local grid
      used by the other eight models — panel is labeled accordingly and
      plotted on its own x-axis scale)

Each panel shows warmth mean callback-margin delta vs. steering strength.
Monotone dose-response panels are drawn in blue; panels whose sign reverses
between the two positive-strength points (or the two negative-strength
points) are drawn in red, matching the "fragile / non-monotone" call in the
report.

Outputs (same-basename triplet):
  paper/figures/fig_warmth_steering_fragility_9model.png
  paper/figures/fig_warmth_steering_fragility_9model.pdf
"""

import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).parent
_TABLES = _HERE / ".." / ".." / "results" / "tables"
sys.path.insert(0, str(_HERE))
import style  # shared palette + apply()

style.apply()

MONOTONE = style.ARROW_WARMTH       # blue — monotone dose-response
FRAGILE = "#C0392B"                 # red — non-monotone / sign-reversing


def load_summarized_local(label: str) -> dict[float, float]:
    """Read a pre-summarized `hiring_steering_<label>_local.csv` (axis,
    strength, mean_delta, ci_95_low, ci_95_high, n_names, n_boot)."""
    path = _TABLES / f"hiring_steering_{label}_local.csv"
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["axis"] != "warmth":
                continue
            out[float(row["strength"])] = float(row["mean_delta"])
    return out


def load_raw_per_name(path: pathlib.Path) -> dict[float, float]:
    """Read a per-name `axis,strength,name,margin,delta` CSV and aggregate
    to the mean delta per strength for the warmth axis."""
    sums: dict[float, float] = {}
    counts: dict[float, int] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["axis"] != "warmth":
                continue
            s = float(row["strength"])
            sums[s] = sums.get(s, 0.0) + float(row["delta"])
            counts[s] = counts.get(s, 0) + 1
    return {s: sums[s] / counts[s] for s in sums}


# (panel title, dose-response dict, is_fragile, x-axis note)
MODELS = [
    (
        "Gemma-3-12B",
        load_raw_per_name(_TABLES / "hiring_steering_raw_gemma3_12b.csv"),
        False,
        "broad regime",
    ),
    (
        "Gemma-3-27B",
        load_raw_per_name(
            _TABLES / "hiring_steering_raw_concept_vectors_gemma3_27b.csv"
        ),
        True,
        None,
    ),
    ("Llama-3.1-8B", load_summarized_local("llama31_8b"), False, None),
    ("Gemma-4-12B", load_summarized_local("gemma4_12b"), False, None),
    ("Gemma-4-26B-A4B", load_summarized_local("gemma4_26b_a4b"), False, "near-inert"),
    ("Gemma-4-31B", load_summarized_local("gemma4_31b"), True, None),
    ("Qwen3-14B", load_summarized_local("qwen3_14b"), False, None),
    ("Qwen3.6-27B", load_summarized_local("qwen36_27b"), False, None),
    ("Qwen3.6-35B-A3B", load_summarized_local("qwen36_35b_a3b"), False, None),
]

fig, axes = plt.subplots(3, 3, figsize=(9.0, 8.0), sharey=False)

for ax, (title, dose, is_fragile, note) in zip(axes.flat, MODELS):
    strengths = sorted(dose.keys())
    deltas = [dose[s] for s in strengths]
    color = FRAGILE if is_fragile else MONOTONE
    ax.axhline(0, color="#CCCCCC", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#CCCCCC", linewidth=0.8, zorder=1)
    ax.plot(strengths, deltas, marker="o", color=color, markersize=5, zorder=3)
    label = title + (f"\n({note})" if note else "")
    ax.set_title(label, fontsize=10)
    ax.tick_params(labelsize=8)

for ax in axes[-1, :]:
    ax.set_xlabel("Steering strength " + r"$\alpha$", fontsize=8.5)
for ax in axes[:, 0]:
    ax.set_ylabel(r"Mean $\Delta$ callback margin", fontsize=8.5)

fig.suptitle(
    "Warmth steering dose-response across nine models (local regime except "
    "Gemma-3-12B)",
    fontsize=11,
)

legend_handles = [
    plt.Line2D([0], [0], color=MONOTONE, marker="o", label="Monotone"),
    plt.Line2D([0], [0], color=FRAGILE, marker="o", label="Non-monotone / fragile"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.01),
)

plt.tight_layout(rect=(0, 0.02, 1, 0.97))

out = _HERE / "fig_warmth_steering_fragility_9model"
fig.savefig(out.with_suffix(".png"), dpi=300)
fig.savefig(out.with_suffix(".pdf"))
print(f"Saved {out}.png  and  {out}.pdf")
for title, dose, is_fragile, _ in MODELS:
    tag = "FRAGILE" if is_fragile else "monotone"
    print(f"  {title:20s} [{tag}] " + " ".join(f"{s:+.2f}:{d:+.3f}" for s, d in sorted(dose.items())))
