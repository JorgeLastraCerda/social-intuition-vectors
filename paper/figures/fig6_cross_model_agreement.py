"""Cross-model story-ranking agreement, all 36 pairs, as a dot plot.

Replaces the four 9x9 heatmaps previously used for this result. Those panels
carried 324 cells, half of them redundant by symmetry, and the two "overall"
panels held no visible variation because every value fell between 0.74 and
0.99. This figure shows the same four distributions as strips of 36 points
with a median marker, which is what the Results prose actually reports.

Run from the repository root:

    python paper/figures/fig6_cross_model_agreement.py

Writes fig6_cross_model_agreement.{pdf,png} next to this file.

Data source: results/tables/cross_model_agreement_9model.csv, produced by
src/validate_cross_model_agreement.py. That CSV is gitignored, so if it is
absent this script falls back to parsing the committed LaTeX table
results/tables/cross_model_agreement_9model.tex, which carries the same two
columns for all 72 pair-axis rows.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402  (repo-local, applies shared rcParams)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TABLE_DIR = ROOT / "results" / "tables"
CSV = TABLE_DIR / "cross_model_agreement_9model.csv"
TEX = TABLE_DIR / "cross_model_agreement_9model.tex"

# Row order is top to bottom in the plot; keep overall above within so the
# drop between them reads downward.
ROWS = [
    ("Warmth", "overall", "Warmth\noverall"),
    ("Warmth", "within", "Warmth\nwithin-condition"),
    ("Competence", "overall", "Competence\noverall"),
    ("Competence", "within", "Competence\nwithin-condition"),
]


def load_from_csv() -> dict[tuple[str, str], list[float]]:
    import pandas as pd

    df = pd.read_csv(CSV)
    out: dict[tuple[str, str], list[float]] = {}
    for axis in df["axis"].unique():
        sub = df[df["axis"] == axis]
        key_axis = str(axis).capitalize()
        out[(key_axis, "overall")] = sub["overall_rho"].astype(float).tolist()
        out[(key_axis, "within")] = sub["within_condition_rho"].astype(float).tolist()
    return out


def load_from_tex() -> dict[tuple[str, str], list[float]]:
    """Fallback: parse the committed LaTeX table.

    Rows look like:  Gemma-3-12B & Gemma-3-27B & 0.818 & 0.555 \\
    The table lists all Warmth pairs first, then all Competence pairs.
    """
    text = TEX.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^\s*([A-Za-z0-9.\-]+)\s*&\s*([A-Za-z0-9.\-]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)",
        re.M,
    )
    pairs = [(float(m.group(3)), float(m.group(4))) for m in pattern.finditer(text)]
    if len(pairs) != 72:
        raise RuntimeError(
            f"expected 72 pair rows in {TEX.name}, parsed {len(pairs)}; "
            "regenerate the CSV instead of relying on this fallback"
        )
    warmth, competence = pairs[:36], pairs[36:]
    return {
        ("Warmth", "overall"): [p[0] for p in warmth],
        ("Warmth", "within"): [p[1] for p in warmth],
        ("Competence", "overall"): [p[0] for p in competence],
        ("Competence", "within"): [p[1] for p in competence],
    }


def main() -> None:
    if CSV.exists():
        data = load_from_csv()
        source = CSV.name
    elif TEX.exists():
        data = load_from_tex()
        source = TEX.name + " (fallback)"
    else:
        raise SystemExit(f"no data found: looked for {CSV} and {TEX}")

    style.apply()
    rng = np.random.default_rng(20260527)  # same seed convention as the pipeline

    # Single-column width for a twocolumn article: ~3.4in usable.
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    colour = {
        "overall": style.PALETTE["low_competence"],
        "within": style.PALETTE["high_warmth"],
    }

    for i, (axis, kind, label) in enumerate(ROWS):
        y = len(ROWS) - 1 - i  # first row at top
        vals = np.asarray(data[(axis, kind)], dtype=float)
        jitter = rng.uniform(-0.16, 0.16, size=vals.size)
        ax.scatter(
            vals,
            np.full_like(vals, y) + jitter,
            s=11,
            alpha=0.55,
            color=colour[kind],
            edgecolors="none",
            zorder=2,
        )
        med = float(np.median(vals))
        ax.plot(
            [med, med], [y - 0.30, y + 0.30],
            color="black", linewidth=1.4, zorder=3, solid_capstyle="butt",
        )
        ax.annotate(
            f"{med:.2f}",
            xy=(med, y + 0.34),
            ha="center", va="bottom", fontsize=7, zorder=4,
        )

    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[2] for r in reversed(ROWS)])
    ax.set_ylim(-0.6, len(ROWS) - 0.35)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Spearman $\rho$ between model pairs")
    ax.axvline(0, color="0.85", linewidth=0.8, zorder=1)
    ax.grid(axis="x", color="0.9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    for ext in ("pdf", "png"):
        out = HERE / f"fig6_cross_model_agreement.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out.relative_to(ROOT)}")
    plt.close(fig)

    print(f"source: {source}")
    for axis, kind, label in ROWS:
        v = np.asarray(data[(axis, kind)], dtype=float)
        print(
            f"  {axis:11} {kind:7} n={v.size:2d} "
            f"min={v.min():.2f} median={np.median(v):.2f} max={v.max():.2f}"
        )


if __name__ == "__main__":
    main()
