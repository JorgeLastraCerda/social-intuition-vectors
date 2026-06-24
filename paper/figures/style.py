import matplotlib as mpl


def apply() -> None:
    mpl.style.use(["seaborn-v0_8-paper", "seaborn-v0_8-ticks"])
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "figure.dpi": 100,
        "lines.linewidth": 1.8,
    })


PALETTE = {
    "high_warmth":     "#2E86AB",
    "low_warmth":      "#A23B72",
    "high_competence": "#F18F01",
    "low_competence":  "#6B7280",
}

LABELS = {
    "high_warmth":     "High warmth",
    "low_warmth":      "Low warmth",
    "high_competence": "High competence",
    "low_competence":  "Low competence",
}

# Colours and style constants for paper-draft figures (paper_figure1–3)
ARROW_WARMTH     = "#1A5276"  # deep blue — warmth direction arrow
ARROW_COMPETENCE = "#7D6608"  # deep gold — competence direction arrow
CONTROL_ALPHA    = 0.45       # opacity for control / non-dense steering lines
