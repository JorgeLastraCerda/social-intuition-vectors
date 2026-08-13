import matplotlib as mpl


def apply() -> None:
    mpl.style.use(["seaborn-v0_8-paper", "seaborn-v0_8-ticks"])
    mpl.rcParams.update({
        # Match the manuscript, which is \documentclass{article} with no font
        # package and therefore Computer Modern serif. Sans-serif figure text
        # read as a different document when placed next to the body type.
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif",
                       "Times New Roman"],
        "mathtext.fontset": "cm",
        # Sizes are set for a figure placed at \textwidth in a two-column
        # layout, where the body font is 10pt. Previous values rendered larger
        # than the surrounding prose.
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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
