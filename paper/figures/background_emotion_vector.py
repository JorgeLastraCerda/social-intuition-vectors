"""
paper/figures/background_emotion_vector.py

Schematic Figure 1 for §2 Background: an emotion vector is a direction
in the model's activation space.

Concept: given a prompt, the model's residual-stream state moves along
a specific direction (the emotion direction).  Multiple emotion directions
radiate from a shared neutral origin; fear is highlighted.

SCHEMATIC — emotion vectors after Sofroniew, Lindsey et al. (2026).
Illustrative only; not measured data.

Outputs (same-basename triplet):
  paper/figures/background_emotion_vector.png
  paper/figures/background_emotion_vector.pdf
"""

import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
import style  # shared palette + apply()

style.apply()

# ── local emotion colours (different palette from warmth/competence) ──────────
COL_FEAR    = "#C0392B"   # crimson — highlighted; same red as Fig 2 steering
COL_JOY     = "#2980B9"   # steel blue
COL_SADNESS = "#7F8C8D"   # slate grey

ALPHA_SUPPORT = 0.74

# ── geometry ──────────────────────────────────────────────────────────────────
origin = np.array([0.16, 0.48])   # neutral model state h — left of centre

# emotion unit directions (angles in degrees from horizontal)
angle_joy     = 55.0
angle_fear    =  5.0   # nearly horizontal, pointing right
angle_sadness = -50.0

def unit(deg):
    r = np.radians(deg)
    return np.array([np.cos(r), np.sin(r)])

L_dim  = 0.48   # arrow length for joy / sadness
L_fear = 0.62   # longer highlighted fear arrow

tip_joy     = origin + L_dim  * unit(angle_joy)
tip_fear    = origin + L_fear * unit(angle_fear)
tip_sadness = origin + L_dim  * unit(angle_sadness)

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.35, 2.75))

# --- joy arrow ---
ax.annotate("", xy=tip_joy, xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=COL_JOY,
                            lw=2.0, mutation_scale=13,
                            alpha=ALPHA_SUPPORT),
            zorder=4)
ax.text(tip_joy[0] + 0.025, tip_joy[1] + 0.025,
        "joy",
        fontsize=8.8, color=COL_JOY, alpha=0.88,
        ha="left", va="bottom", style="italic")
ax.text(origin[0] + 0.29, origin[1] + 0.30,
        r"$\hat{v}_{\mathrm{joy}}=[\,0.34,\,0.41,\,-0.06,\ldots\,]$",
        fontsize=5.8, color=COL_JOY, alpha=0.82,
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                  edgecolor="none", alpha=0.82))

# --- sadness arrow ---
ax.annotate("", xy=tip_sadness, xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=COL_SADNESS,
                            lw=2.0, mutation_scale=13,
                            alpha=ALPHA_SUPPORT),
            zorder=4)
ax.text(tip_sadness[0] + 0.025, tip_sadness[1] - 0.025,
        "sadness",
        fontsize=8.8, color=COL_SADNESS, alpha=0.88,
        ha="left", va="top", style="italic")
ax.text(origin[0] + 0.30, origin[1] - 0.29,
        r"$\hat{v}_{\mathrm{sad}}=[\,-0.18,\,-0.09,\,0.47,\ldots\,]$",
        fontsize=5.8, color=COL_SADNESS, alpha=0.82,
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                  edgecolor="none", alpha=0.82))

# --- fear arrow (highlighted) ---
ax.annotate("", xy=tip_fear, xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=COL_FEAR,
                            lw=2.6, mutation_scale=15,
                            shrinkB=6),
            zorder=6)
ax.text((origin[0] + tip_fear[0]) / 2,
        (origin[1] + tip_fear[1]) / 2 + 0.07,
        "fear", fontsize=9.5, color=COL_FEAR,
        ha="center", va="bottom", fontweight="bold", style="italic")

# --- model state star at fear tip ---
ax.scatter(*tip_fear, color=COL_FEAR, s=110, marker="*", zorder=8,
           edgecolors="#800000", linewidths=0.6)

# --- numeric bracket vector label (below fear arrow) ---
ax.text((origin[0] + tip_fear[0]) / 2,
        (origin[1] + tip_fear[1]) / 2 - 0.075,
        r"$\hat{v}_{\mathrm{fear}}"
        r"= [\,0.21,\,-0.52,\,0.08,\,\ldots,\,0.14\,]\!\in\!\mathbb{R}^{d}$",
        fontsize=6.2, color=COL_FEAR, alpha=0.85,
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="none", alpha=0.80))

# --- neutral origin: model state h ---
ax.scatter(*origin, color="#2C3E50", s=80, zorder=7,
           edgecolors="white", linewidths=0.8)
ax.text(origin[0] - 0.035, origin[1] + 0.07,
        r"$h$  (neutral state)",
        fontsize=8, color="#2C3E50", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                  edgecolor="none", alpha=0.75))

# --- prompt cue entering from the left ---
ax.annotate("prompt",
            xy=origin, xytext=(origin[0] - 0.20, origin[1]),
            fontsize=7.5, color="#555555",
            ha="right", va="center",
            style="italic",
            arrowprops=dict(arrowstyle="-|>", color="#999999",
                            lw=1.0, mutation_scale=9))

# --- "activation space R^d" corner label ---
ax.text(0.98, 0.98,
        r"activation space $\mathbb{R}^{d}$",
        transform=ax.transAxes,
        fontsize=7, color="#9A9A9A",
        ha="right", va="top", style="italic")

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.12, 0.98)
ax.set_ylim(0.03, 1.02)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.set_xlabel("")
ax.set_ylabel("")

plt.tight_layout()

out = _HERE / "background_emotion_vector"
fig.savefig(out.with_suffix(".png"), dpi=300)
fig.savefig(out.with_suffix(".pdf"))
print(f"Saved {out}.png  and  {out}.pdf")
