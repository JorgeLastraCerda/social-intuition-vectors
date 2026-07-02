"""
paper/figures/background_concept_geometry.py

Schematic figure for §2 Background: residual-stream projection showing
(1) two concept-condition clouds  (2) the mean-difference direction
v = h̄_high − h̄_low  (3) a single steering step h' = h + α v̂
that moves a marginal point across the decision boundary.

N = 50 per cloud — matches the real per-condition story count in the corpus.
SCHEMATIC — illustrative synthetic activations, not measured data.

Outputs (same-basename triplet):
  paper/figures/background_concept_geometry.png
  paper/figures/background_concept_geometry.pdf
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

rng = np.random.default_rng(42)

# ── synthetic clouds — N = 50 matching real per-condition story count ─────────
N = 50

mu_high = np.array([ 2.2,  2.0])
mu_low  = np.array([-2.2, -1.7])
std     = 0.50

pts_high = rng.normal(mu_high, std, (N, 2))
pts_low  = rng.normal(mu_low,  std, (N, 2))

# mean-difference vector and unit direction
v     = mu_high - mu_low           # [4.4, 3.7]
v_hat = v / np.linalg.norm(v)     # ≈ [0.765, 0.644]

midpoint = (mu_high + mu_low) / 2  # [0.0, 0.15]

# decision boundary: perpendicular to v̂ at midpoint
perp = np.array([-v_hat[1], v_hat[0]])
t    = np.linspace(-2.6, 2.6, 300)
bx   = midpoint[0] + t * perp[0]
by   = midpoint[1] + t * perp[1]

# ── steering example ──────────────────────────────────────────────────────────
# marginal low-warmth point just inside the low-warmth side of the boundary
# back to the 0.85*perp offset — separates from the blue arrow without going too high
src   = np.array([-0.25, 0.05]) - 1.0 * perp     # ≈ [0.39, -0.72] — below main arrow
alpha = 1.65
dst   = src + alpha * v_hat   # ≈ [1.66, 0.34]

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.3, 2.7))

# --- story clouds (50 dots per cloud, small crisp markers) ---
ax.scatter(*pts_high.T,
           color=style.PALETTE["high_warmth"], alpha=0.65, s=20, zorder=3,
           linewidths=0.3, edgecolors="white",
           label=r"High-warmth stories ($n{=}50$)")
ax.scatter(*pts_low.T,
           color=style.PALETTE["low_warmth"],  alpha=0.65, s=20, zorder=3,
           linewidths=0.3, edgecolors="white",
           label=r"Low-warmth stories ($n{=}50$)")

# --- centroid markers (×) anchoring the mean-difference arrow ---
for mu in (mu_high, mu_low):
    ax.scatter(*mu, color=style.ARROW_WARMTH, s=70, marker="X", zorder=6,
               edgecolors="white", linewidths=0.5)

# --- mean-difference arrow (µ_low → µ_high) ---
ax.annotate("", xy=mu_high, xytext=mu_low,
            arrowprops=dict(arrowstyle="-|>",
                            color=style.ARROW_WARMTH,
                            lw=2.0, mutation_scale=13),
            zorder=5)

# --- v label — lower-right quadrant, clear of all other elements ---
ax.text(2.4, -1.75,
        r"$v = \bar{h}_{\mathrm{high}} - \bar{h}_{\mathrm{low}}$",
        fontsize=7.5, color=style.ARROW_WARMTH,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor="none", alpha=0.85))

# --- decision boundary (dashed) ---
ax.plot(bx, by, color="#BBBBBB", linewidth=0.85, linestyle="--", zorder=2)

# boundary label near the upper-left end of the boundary line
bnd_lbl_x = midpoint[0] + 2.3 * perp[0] - 0.1   # ≈ -1.58
bnd_lbl_y = midpoint[1] + 2.3 * perp[1] + 0.15  # ≈  1.93
ax.text(bnd_lbl_x, bnd_lbl_y, "decision\nboundary",
        fontsize=6.5, color="#999999", ha="center", va="bottom",
        linespacing=1.2)

# --- pre-steering point h (marginal low-warmth example) ---
ax.scatter(*src, color=style.PALETTE["low_warmth"], s=52, zorder=7,
           edgecolors="#333333", linewidths=0.9)
# label below-left of the point
ax.text(src[0] - 0.15, src[1] - 0.32, r"$h$",
        fontsize=9, ha="right", va="top", color="#333333")

# --- steering arrow (dashed red) ---
ax.annotate("", xy=dst, xytext=src,
            arrowprops=dict(arrowstyle="-|>",
                            color="#C0392B", lw=1.7,
                            linestyle="dashed", mutation_scale=11,
                            shrinkB=5),
            zorder=8)

# --- post-steering point h' ---
ax.scatter(*dst, color="#C0392B", s=60, marker="*", zorder=9,
           edgecolors="#800000", linewidths=0.5)
# h' label: to the right of the star
ax.text(dst[0] + 0.18, dst[1] + 0.05,
        r"$h' \!=\! h + \alpha\hat{v}$",
        fontsize=7.5, ha="left", va="center", color="#C0392B")

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel("Residual-stream projection 1", fontsize=8.5)
ax.set_ylabel("Residual-stream projection 2", fontsize=8.5)
ax.set_xlim(-4.3, 4.5)
ax.set_ylim(-3.6, 4.1)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# --- legend: upper-left, compact ---
ax.legend(loc="upper left", fontsize=7.5, frameon=False,
          labelspacing=0.3, handletextpad=0.4,
          borderaxespad=0.55, handlelength=1.0, markerscale=1.3)

plt.tight_layout()

out = _HERE / "background_concept_geometry"
fig.savefig(out.with_suffix(".png"), dpi=300)
fig.savefig(out.with_suffix(".pdf"))
print(f"Saved {out}.png  and  {out}.pdf")
print(f"  High-warmth dots: {N}  |  Low-warmth dots: {N}")
