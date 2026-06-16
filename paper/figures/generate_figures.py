"""
Generate presentation-quality figures for the probe findings report.

Usage:
    python paper/figures/generate_figures.py --fig all
    python paper/figures/generate_figures.py --fig 1
    python paper/figures/generate_figures.py --fig 2,3

Outputs: paper/figures/figN_*.{pdf,png}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

# Ensure repo root is on path when run from anywhere
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import style as _style  # noqa: E402 — sibling file

OUT_DIR = Path(__file__).parent
VEC_DIR = ROOT / "data" / "processed" / "concept_vectors"

CONDITIONS = ["high_warmth", "low_warmth", "high_competence", "low_competence"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> dict:
    data = {}
    for cond in CONDITIONS:
        data[cond] = np.load(VEC_DIR / f"X_{cond}.npy").astype(np.float64)
    data["warmth_vec"] = np.load(VEC_DIR / "warmth_vec.npy").astype(np.float64)
    data["competence_vec"] = np.load(VEC_DIR / "competence_vec.npy").astype(np.float64)
    return data


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = float(np.sqrt((a.var() + b.var()) / 2.0) + 1e-12)
    return float((a.mean() - b.mean()) / pooled_std)


def save(name: str) -> None:
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        plt.savefig(path, format=ext)
        print(f"  saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 1 — Joint density of story representations
# ---------------------------------------------------------------------------

def fig1_joint_density(data: dict) -> None:
    wu = unit(data["warmth_vec"])
    cu = unit(data["competence_vec"])

    # Project all stories onto both axes
    all_w, all_c = [], []
    cond_projs = {}
    for cond in CONDITIONS:
        w = data[cond] @ wu
        c = data[cond] @ cu
        cond_projs[cond] = (w, c)
        all_w.append(w)
        all_c.append(c)

    all_w = np.concatenate(all_w)
    all_c = np.concatenate(all_c)
    mu_w, sigma_w = all_w.mean(), all_w.std()
    mu_c, sigma_c = all_c.mean(), all_c.std()

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.04, wspace=0.04,
    )
    ax_joint  = fig.add_subplot(gs[1, 0])
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    for cond in CONDITIONS:
        w_z = (cond_projs[cond][0] - mu_w) / sigma_w
        c_z = (cond_projs[cond][1] - mu_c) / sigma_c
        color = _style.PALETTE[cond]
        label = _style.LABELS[cond]

        sns.kdeplot(
            x=w_z, y=c_z,
            ax=ax_joint,
            color=color,
            levels=5,
            linewidths=1.6,
            label=label,
        )
        sns.kdeplot(x=w_z, ax=ax_marg_x, color=color, fill=True, alpha=0.25, linewidth=1.2)
        sns.kdeplot(y=c_z, ax=ax_marg_y, color=color, fill=True, alpha=0.25, linewidth=1.2)

    ax_joint.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_joint.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_joint.set_xlabel("Warmth projection (z-score)")
    ax_joint.set_ylabel("Competence projection (z-score)")

    # Manual legend proxies — seaborn kdeplot doesn't register labels on JointGrid
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=_style.PALETTE[c], linewidth=2, label=_style.LABELS[c])
        for c in CONDITIONS
    ]
    ax_joint.legend(handles=legend_handles, loc="lower right", framealpha=0.9, fontsize=9)

    ax_marg_x.set_ylabel("")
    ax_marg_y.set_xlabel("")
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    for ax in (ax_marg_x, ax_marg_y):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    save("fig1_joint_density")


# ---------------------------------------------------------------------------
# Figure 2 — Random-direction baseline
# ---------------------------------------------------------------------------

def fig2_random_baseline(data: dict, n_random: int = 1000, seed: int = 20260527) -> None:
    rng = np.random.default_rng(seed)
    d = data["warmth_vec"].shape[0]

    wu = unit(data["warmth_vec"])
    cu = unit(data["competence_vec"])

    actual_d_warmth = cohens_d(
        data["high_warmth"] @ wu,
        data["low_warmth"] @ wu,
    )
    actual_d_competence = cohens_d(
        data["high_competence"] @ cu,
        data["low_competence"] @ cu,
    )

    rand_vecs = rng.standard_normal((n_random, d))
    rand_vecs /= np.linalg.norm(rand_vecs, axis=1, keepdims=True)

    null_warmth     = np.array([cohens_d(data["high_warmth"] @ r, data["low_warmth"] @ r) for r in rand_vecs])
    null_competence = np.array([cohens_d(data["high_competence"] @ r, data["low_competence"] @ r) for r in rand_vecs])

    # Take absolute values so direction doesn't matter
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)

    for ax, null, actual, axis_label in [
        (axes[0], null_warmth,     actual_d_warmth,     "warmth"),
        (axes[1], null_competence, actual_d_competence, "competence"),
    ]:
        x_min = min(null.min(), 0) * 1.05
        x_max = max(actual * 1.15, null.max() * 1.05)
        xs = np.linspace(x_min, x_max, 500)
        kde = gaussian_kde(null, bw_method=0.3)

        ax.fill_between(xs, kde(xs), alpha=0.25, color="#9CA3AF")
        ax.plot(xs, kde(xs), color="#6B7280", linewidth=1.6, label="Random directions")
        ax.axvline(0, color="#9CA3AF", linewidth=0.8, linestyle=":")

        # Our vector
        ax.axvline(actual, color="#DC2626", linewidth=2.2, linestyle="-")
        ax.annotate(
            f"Our direction\n$d$ = {actual:.2f}",
            xy=(actual, 0.05),
            xytext=(actual - 1.55, 0.4),
            arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2),
            color="#DC2626",
            fontsize=10,
        )
        null_std = float(np.std(null))
        z_score = actual / null_std
        n_exceed = int((null >= actual).sum())
        p_val = max(n_exceed, 1) / len(null)
        print(f"  [{axis_label}] null σ={null_std:.2f}  d={actual:.2f}  z={z_score:.1f}  p<{p_val:.3f}  exceed={n_exceed}/{len(null)}")
        ax.set_xlabel("Cohen's $d$")
        ax.set_ylabel("Density" if ax is axes[0] else "")
        ax.set_title(f"Null distribution — {axis_label} axis")

    fig.tight_layout()
    save("fig2_random_baseline")


# ---------------------------------------------------------------------------
# Figure 3 — Lorenz / cumulative norm concentration
# ---------------------------------------------------------------------------

def fig3_lorenz_concentration(data: dict) -> None:
    d = data["warmth_vec"].shape[0]
    dims = np.arange(1, d + 1)

    def lorenz(v: np.ndarray):
        sq = v ** 2
        sq_sorted = np.sort(sq)[::-1]
        return np.cumsum(sq_sorted) / sq_sorted.sum()

    warmth_curve = lorenz(data["warmth_vec"])
    competence_curve = lorenz(data["competence_vec"])
    uniform_curve = dims / d

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(dims, uniform_curve, color="#9CA3AF", linestyle="--", linewidth=1.2,
            label="Uniform baseline (signal spread evenly)")
    ax.plot(dims, warmth_curve, color=_style.PALETTE["high_warmth"], linewidth=2.0,
            label="Warmth direction")
    ax.plot(dims, competence_curve, color=_style.PALETTE["high_competence"], linewidth=2.0,
            linestyle=(0, (5, 2)),
            label="Competence direction")

    # Annotations for warmth curve
    for frac, offset_x_factor in [(0.50, 2.5), (0.80, 2.0), (0.95, 1.5)]:
        idx = int(np.searchsorted(warmth_curve, frac))
        dim_count = idx + 1
        ax.annotate(
            f"Top {dim_count} dims = {int(frac*100)}%",
            xy=(dim_count, frac),
            xytext=(dim_count * offset_x_factor, frac - 0.07),
            arrowprops=dict(arrowstyle="->", color="#374151", lw=0.9),
            fontsize=9, color="#374151",
        )

    ax.set_xscale("log")
    ax.set_xlim(1, d)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Number of dimensions (sorted by contribution, log scale)")
    ax.set_ylabel("Cumulative fraction of vector norm²")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    save("fig3_lorenz_concentration")


# ---------------------------------------------------------------------------
# Figure 4 — Axis geometry vs behavioural independence
# ---------------------------------------------------------------------------

def fig4_axis_geometry() -> None:
    labels = ["Warmth", "Competence"]

    cosine_matrix = np.array([[1.000, 0.749],
                               [0.749, 1.000]])

    cv_matrix = np.array([[1.000, 0.500],
                           [0.500, 1.000]])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    # Left — cosine similarity
    sns.heatmap(
        cosine_matrix,
        ax=axes[0],
        cmap="Blues",
        vmin=0, vmax=1,
        annot=True, fmt=".3f",
        annot_kws={"size": 13, "weight": "bold"},
        square=True,
        linewidths=1.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Vector geometry\n(cosine similarity)", fontsize=11, pad=10)
    axes[0].set_xlabel("Direction vector")
    axes[0].set_ylabel("Direction vector")

    # Right — behavioural discriminability
    sns.heatmap(
        cv_matrix,
        ax=axes[1],
        cmap="RdYlGn",
        vmin=0.4, vmax=1.0,
        annot=True, fmt=".2f",
        annot_kws={"size": 13, "weight": "bold"},
        square=True,
        linewidths=1.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"shrink": 0.8},
    )
    axes[1].set_title("Behavioural discriminability\n(5-fold CV accuracy)", fontsize=11, pad=10)
    axes[1].set_xlabel("Test condition")
    axes[1].set_ylabel("Probe direction")

    fig.suptitle(
        "High geometric similarity, full behavioural independence",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    save("fig4_axis_geometry")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation figures.")
    parser.add_argument(
        "--fig", default="all",
        help="Figure(s) to generate: 1, 2, 3, 4, or comma-separated, or 'all'",
    )
    args = parser.parse_args()

    _style.apply()

    if args.fig == "all":
        selected = {1, 2, 3, 4}
    else:
        selected = {int(x.strip()) for x in args.fig.split(",")}

    if selected & {1, 2, 3}:
        print("Loading activation data …")
        data = load_data()
    else:
        data = None

    if 1 in selected:
        print("Figure 1: joint density …")
        fig1_joint_density(data)

    if 2 in selected:
        print("Figure 2: random baseline …")
        fig2_random_baseline(data)

    if 3 in selected:
        print("Figure 3: Lorenz concentration …")
        fig3_lorenz_concentration(data)

    if 4 in selected:
        print("Figure 4: axis geometry heatmap …")
        fig4_axis_geometry()

    print("Done.")


if __name__ == "__main__":
    main()
