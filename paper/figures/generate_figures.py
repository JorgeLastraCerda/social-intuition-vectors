"""
Generate presentation-quality figures for the probe findings report.

Usage:
    python paper/figures/generate_figures.py --fig all
    python paper/figures/generate_figures.py --fig 1
    python paper/figures/generate_figures.py --fig 2,3
    python paper/figures/generate_figures.py --fig 1,2,3 \
        --vec-dir data/processed/concept_vectors_qwen3_14b \
        --out-dir paper/figures/qwen3_14b

    # Cross-model comparison (fig5); requires one metrics CSV per model:
    python paper/figures/generate_figures.py --fig 5 \
        --metrics results/tables/probe_metrics.csv,results/tables/probe_metrics_qwen3_14b.csv,results/tables/probe_metrics_llama31_8b.csv \
        --labels Gemma-3-12B,Qwen3-14B,Llama-3.1-8B

Outputs: <out-dir>/figN_*.{pdf,png}
"""
from __future__ import annotations

import argparse
import csv
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

# Module-level defaults — overridden at runtime by parse_args().
_DEFAULT_VEC_DIR = ROOT / "data" / "processed" / "concept_vectors"
_DEFAULT_OUT_DIR = Path(__file__).parent

CONDITIONS = ["high_warmth", "low_warmth", "high_competence", "low_competence"]

# Runtime-resolved directories (set in main() from CLI args).
VEC_DIR: Path = _DEFAULT_VEC_DIR
OUT_DIR: Path = _DEFAULT_OUT_DIR


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
# Figure 4 — Axis geometry vs behavioural discriminability
# ---------------------------------------------------------------------------

def fig4_axis_geometry(data: dict) -> None:
    """Heatmap of vector cosine similarity and 5-fold CV discriminability.

    Cosine(W,C) is computed from the loaded vectors so it stays accurate
    regardless of which model's concept_vectors directory is used.
    Cross-axis CV values are computed via a 1-D logistic regression on the
    projection scores (same approach as validate_probes.py:cross_axis_accuracy).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    seed = 20260527
    labels = ["Warmth", "Competence"]

    wv = unit(data["warmth_vec"])
    cv = unit(data["competence_vec"])
    axis_cosine = float(np.dot(wv, cv))

    cosine_matrix = np.array([[1.0, axis_cosine],
                               [axis_cosine, 1.0]])

    def _cv_1d(X_pos, X_neg, direction):
        proj_pos = X_pos @ direction
        proj_neg = X_neg @ direction
        X_proj = np.concatenate([proj_pos, proj_neg]).reshape(-1, 1)
        y = np.array([1] * len(proj_pos) + [0] * len(proj_neg))
        lr = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        cv_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return float(cross_val_score(lr, X_proj, y, cv=cv_kf, scoring="accuracy").mean())

    # On-diagonal: warmth-probe on warmth stories, competence-probe on competence stories.
    w_cv = _cv_1d(data["high_warmth"], data["low_warmth"], wv)
    c_cv = _cv_1d(data["high_competence"], data["low_competence"], cv)
    # Off-diagonal: cross-axis probes.
    wv_on_c = _cv_1d(data["high_competence"], data["low_competence"], wv)
    cv_on_w = _cv_1d(data["high_warmth"], data["low_warmth"], cv)

    cv_matrix = np.array([[w_cv, cv_on_w],
                           [wv_on_c, c_cv]])

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

    # Right — behavioural discriminability (5-fold CV, 1-D projection)
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
    axes[1].set_title("Behavioural discriminability\n(5-fold CV accuracy, 1-D projection)", fontsize=11, pad=10)
    axes[1].set_xlabel("Test condition")
    axes[1].set_ylabel("Probe direction")

    fig.suptitle(
        f"Axis geometry (cos={axis_cosine:.3f}) and behavioural discriminability",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    save("fig4_axis_geometry")


# ---------------------------------------------------------------------------
# Figure 5 — Cross-model comparison bar chart
# ---------------------------------------------------------------------------

def fig5_cross_model(metrics_paths: list[Path], model_labels: list[str]) -> None:
    """Grouped bar chart comparing warmth/competence CV, Cohen's d, and cos(W,C)
    across models.  Reads one probe_metrics_<label>.csv per model.

    Each CSV must have two rows (axis=warmth, axis=competence) with at minimum the
    columns: axis, cv_mean, cohens_d.  cos(W,C) is read from an optional
    validate_probes JSON log in the same directory; if absent it is omitted.
    """
    records: list[dict] = []
    for path, label in zip(metrics_paths, model_labels):
        row: dict = {"model": label}
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                axis = r["axis"]
                row[f"{axis}_cv"] = float(r["cv_mean"])
                row[f"{axis}_d"] = float(r["cohens_d"])
        records.append(row)

    models = [r["model"] for r in records]
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1 — 5-fold CV accuracy
    ax = axes[0]
    ax.bar(x - width / 2, [r["warmth_cv"] for r in records], width, label="Warmth", color="#4682B4", alpha=0.85)
    ax.bar(x + width / 2, [r["competence_cv"] for r in records], width, label="Competence", color="#E07B39", alpha=0.85)
    ax.axhline(0.8, color="green", linestyle="--", linewidth=1, label="threshold (0.80)")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, label="chance (0.50)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("Probe accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=8)

    # Panel 2 — Cohen's d
    ax = axes[1]
    ax.bar(x - width / 2, [r["warmth_d"] for r in records], width, color="#4682B4", alpha=0.85, label="Warmth")
    ax.bar(x + width / 2, [r["competence_d"] for r in records], width, color="#E07B39", alpha=0.85, label="Competence")
    ax.axhline(0.8, color="green", linestyle="--", linewidth=1, label="large effect (0.80)")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect size")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=8)

    # Panel 3 — cos(W, C) per model (valence overlap)
    # Try to read from a sidecar JSON produced by validate_probes.py.
    cos_vals: list[float | None] = []
    for path in metrics_paths:
        # Look for the most recent validate_probes_*.json in results/logs/
        logs_dir = ROOT / "results" / "logs"
        candidates = sorted(logs_dir.glob("validate_probes_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        found = None
        # Match by the model name in the JSON meta field
        for c in candidates:
            try:
                import json
                data = json.loads(c.read_text(encoding="utf-8"))
                if data.get("axis_cosine") is not None:
                    # Identify by matching the metrics CSV label to the path stem
                    # Heuristic: the CSV name contains the label, so check meta.model
                    # against known model IDs; accept the first unambiguous match.
                    # If uncertain, just return None and skip the panel row.
                    pass
            except Exception:
                pass
        cos_vals.append(found)

    # If we can read cos values from the CSVs directly (validate_probes writes them
    # to the JSON, not the CSV), skip drawing the third panel and note it.
    # The panel is only drawn when all values are available.
    if all(v is not None for v in cos_vals):
        ax = axes[2]
        ax.bar(x, cos_vals, color="#9B59B6", alpha=0.85)
        ax.axhline(0.3, color="red", linestyle="--", linewidth=1, label="|cos| = 0.30 target")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("cos(warmth_vec, competence_vec)")
        ax.set_title("Axis overlap (valence)")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.legend(fontsize=8)
    else:
        axes[2].set_visible(False)
        print("  [fig5] cos(W,C) values not auto-detected from logs; panel 3 omitted.")
        print("  Fill in manually from validate_probes JSON logs if needed.")

    fig.suptitle("Cross-model warmth & competence probe comparison (200 concept stories)", fontsize=11)
    fig.tight_layout()
    save("fig5_cross_model")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global VEC_DIR, OUT_DIR  # allow reassignment from CLI

    parser = argparse.ArgumentParser(description="Generate presentation figures.")
    parser.add_argument(
        "--fig", default="all",
        help="Figure(s) to generate: 1, 2, 3, 4, 5, or comma-separated, or 'all'",
    )
    parser.add_argument(
        "--vec-dir",
        default=None,
        help="Path to concept_vectors directory (default: data/processed/concept_vectors). "
             "Use to point at a different model's outputs, e.g. "
             "data/processed/concept_vectors_qwen3_14b.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for figures (default: paper/figures).",
    )
    # fig5-specific args
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated paths to probe_metrics*.csv files (required for --fig 5).",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated model labels matching --metrics (required for --fig 5).",
    )
    args = parser.parse_args()

    # Resolve runtime dirs
    if args.vec_dir is not None:
        VEC_DIR = Path(args.vec_dir)
    if args.out_dir is not None:
        OUT_DIR = Path(args.out_dir)

    _style.apply()

    if args.fig == "all":
        selected = {1, 2, 3, 4}
    else:
        selected = {int(x.strip()) for x in args.fig.split(",")}

    if selected & {1, 2, 3, 4}:
        print(f"Loading activation data from {VEC_DIR} …")
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
        fig4_axis_geometry(data)

    if 5 in selected:
        print("Figure 5: cross-model comparison …")
        if not args.metrics or not args.labels:
            parser.error("--fig 5 requires --metrics and --labels")
        metrics_paths = [Path(p.strip()) for p in args.metrics.split(",")]
        model_labels = [lb.strip() for lb in args.labels.split(",")]
        if len(metrics_paths) != len(model_labels):
            parser.error("--metrics and --labels must have the same number of entries")
        fig5_cross_model(metrics_paths, model_labels)

    print("Done.")


if __name__ == "__main__":
    main()
