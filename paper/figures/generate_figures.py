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
        print(f"  [{axis_label}] null std={null_std:.2f}  d={actual:.2f}  z={z_score:.1f}  p<{p_val:.3f}  exceed={n_exceed}/{len(null)}")
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

def fig5_cross_model(
    metrics_paths: list[Path],
    model_labels: list[str],
    log_paths: list[Path] | None = None,
) -> None:
    """Grouped bar chart: warmth/competence CV, Cohen's d, and cos(W,C) across models.

    Parameters
    ----------
    metrics_paths:
        One probe_metrics_<label>.csv per model (two rows: warmth + competence).
    model_labels:
        Display labels matching metrics_paths.
    log_paths:
        One validate_probes_*.json per model (same order as metrics_paths).
        Required for the cos(W,C) panel. Pass via --logs on CLI.
    """
    import json as _json

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

    # Read cos(W,C) from JSON logs if provided.
    cos_vals: list[float | None] = []
    if log_paths:
        for lp in log_paths:
            try:
                d = _json.loads(lp.read_text(encoding="utf-8"))
                cos_vals.append(float(d["axis_cosine"]))
            except Exception:
                cos_vals.append(None)
    else:
        cos_vals = [None] * len(records)

    models = [r["model"] for r in records]
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1 — 5-fold CV accuracy
    ax = axes[0]
    ax.bar(x - width / 2, [r["warmth_cv"] for r in records], width, label="Warmth", color="#4682B4", alpha=0.85)
    ax.bar(x + width / 2, [r["competence_cv"] for r in records], width, label="Competence", color="#E07B39", alpha=0.85)
    ax.axhline(0.8, color="green", linestyle="--", linewidth=1, label="threshold (0.80)")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, label="chance (0.50)")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("Probe accuracy\n(5-fold CV)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=8)

    # Panel 2 — Cohen's d
    ax = axes[1]
    ax.bar(x - width / 2, [r["warmth_d"] for r in records], width, color="#4682B4", alpha=0.85, label="Warmth")
    ax.bar(x + width / 2, [r["competence_d"] for r in records], width, color="#E07B39", alpha=0.85, label="Competence")
    ax.axhline(0.8, color="green", linestyle="--", linewidth=1, label="large effect (0.80)")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Effect size\n(Cohen's d)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(fontsize=8)

    # Panel 3 — cos(W, C) per model (valence overlap indicator).
    ax = axes[2]
    if all(v is not None for v in cos_vals):
        bars = ax.bar(x, cos_vals, color="#9B59B6", alpha=0.85)
        ax.axhline(0.3, color="red", linestyle="--", linewidth=1, label="|cos| = 0.30 target")
        # Annotate bars with exact values.
        for bar, val in zip(bars, cos_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("cos(warmth_vec, competence_vec)")
        ax.set_title("Axis geometric overlap\n(valence entanglement)")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.legend(fontsize=8)
    else:
        ax.set_visible(False)
        print("  [fig5] log_paths not provided; cos(W,C) panel omitted. Pass --logs to enable it.")

    fig.suptitle("Cross-model warmth & competence probe comparison (200 concept stories)", fontsize=11)
    fig.tight_layout()
    save("fig5_cross_model")


# ---------------------------------------------------------------------------
# Figure 6 — Cross-model per-story agreement (Spearman heatmaps)
# ---------------------------------------------------------------------------

def fig6_cross_model_story_agreement(
    vec_dirs: list[Path],
    model_labels: list[str],
) -> None:
    """Two 3×3 Spearman-ρ heatmaps: warmth and competence per-story projections.

    For each model, project all 200 stories onto that model's unit warmth_vec (and
    competence_vec). The file order of X_<cond>.npy is the deterministic JSONL order,
    so row i corresponds to the same story across all models.  High ρ between models
    means the *same stories* are ranked the same → shared underlying construct.

    Each 50-story condition is projected and the full 200-story projection vector is
    assembled in JSONL order: high_warmth (0–49), low_warmth (50–99),
    high_competence (100–149), low_competence (150–199).
    """
    from scipy.stats import spearmanr

    def load_model_vecs(vec_dir: Path) -> dict:
        d: dict = {}
        for cond in CONDITIONS:
            d[cond] = np.load(vec_dir / f"X_{cond}.npy").astype(np.float64)
        d["warmth_vec"] = np.load(vec_dir / "warmth_vec.npy").astype(np.float64)
        d["competence_vec"] = np.load(vec_dir / "competence_vec.npy").astype(np.float64)
        return d

    n_models = len(vec_dirs)
    # Build per-model full projection arrays (200 stories, same order as JSONL).
    warmth_projs: list[np.ndarray] = []
    comp_projs: list[np.ndarray] = []

    for vd in vec_dirs:
        d = load_model_vecs(vd)
        wv = unit(d["warmth_vec"])
        cv = unit(d["competence_vec"])
        # Stack in JSONL order: high_warmth, low_warmth, high_competence, low_competence.
        all_X = np.concatenate([d[c] for c in CONDITIONS], axis=0)  # [200, d_model]
        warmth_projs.append(all_X @ wv)
        comp_projs.append(all_X @ cv)

    # Compute Spearman ρ matrices.
    def spearman_matrix(projs: list[np.ndarray]) -> np.ndarray:
        mat = np.ones((n_models, n_models))
        for i in range(n_models):
            for j in range(i + 1, n_models):
                rho, _ = spearmanr(projs[i], projs[j])
                mat[i, j] = mat[j, i] = float(rho)
        return mat

    w_mat = spearman_matrix(warmth_projs)
    c_mat = spearman_matrix(comp_projs)

    print("  [fig6] Warmth Spearman matrix:")
    for i, lbl in enumerate(model_labels):
        row_str = "  ".join(f"{w_mat[i, j]:.3f}" for j in range(n_models))
        print(f"    {lbl:20s} {row_str}")
    print("  [fig6] Competence Spearman matrix:")
    for i, lbl in enumerate(model_labels):
        row_str = "  ".join(f"{c_mat[i, j]:.3f}" for j in range(n_models))
        print(f"    {lbl:20s} {row_str}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mat, title in [
        (axes[0], w_mat, "Warmth projection\n(Spearman ρ, 200 stories)"),
        (axes[1], c_mat, "Competence projection\n(Spearman ρ, 200 stories)"),
    ]:
        sns.heatmap(
            mat,
            ax=ax,
            cmap="Blues",
            vmin=0, vmax=1,
            annot=True, fmt=".3f",
            annot_kws={"size": 12, "weight": "bold"},
            square=True,
            linewidths=1.5,
            linecolor="white",
            xticklabels=model_labels,
            yticklabels=model_labels,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=11, pad=10)

    fig.suptitle(
        "Per-story ranking agreement across models\n"
        "(High ρ = models rank the same stories the same way → shared construct)",
        fontsize=10,
    )
    fig.tight_layout()
    save("fig6_cross_model_story_agreement")


# ---------------------------------------------------------------------------
# Figure 7 — Same-story, three-model z-scored coordinate demo
# ---------------------------------------------------------------------------

def fig7_same_story_demo(
    vec_dirs: list[Path],
    model_labels: list[str],
    stories_jsonl: Path | None = None,
) -> None:
    """Scatter: warmth-z vs competence-z for the same 6 exemplar stories across 3 models.

    For each model, z-score ALL 200 story projections within that model's distribution.
    Then plot the (warmth-z, competence-z) coordinate of ~6 extreme stories.
    Same story → similar coordinate across models = convergent placement.

    Story selection: most extreme per condition (top/bottom 1 on each axis) gives 4;
    add 2 cross-axis sanity cases (high-warmth/mid-competence, high-competence/mid-warmth).
    Stories are identified by their within-condition rank so no text lookup is needed
    (text labels are optional if stories_jsonl is provided).
    """
    def load_model_projs(vec_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        d: dict = {}
        for cond in CONDITIONS:
            d[cond] = np.load(vec_dir / f"X_{cond}.npy").astype(np.float64)
        wv = unit(np.load(vec_dir / "warmth_vec.npy").astype(np.float64))
        cv = unit(np.load(vec_dir / "competence_vec.npy").astype(np.float64))
        all_X = np.concatenate([d[c] for c in CONDITIONS], axis=0)  # [200, d_model]
        return all_X @ wv, all_X @ cv  # raw projections, 200 entries each

    # Collect projections per model.
    all_w_raw: list[np.ndarray] = []
    all_c_raw: list[np.ndarray] = []
    for vd in vec_dirs:
        wp, cp = load_model_projs(vd)
        all_w_raw.append(wp)
        all_c_raw.append(cp)

    # Z-score within each model (over all 200 stories).
    all_w_z = [(wp - wp.mean()) / (wp.std() + 1e-12) for wp in all_w_raw]
    all_c_z = [(cp - cp.mean()) / (cp.std() + 1e-12) for cp in all_c_raw]

    # Select exemplar stories by condition rank in the FIRST model (Gemma as reference).
    # Indices in the stacked array: high_warmth=0..49, low_warmth=50..99,
    # high_competence=100..149, low_competence=150..199.
    w0 = all_w_z[0]
    c0 = all_c_z[0]

    # Within each 50-item block, find the story with the most extreme target-axis value.
    exemplar_indices: list[int] = []
    exemplar_names: list[str] = []

    hw_block = np.arange(0, 50)
    lw_block = np.arange(50, 100)
    hc_block = np.arange(100, 150)
    lc_block = np.arange(150, 200)

    # Extreme warmth and competence exemplars (top/bottom on target axis).
    top_hw_idx  = hw_block[w0[hw_block].argmax()]
    bot_lw_idx  = lw_block[w0[lw_block].argmin()]
    top_hc_idx  = hc_block[c0[hc_block].argmax()]
    bot_lc_idx  = lc_block[c0[lc_block].argmin()]
    # Cross-axis sanity: high-warmth story with mid competence (|c-z| smallest in hw block).
    xax_w_idx = hw_block[np.abs(c0[hw_block]).argmin()]
    # High-competence story with mid warmth.
    xax_c_idx = hc_block[np.abs(w0[hc_block]).argmin()]

    story_specs: list[tuple[int, str, str]] = [
        (top_hw_idx,  "Strongest warm",        "#2166AC"),
        (bot_lw_idx,  "Strongest cold",         "#D73027"),
        (top_hc_idx,  "Strongest competent",    "#1A9850"),
        (bot_lc_idx,  "Weakest competent",      "#D95F02"),
        (xax_w_idx,   "Warm / mid-competence",  "#7FCDBB"),
        (xax_c_idx,   "Competent / mid-warmth", "#FD8D3C"),
    ]

    # Optionally load story text snippets for tooltip-like labels.
    story_texts: dict[int, str] = {}
    if stories_jsonl and stories_jsonl.exists():
        import json as _json
        all_texts: list[str] = []
        with stories_jsonl.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_texts.append(_json.loads(line)["text"][:60] + "…")
        for idx, _, _ in story_specs:
            if idx < len(all_texts):
                story_texts[idx] = all_texts[idx]

    n_models = len(vec_dirs)
    markers = ["o", "s", "D"]
    model_colors = ["#333333", "#666666", "#999999"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0, color="lightgray", linewidth=0.7)
    ax.axvline(0, color="lightgray", linewidth=0.7)

    # Plot each story × each model.
    for i_m in range(n_models):
        for idx, name, color in story_specs:
            wx = all_w_z[i_m][idx]
            cy = all_c_z[i_m][idx]
            ax.scatter(wx, cy, marker=markers[i_m], color=color, s=90,
                       edgecolors="white", linewidths=0.6, zorder=3)

    # Connect the same story across models with thin lines.
    for idx, name, color in story_specs:
        xs = [all_w_z[i_m][idx] for i_m in range(n_models)]
        ys = [all_c_z[i_m][idx] for i_m in range(n_models)]
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.6, zorder=2)
        # Label at the mean position.
        ax.text(float(np.mean(xs)) + 0.05, float(np.mean(ys)) + 0.05,
                name, fontsize=7.5, color=color, va="bottom")

    # Legend: model markers.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=markers[i], color="gray", linestyle="None",
               markersize=8, label=model_labels[i])
        for i in range(n_models)
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("Warmth projection (z-score within model)")
    ax.set_ylabel("Competence projection (z-score within model)")
    ax.set_title(
        "Same stories, three models: convergent placement\n"
        "(Lines connect the same story across models; shapes = models)"
    )
    fig.tight_layout()
    save("fig7_same_story_demo")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global VEC_DIR, OUT_DIR  # allow reassignment from CLI

    parser = argparse.ArgumentParser(description="Generate presentation figures.")
    parser.add_argument(
        "--fig", default="all",
        help="Figure(s) to generate: 1-7, comma-separated, or 'all' (runs 1-4 only).",
    )
    parser.add_argument(
        "--vec-dir",
        default=None,
        help="Path to concept_vectors directory for single-model figures (1-4). "
             "Default: data/processed/concept_vectors.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for figures (default: paper/figures).",
    )
    # fig5 args
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated paths to probe_metrics*.csv files (required for --fig 5).",
    )
    parser.add_argument(
        "--logs",
        default=None,
        help="Comma-separated paths to validate_probes_*.json files; same order as "
             "--metrics. Enables the cos(W,C) panel in fig5.",
    )
    # shared multi-model args (fig5, fig6, fig7)
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated model labels (required for --fig 5/6/7).",
    )
    parser.add_argument(
        "--vec-dirs",
        default=None,
        help="Comma-separated concept_vectors directories, one per model "
             "(required for --fig 6/7). E.g. "
             "data/processed/concept_vectors,"
             "data/processed/concept_vectors_qwen3_14b,"
             "data/processed/concept_vectors_llama31_8b",
    )
    parser.add_argument(
        "--stories",
        default=None,
        help="Path to concept_stories.jsonl for fig7 text labels (optional).",
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

    # Parse shared multi-model args once.
    model_labels: list[str] = [lb.strip() for lb in args.labels.split(",")] if args.labels else []

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
        if not args.metrics or not model_labels:
            parser.error("--fig 5 requires --metrics and --labels")
        metrics_paths = [Path(p.strip()) for p in args.metrics.split(",")]
        if len(metrics_paths) != len(model_labels):
            parser.error("--metrics and --labels must have the same number of entries")
        log_paths: list[Path] | None = None
        if args.logs:
            log_paths = [Path(p.strip()) for p in args.logs.split(",")]
            if len(log_paths) != len(model_labels):
                parser.error("--logs and --labels must have the same number of entries")
        fig5_cross_model(metrics_paths, model_labels, log_paths=log_paths)

    if 6 in selected:
        print("Figure 6: cross-model story agreement (Spearman) …")
        if not args.vec_dirs or not model_labels:
            parser.error("--fig 6 requires --vec-dirs and --labels")
        vec_dirs = [Path(p.strip()) for p in args.vec_dirs.split(",")]
        if len(vec_dirs) != len(model_labels):
            parser.error("--vec-dirs and --labels must have the same number of entries")
        fig6_cross_model_story_agreement(vec_dirs, model_labels)

    if 7 in selected:
        print("Figure 7: same-story three-model demo …")
        if not args.vec_dirs or not model_labels:
            parser.error("--fig 7 requires --vec-dirs and --labels")
        vec_dirs = [Path(p.strip()) for p in args.vec_dirs.split(",")]
        if len(vec_dirs) != len(model_labels):
            parser.error("--vec-dirs and --labels must have the same number of entries")
        stories_jsonl = Path(args.stories) if args.stories else None
        fig7_same_story_demo(vec_dirs, model_labels, stories_jsonl=stories_jsonl)

    print("Done.")


if __name__ == "__main__":
    main()
