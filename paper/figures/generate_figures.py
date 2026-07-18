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
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde, spearmanr
from sklearn.decomposition import PCA

# Ensure repo root is on path when run from anywhere
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import style as _style  # noqa: E402 — sibling file
from src.validate_probes import projected_cv_accuracy  # noqa: E402

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
    All CV values use the scale-invariant 1-D projection metric from
    validate_probes.py.
    """
    seed = 20260527
    labels = ["Warmth", "Competence"]

    wv = unit(data["warmth_vec"])
    cv = unit(data["competence_vec"])
    axis_cosine = float(np.dot(wv, cv))

    cosine_matrix = np.array([[1.0, axis_cosine],
                               [axis_cosine, 1.0]])

    # On-diagonal: warmth-probe on warmth stories, competence-probe on competence stories.
    w_cv = projected_cv_accuracy(data["high_warmth"], data["low_warmth"], wv, seed)
    c_cv = projected_cv_accuracy(data["high_competence"], data["low_competence"], cv, seed)
    # Off-diagonal: cross-axis probes.
    wv_on_c = projected_cv_accuracy(
        data["high_competence"], data["low_competence"], wv, seed
    )
    cv_on_w = projected_cv_accuracy(
        data["high_warmth"], data["low_warmth"], cv, seed
    )

    cv_matrix = np.array([[w_cv, wv_on_c],
                          [cv_on_w, c_cv]])

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
    agreement_csv: Path | None = None,
) -> None:
    """Compare overall and within-condition story-ranking agreement."""
    from src.validate_cross_model_agreement import compute_agreement_records

    if agreement_csv is not None:
        with agreement_csv.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
    else:
        records = compute_agreement_records(vec_dirs, model_labels)

    n_models = len(model_labels)
    label_to_index = {label: index for index, label in enumerate(model_labels)}
    matrices = {
        (axis, metric): np.eye(n_models, dtype=np.float64)
        for axis in ("warmth", "competence")
        for metric in ("overall_rho", "within_condition_rho")
    }
    for record in records:
        axis = str(record["axis"])
        i = label_to_index[str(record["model_a"])]
        j = label_to_index[str(record["model_b"])]
        for metric in ("overall_rho", "within_condition_rho"):
            value = float(record[metric])
            matrices[(axis, metric)][i, j] = matrices[(axis, metric)][j, i] = value

    for axis in ("warmth", "competence"):
        print(f"  [fig6] {axis.title()} overall / within-condition Spearman:")
        for i, label in enumerate(model_labels):
            overall = "  ".join(
                f"{matrices[(axis, 'overall_rho')][i, j]:.3f}" for j in range(n_models)
            )
            within = "  ".join(
                f"{matrices[(axis, 'within_condition_rho')][i, j]:.3f}"
                for j in range(n_models)
            )
            print(f"    {label:20s} overall={overall}  within={within}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        (axes[0, 0], matrices[("warmth", "overall_rho")], "Warmth: overall ranking"),
        (axes[0, 1], matrices[("competence", "overall_rho")], "Competence: overall ranking"),
        (
            axes[1, 0],
            matrices[("warmth", "within_condition_rho")],
            "Warmth: within-condition ranking",
        ),
        (
            axes[1, 1],
            matrices[("competence", "within_condition_rho")],
            "Competence: within-condition ranking",
        ),
    ]
    for ax, mat, title in panels:
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
        ax.set_title(f"{title}\n(Spearman ρ)", fontsize=10, pad=8)

    fig.suptitle(
        "Cross-model story-ranking agreement\n"
        "Overall includes condition separation; within-condition compares story ordering",
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
        (bot_lc_idx,  "Least competent",        "#D95F02"),
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
        # Label at the mean position with a fixed display offset to avoid markers.
        ax.annotate(
            name,
            xy=(float(np.mean(xs)), float(np.mean(ys))),
            xytext=(18, 8),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            va="bottom",
        )

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
        "Six Gemma-4-12B-selected exemplar stories across three models\n"
        "(Qualitative illustration; lines connect the same story; shapes = models)"
    )
    fig.tight_layout()
    save("fig7_same_story_demo")


# ---------------------------------------------------------------------------
# Figure 8 — Layer emergence curves (topic-holdout CV and Cohen's d vs depth)
# ---------------------------------------------------------------------------

def fig8_layer_emergence(
    sweep_csv_paths: list[Path],
    model_labels: list[str],
) -> None:
    """Two-panel layer-sweep figure for representation strength and geometry.

    Panel 1 (left): Cohen's d vs layer fraction for warmth and competence axes.
    Solid lines = warmth; dotted = competence.  One color per model.
    Shows WHERE representations become strongest and whether frac=0.66 is optimal.

    Panel 2 (right): cos(warmth_vec, competence_vec) vs layer fraction.
    This compares depth-wise vector geometry across model families. It does not
    measure cross-axis classification accuracy.

    CSV columns expected (from src/layer_sweep.py):
        frac, warmth_cohens_d, comp_cohens_d, cos_wc
    """
    import csv as _csv

    # Colors: Gemma-12B light green, Gemma-27B dark teal (same family, distinct shade),
    # Qwen purple, Llama red.  Linestyles: solid, dash-dot-dot, dashed, dash-dot.
    model_colors = ["#1b7837", "#006d6d", "#762a83", "#d6604d"]
    model_ls     = ["-", (0, (3, 1, 1, 1)), "--", "-."]

    sweeps: list[dict[str, list]] = []
    for path in sweep_csv_paths:
        rows: dict[str, list] = {"frac": [], "warmth_d": [], "comp_d": [], "cos": []}
        with path.open(newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                rows["frac"].append(float(row["frac"]))
                rows["warmth_d"].append(float(row["warmth_cohens_d"]))
                rows["comp_d"].append(float(row["comp_cohens_d"]))
                rows["cos"].append(float(row["cos_wc"]))
        sweeps.append(rows)
        print(f"  [fig8] {path.name}: {len(rows['frac'])} layers")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- Panel 1: Cohen's d emergence ----
    ax = axes[0]
    for i_m, (sweep, label) in enumerate(zip(sweeps, model_labels)):
        c  = model_colors[i_m % len(model_colors)]
        ls = model_ls[i_m % len(model_ls)]
        ax.plot(sweep["frac"], sweep["warmth_d"], color=c, linestyle=ls,
                linewidth=1.8, label=f"{label} (warmth)", zorder=3)
        ax.plot(sweep["frac"], sweep["comp_d"], color=c, linestyle=":",
                linewidth=1.2, alpha=0.7, label=f"{label} (comp.)", zorder=2)

    ax.axvline(0.66, color="gray", linestyle=":", linewidth=1.2,
               label="probe layer (frac=0.66)", zorder=1)
    ax.axhline(0.8, color="green", linestyle="--", linewidth=0.8,
               label="large effect (d=0.80)", zorder=1)
    ax.set_xlabel("Normalized block depth (layer / (n_layers - 1))")
    ax.set_ylabel("Cohen's d")
    ax.set_title("Representation strength by depth\n(solid = warmth, dotted = competence)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7.5, loc="upper left", ncol=1, framealpha=0.9)

    # ---- Panel 2: cos(W,C) depth profile ----
    ax2 = axes[1]
    for i_m, (sweep, label) in enumerate(zip(sweeps, model_labels)):
        c  = model_colors[i_m % len(model_colors)]
        ls = model_ls[i_m % len(model_ls)]
        ax2.plot(sweep["frac"], sweep["cos"], color=c, linestyle=ls,
                 linewidth=1.8, label=label, zorder=3)

    ax2.axvline(0.66, color="gray", linestyle=":", linewidth=1.2,
                label="probe layer (frac=0.66)", zorder=1)
    ax2.axhspan(-0.3, 0.3, color="orange", alpha=0.08, zorder=0)
    ax2.axhline(0.3, color="orange", linestyle="--", linewidth=0.8,
                label="orthogonality bounds (±0.30)", zorder=1)
    ax2.axhline(-0.3, color="orange", linestyle="--", linewidth=0.8, zorder=1)
    ax2.axhline(0.0, color="gray", linewidth=0.6, zorder=1)
    ax2.set_xlabel("Normalized block depth (layer / (n_layers - 1))")
    ax2.set_ylabel("cos(warmth_vec, competence_vec)")
    ax2.set_title(
        "Warmth/competence axis overlap by depth\n"
        "(negative and positive alignment shown)"
    )
    ax2.set_xlim(0, 1)
    all_cos = np.concatenate([np.asarray(sweep["cos"], dtype=float) for sweep in sweeps])
    ax2.set_ylim(min(-0.3, float(all_cos.min()) - 0.05), max(0.8, float(all_cos.max()) + 0.05))
    ax2.legend(fontsize=9, loc="upper left", framealpha=0.9)

    model_scope = "one model" if len(sweeps) == 1 else f"{len(sweeps)} model variants"
    fig.suptitle(
        "Legacy-compatible layer sweep: descriptive effect size and axis geometry  "
        f"({model_scope})",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    save("fig8_layer_emergence")


# ---------------------------------------------------------------------------
# Figure 9 — Gemma Scope reconstruction and retained concept signal
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fig8b_stage3b_validation(
    sweep_csv_paths: list[Path],
    model_labels: list[str],
) -> None:
    """Enhanced Stage 3B direction, transfer, and bootstrap profiles."""
    rows_by_model = [_read_csv(path) for path in sweep_csv_paths]
    colors = ["#1b7837", "#006d6d", "#762a83"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)

    for model_index, (rows, label) in enumerate(zip(rows_by_model, model_labels)):
        color = colors[model_index % len(colors)]
        x = np.asarray([float(row["frac"]) for row in rows])
        axes[0, 0].plot(
            x, [float(row["warmth_direction_topic_cv"]) for row in rows],
            color=color, linewidth=1.8, label=f"{label} warmth",
        )
        axes[0, 0].plot(
            x, [float(row["comp_direction_topic_cv"]) for row in rows],
            color=color, linewidth=1.4, linestyle=":", label=f"{label} competence",
        )
        axes[0, 1].plot(
            x, [float(row["warmth_to_comp_topic_transfer"]) for row in rows],
            color=color, linewidth=1.8, label=f"{label} W→C",
        )
        axes[0, 1].plot(
            x, [float(row["comp_to_warmth_topic_transfer"]) for row in rows],
            color=color, linewidth=1.4, linestyle=":", label=f"{label} C→W",
        )
        for metric, linestyle, axis_name in (
            ("warmth_cohens_d", "-", "warmth"),
            ("comp_cohens_d", ":", "competence"),
        ):
            estimate = np.asarray([float(row[metric]) for row in rows])
            low = np.asarray([float(row[f"{metric}_ci_low"]) for row in rows])
            high = np.asarray([float(row[f"{metric}_ci_high"]) for row in rows])
            axes[1, 0].plot(
                x, estimate, color=color, linestyle=linestyle, linewidth=1.6,
                label=f"{label} {axis_name}",
            )
            axes[1, 0].fill_between(x, low, high, color=color, alpha=0.07)
        estimate = np.asarray([float(row["cos_wc"]) for row in rows])
        low = np.asarray([float(row["cos_wc_ci_low"]) for row in rows])
        high = np.asarray([float(row["cos_wc_ci_high"]) for row in rows])
        axes[1, 1].plot(x, estimate, color=color, linewidth=1.8, label=label)
        axes[1, 1].fill_between(x, low, high, color=color, alpha=0.12)

    for ax in axes.flat:
        ax.axvline(0.66, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalized block depth (layer / (n_layers - 1))")
        ax.legend(fontsize=6.8, framealpha=0.9, ncol=2)
    axes[0, 0].set_title("Fold-internal mean-difference direction")
    axes[0, 0].set_ylabel("Topic-held-out accuracy")
    axes[0, 0].set_ylim(0.45, 1.02)
    axes[0, 1].set_title("Strict cross-axis transfer without target recalibration")
    axes[0, 1].set_ylabel("Topic-held-out transfer accuracy")
    axes[0, 1].set_ylim(0.45, 1.02)
    axes[1, 0].set_title("Descriptive separation with paired-topic 95% intervals")
    axes[1, 0].set_ylabel("Cohen's d")
    axes[1, 0].set_ylim(0, None)
    axes[1, 1].set_title("Axis cosine with paired-topic 95% intervals")
    axes[1, 1].set_ylabel("cos(warmth, competence)")
    axes[1, 1].axhspan(-0.3, 0.3, color="orange", alpha=0.08)
    axes[1, 1].axhline(0.0, color="gray", linewidth=0.6)
    axes[1, 1].set_ylim(-0.4, 1.0)
    fig.suptitle("Gemma 4 Stage 3B: generalization, transfer, and topic uncertainty by depth")
    fig.tight_layout()
    save("fig8b_stage3b_validation")


def fig9_gemma_scope_decomposition(
    metrics_paths: list[Path],
    model_labels: list[str],
) -> None:
    width_order = ["16k", "65k", "262k"]
    width_x = np.arange(len(width_order))
    colors = ["#1b7837", "#006d6d"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))

    for model_i, (path, label) in enumerate(zip(metrics_paths, model_labels)):
        rows = {row["width"]: row for row in _read_csv(path)}
        color = colors[model_i % len(colors)]
        reconstruction = [
            float(rows[width]["reconstruction_cosine_mean"])
            for width in width_order
        ]
        active = [
            float(rows[width]["active_features_mean"])
            for width in width_order
        ]
        axes[0, 0].plot(width_x, reconstruction, marker="o", color=color, label=label)
        axes[0, 1].plot(width_x, active, marker="o", color=color, label=label)

        for axis, linestyle in (("warmth", "-"), ("competence", "--")):
            topic_cv = [
                float(rows[width][f"{axis}_topic_cv"])
                for width in width_order
            ]
            alignment = [
                float(rows[width][f"decoded_{axis}_alignment"])
                for width in width_order
            ]
            axis_label = axis.capitalize()
            axes[1, 0].plot(
                width_x,
                topic_cv,
                marker="o",
                color=color,
                linestyle=linestyle,
                label=f"{label} — {axis_label}",
            )
            axes[1, 1].plot(
                width_x,
                alignment,
                marker="o",
                color=color,
                linestyle=linestyle,
                label=f"{label} — {axis_label}",
            )

    for ax in axes.flat:
        ax.set_xticks(width_x, width_order)
        ax.set_xlabel("SAE width")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_title("Residual reconstruction")
    axes[0, 0].set_ylabel("Cosine(original, reconstruction)")
    axes[0, 0].set_ylim(0.98, 1.0)
    axes[0, 0].legend(framealpha=0.9)
    axes[0, 1].set_title("Sparsity")
    axes[0, 1].set_ylabel("Mean active features per story vector")
    axes[1, 0].set_title("Concept signal in sparse feature space")
    axes[1, 0].set_ylabel("Topic-holdout CV accuracy")
    axes[1, 0].set_ylim(0.45, 1.0)
    axes[1, 0].legend(fontsize=8, framealpha=0.9)
    axes[1, 1].set_title("Decoded direction vs dense direction")
    axes[1, 1].set_ylabel("Cosine alignment")
    axes[1, 1].set_ylim(0.0, 1.0)
    fig.suptitle(
        "Gemma Scope 2 preserves residual activations better than it isolates "
        "warmth and competence",
        fontsize=12,
    )
    fig.tight_layout()
    save("fig9_gemma_scope_decomposition")


# ---------------------------------------------------------------------------
# Figure 10 — Held-out concept steering
# ---------------------------------------------------------------------------

def fig10_gemma_scope_steering(
    causality_paths: list[Path],
    model_labels: list[str],
) -> None:
    directions = [
        "raw_dense",
        "sae_reconstructed",
        "axis_specific",
        "shared",
        "other_axis",
        "random",
    ]
    labels = {
        "raw_dense": "Dense direction",
        "sae_reconstructed": "SAE reconstruction",
        "axis_specific": "Axis-specific features",
        "shared": "Shared features",
        "other_axis": "Other axis",
        "random": "Random direction",
    }
    colors = {
        "raw_dense": "#1b7837",
        "sae_reconstructed": "#006d6d",
        "axis_specific": "#762a83",
        "shared": "#d6604d",
        "other_axis": "#4575b4",
        "random": "#777777",
    }
    fig, axes = plt.subplots(
        2,
        len(causality_paths),
        figsize=(6.2 * len(causality_paths), 8),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(2, len(causality_paths))
    datasets = [_read_csv(path) for path in causality_paths]
    max_strength = max(
        abs(float(row["strength"]))
        for rows in datasets
        for row in rows
        if row["mode"] == "steering"
    )
    for model_i, (rows, model_label) in enumerate(
        zip(datasets, model_labels)
    ):
        for axis_i, axis_name in enumerate(("warmth", "competence")):
            ax = axes[axis_i, model_i]
            for direction in directions:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["mode"] == "steering"
                        and row["axis"] == axis_name
                        and row["direction"] == direction
                    ),
                    key=lambda row: float(row["strength"]),
                )
                x = np.array([float(row["strength"]) for row in selected])
                y = np.array([float(row["effect"]) for row in selected])
                low = np.array([float(row["ci_low"]) for row in selected])
                high = np.array([float(row["ci_high"]) for row in selected])
                ax.plot(
                    x,
                    y,
                    marker="o",
                    color=colors[direction],
                    label=labels[direction],
                )
                ax.fill_between(x, low, high, color=colors[direction], alpha=0.10)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.set_title(f"{model_label} — {axis_name.capitalize()}")
            ax.set_ylabel("Change in Yes-vs-No logit margin")
            ax.set_xlabel("Steering strength × mean residual norm")
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8, framealpha=0.9)
    regime = "Local-regime " if max_strength <= 0.1 else ""
    fig.suptitle(
        f"{regime}held-out concept judgements under residual-stream steering",
        fontsize=12,
    )
    fig.tight_layout()
    save("fig10_gemma_scope_steering")


# ---------------------------------------------------------------------------
# Figure 11 — Error-preserving feature ablation
# ---------------------------------------------------------------------------

def fig11_gemma_scope_ablation(
    causality_paths: list[Path],
    model_labels: list[str],
) -> None:
    directions = ["target_axis", "shared", "other_axis", "random_features"]
    display = ["Target axis", "Shared", "Other axis", "Random"]
    colors = ["#762a83", "#d6604d", "#4575b4", "#777777"]
    fig, axes = plt.subplots(
        2,
        len(causality_paths),
        figsize=(5.4 * len(causality_paths), 7.6),
        sharey=False,
    )
    axes = np.asarray(axes).reshape(2, len(causality_paths))
    for model_i, (path, model_label) in enumerate(
        zip(causality_paths, model_labels)
    ):
        rows = _read_csv(path)
        for axis_i, axis_name in enumerate(("warmth", "competence")):
            ax = axes[axis_i, model_i]
            selected = {
                row["direction"]: row
                for row in rows
                if row["mode"] == "ablation" and row["axis"] == axis_name
            }
            effects = np.array(
                [float(selected[direction]["effect"]) for direction in directions]
            )
            low = np.array(
                [float(selected[direction]["ci_low"]) for direction in directions]
            )
            high = np.array(
                [float(selected[direction]["ci_high"]) for direction in directions]
            )
            yerr = np.vstack([effects - low, high - effects])
            ax.bar(
                np.arange(len(directions)),
                effects,
                yerr=yerr,
                capsize=3,
                color=colors,
                alpha=0.9,
            )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(np.arange(len(directions)), display, rotation=20)
            ax.set_ylabel("Change in high–low margin gap")
            ax.set_title(f"{model_label} — {axis_name.capitalize()}")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Error-preserving ablation of 65k Gemma Scope feature sets",
        fontsize=12,
    )
    fig.tight_layout()
    save("fig11_gemma_scope_ablation")


# ---------------------------------------------------------------------------
# Figure 12 — Cross-scale feature matching
# ---------------------------------------------------------------------------

def fig12_gemma_scope_feature_matching(
    matches_path: Path,
    null_path: Path,
) -> None:
    rows = _read_csv(matches_path)
    null_rows = {
        row["vector"]: row
        for row in _read_csv(null_path)
    }
    vector_order = [
        "warmth",
        "competence",
        "shared",
        "warmth_specific",
        "competence_specific",
    ]
    values = [
        [
            float(row["story_profile_correlation"])
            for row in rows
            if row["vector"] == vector_name
        ]
        for vector_name in vector_order
    ]
    labels = [
        "Warmth",
        "Competence",
        "Shared",
        "Warmth-specific",
        "Competence-specific",
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    parts = ax.boxplot(
        values,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#004c4c", "linewidth": 2},
    )
    for box in parts["boxes"]:
        box.set_facecolor("#77b5b5")
        box.set_alpha(0.65)
    null_means = np.array(
        [float(null_rows[name]["null_mean"]) for name in vector_order]
    )
    null_low = np.array(
        [float(null_rows[name]["null_mean_ci_low"]) for name in vector_order]
    )
    null_high = np.array(
        [float(null_rows[name]["null_mean_ci_high"]) for name in vector_order]
    )
    ax.errorbar(
        np.arange(1, len(labels) + 1),
        null_means,
        yerr=np.vstack([null_means - null_low, null_high - null_means]),
        fmt="D",
        color="#d6604d",
        capsize=4,
        label="Permutation-null matched mean (95% interval)",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=18)
    ax.set_ylabel("Centered story-profile correlation")
    ax.set_title(
        "One-to-one 12B↔27B feature matches exceed the permutation null\n"
        "(orange diamonds: null matched mean and 95% interval)"
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save("fig12_gemma_scope_feature_matching")


# ---------------------------------------------------------------------------
# Figure 13 — Dense steering: per-model dose-response (2 rows × N models)
# ---------------------------------------------------------------------------

def fig13_dense_steering_doseresponse(
    dense_paths: list[Path],
    model_labels: list[str],
) -> None:
    """2-row × N-model grid showing dose-response for raw_dense and random directions.

    Y-axis is free per panel because raw logit effects differ by ~100× across models
    (mean_resid_norm spans ~4 orders of magnitude).
    """
    color_dense = "#1b7837"
    color_random = "#777777"
    n_models = len(dense_paths)
    fig, axes = plt.subplots(
        2,
        n_models,
        figsize=(5.0 * n_models, 8),
        sharey=False,
        sharex=True,
    )
    axes = np.asarray(axes).reshape(2, n_models)
    datasets = [_read_csv(path) for path in dense_paths]
    for model_i, (rows, model_label) in enumerate(zip(datasets, model_labels)):
        for axis_i, axis_name in enumerate(("warmth", "competence")):
            ax = axes[axis_i, model_i]
            for direction, color, ls, lbl in [
                ("raw_dense", color_dense, "-", "Dense direction"),
                ("random", color_random, "--", "Random direction"),
            ]:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["mode"] == "steering"
                        and row["axis"] == axis_name
                        and row["direction"] == direction
                    ),
                    key=lambda r: float(r["strength"]),
                )
                if not selected:
                    continue
                x = np.array([float(r["strength"]) for r in selected])
                y = np.array([float(r["effect"]) for r in selected])
                low = np.array([float(r["ci_low"]) for r in selected])
                high = np.array([float(r["ci_high"]) for r in selected])
                ax.plot(x, y, marker="o", color=color, linestyle=ls, label=lbl)
                ax.fill_between(x, low, high, color=color, alpha=0.10)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.set_title(f"{model_label} — {axis_name.capitalize()}")
            ax.set_ylabel("Change in Yes-vs-No logit margin")
            ax.set_xlabel("Steering strength × mean residual norm")
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8, framealpha=0.9)
    fig.suptitle(
        "Dense (SAE-free) residual-stream steering: per-model dose-response",
        fontsize=12,
    )
    fig.tight_layout()
    save("fig13_dense_steering_doseresponse")


# ---------------------------------------------------------------------------
# Figure 14 — Dense steering: normalized cross-model steerability
# ---------------------------------------------------------------------------

def fig14_dense_steering_normalized(
    dense_paths: list[Path],
    model_labels: list[str],
) -> None:
    """1×2 (warmth | competence): effect / baseline_gap per model, shared y-axis.

    Normalizing by the baseline high_low_margin_gap makes raw logit effects
    comparable across models despite their very different mean_resid_norm scales.
    """
    colors_models = ["#1b7837", "#762a83", "#4575b4", "#d6604d"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis_i, axis_name in enumerate(("warmth", "competence")):
        ax = axes[axis_i]
        for model_i, (path, model_label) in enumerate(zip(dense_paths, model_labels)):
            rows = _read_csv(path)
            # Baseline gap for this model × axis
            baseline_rows = [
                r for r in rows
                if r["mode"] == "baseline"
                and r["axis"] == axis_name
                and r["direction"] == "high_low_margin_gap"
            ]
            if not baseline_rows:
                continue
            baseline_gap = float(baseline_rows[0]["effect"])
            if abs(baseline_gap) < 1e-9:
                continue
            steer_rows = sorted(
                (
                    r for r in rows
                    if r["mode"] == "steering"
                    and r["axis"] == axis_name
                    and r["direction"] == "raw_dense"
                ),
                key=lambda r: float(r["strength"]),
            )
            if not steer_rows:
                continue
            x = np.array([float(r["strength"]) for r in steer_rows])
            y = np.array([float(r["effect"]) for r in steer_rows]) / baseline_gap
            color = colors_models[model_i % len(colors_models)]
            ax.plot(x, y, marker="o", color=color, label=model_label)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"{axis_name.capitalize()} — normalized steerability")
        ax.set_xlabel("Steering strength × mean residual norm")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Steering effect / baseline concept gap")
    axes[0].legend(fontsize=9, framealpha=0.9)
    fig.suptitle(
        "Cross-model steerability (effect normalized by baseline concept separation)",
        fontsize=12,
    )
    fig.tight_layout()
    save("fig14_dense_steering_normalized")


# ---------------------------------------------------------------------------
# Figure 15 — Dense steering: signal vs. control at peak strength
# ---------------------------------------------------------------------------

def fig15_dense_steering_signal_vs_control(
    dense_paths: list[Path],
    model_labels: list[str],
    peak_strength: float = 0.1,
) -> None:
    """1×2 (warmth | competence) grouped bars: raw_dense vs random at peak_strength.

    Annotates panels where the random-control effect rivals the dense direction —
    particularly Gemma-27B competence where random leakage dominates.
    """
    color_dense = "#1b7837"
    color_random = "#777777"
    n_models = len(dense_paths)
    x_pos = np.arange(n_models)
    bar_width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    for axis_i, axis_name in enumerate(("warmth", "competence")):
        ax = axes[axis_i]
        dense_effects: list[float] = []
        random_effects: list[float] = []
        for path in dense_paths:
            rows = _read_csv(path)
            for direction_name, target in [
                ("raw_dense", dense_effects),
                ("random", random_effects),
            ]:
                candidates = [
                    float(r["effect"])
                    for r in rows
                    if r["mode"] == "steering"
                    and r["axis"] == axis_name
                    and r["direction"] == direction_name
                    and abs(float(r["strength"]) - peak_strength) < 1e-6
                ]
                target.append(candidates[0] if candidates else float("nan"))
        dense_arr = np.array(dense_effects)
        random_arr = np.array(random_effects)
        ax.bar(x_pos - bar_width / 2, dense_arr, bar_width,
               color=color_dense, alpha=0.85, label="Dense direction")
        ax.bar(x_pos + bar_width / 2, random_arr, bar_width,
               color=color_random, alpha=0.60, label="Random direction")
        # Warn where |random| ≥ 80% of |dense| (non-specific leakage)
        for i, (d, r) in enumerate(zip(dense_arr, random_arr)):
            if not (np.isnan(d) or np.isnan(r)) and abs(d) > 1e-9 and abs(r) >= abs(d) * 0.8:
                ax.annotate(
                    "⚠",
                    xy=(x_pos[i] + bar_width / 2, r),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    color="#d6604d",
                )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_labels, rotation=18, ha="right")
        ax.set_title(
            f"{axis_name.capitalize()} — signal vs. control (α = {peak_strength})"
        )
        ax.set_ylabel("Steering effect (Δlogit margin)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"Dense direction vs. random control at peak strength "
        f"(α = {peak_strength} × mean_resid_norm)",
        fontsize=11,
    )
    fig.tight_layout()
    save("fig15_dense_steering_signal_vs_control")


# ---------------------------------------------------------------------------
# Figure 16 — Hiring: probe-vs-human Spearman ρ (grouped bars, all 4 models)
# ---------------------------------------------------------------------------

def fig16_hiring_probe_vs_human(
    audit_paths: list[Path],
    model_labels: list[str],
) -> None:
    """Grouped bars of Spearman ρ (model probe score vs. human rating) for each
    model × axis combination.  Negative bars (Llama/Qwen warmth) are shown in a
    distinct colour so the anti-alignment result reads clearly.
    """
    color_pos = "#1b7837"   # positive ρ
    color_neg = "#d6604d"   # negative ρ
    color_ns  = "#aaaaaa"   # not significant (p ≥ 0.05)
    axes_labels = ["warmth", "competence"]
    n_models = len(audit_paths)
    n_axes = 2
    x_pos = np.arange(n_models)
    bar_width = 0.35

    rhos: dict[str, list[float]] = {ax: [] for ax in axes_labels}
    pvals: dict[str, list[float]] = {ax: [] for ax in axes_labels}
    for path in audit_paths:
        rows = list(csv.DictReader(open(path)))
        human_warm  = np.array([float(r["human_warm"])       for r in rows])
        human_comp  = np.array([float(r["human_competent"])  for r in rows])
        model_warm  = np.array([float(r["model_warmth"])     for r in rows])
        model_comp  = np.array([float(r["model_competence"]) for r in rows])
        rw, pw = spearmanr(model_warm,  human_warm)
        rc, pc = spearmanr(model_comp,  human_comp)
        rhos["warmth"].append(float(rw))
        rhos["competence"].append(float(rc))
        pvals["warmth"].append(float(pw))
        pvals["competence"].append(float(pc))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    offsets = {"warmth": -bar_width / 2, "competence": bar_width / 2}
    labels_added: set[str] = set()
    for ax_name in axes_labels:
        for i, (rho, pval, model_label) in enumerate(
            zip(rhos[ax_name], pvals[ax_name], model_labels)
        ):
            if pval >= 0.05:
                clr = color_ns
                lbl = "n.s. (p≥0.05)"
            elif rho >= 0:
                clr = color_pos
                lbl = "ρ > 0"
            else:
                clr = color_neg
                lbl = "ρ < 0"
            legend_label = lbl if lbl not in labels_added else None
            if legend_label:
                labels_added.add(lbl)
            ax.bar(
                x_pos[i] + offsets[ax_name],
                rho,
                bar_width,
                color=clr,
                alpha=0.85,
                label=legend_label,
                hatch="//" if ax_name == "competence" else None,
                edgecolor="white",
            )
            ax.annotate(
                f"{rho:+.2f}",
                xy=(x_pos[i] + offsets[ax_name], rho),
                xytext=(0, 4 if rho >= 0 else -12),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels, rotation=15, ha="right")
    ax.set_ylabel("Spearman ρ (model probe vs. human rating)")
    ax.set_title(
        "Probe-vs-human alignment: model warmth/competence scores vs. crowdsourced ratings\n"
        "(solid = warmth, hatched = competence)"
    )
    ax.set_ylim(-0.6, 0.7)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
    fig.tight_layout()
    save("fig16_hiring_probe_vs_human")


# ---------------------------------------------------------------------------
# Figure 17 — Hiring: steering → callback (2 axes × N models grid)
# ---------------------------------------------------------------------------

def fig17_hiring_steering_callback(
    steering_paths: list[Path],
    model_labels: list[str],
) -> None:
    """2 rows (warmth, competence) × N cols (models): mean Δcallback-margin over 60
    names, with 95% CI (mean ± 1.96·SEM across names).  Free per-panel y-axis.
    """
    color_warm = "#d6604d"
    color_comp = "#4575b4"
    axes_names = ["warmth", "competence"]
    axis_colors = {"warmth": color_warm, "competence": color_comp}
    n_models = len(steering_paths)
    fig, axes = plt.subplots(
        2, n_models,
        figsize=(5.0 * n_models, 8),
        sharey=False,
        sharex=True,
    )
    axes = np.asarray(axes).reshape(2, n_models)
    for model_i, (path, model_label) in enumerate(zip(steering_paths, model_labels)):
        rows = list(csv.DictReader(open(path)))
        for axis_i, ax_name in enumerate(axes_names):
            ax = axes[axis_i, model_i]
            ax_rows = [r for r in rows if r["axis"] == ax_name]
            strengths = sorted({float(r["strength"]) for r in ax_rows})
            means, ci_lo, ci_hi = [], [], []
            for s in strengths:
                deltas = np.array([
                    float(r["delta"]) for r in ax_rows
                    if abs(float(r["strength"]) - s) < 1e-9
                ])
                m = float(np.mean(deltas))
                sem = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
                means.append(m)
                ci_lo.append(m - 1.96 * sem)
                ci_hi.append(m + 1.96 * sem)
            x = np.array(strengths)
            y = np.array(means)
            lo = np.array(ci_lo)
            hi = np.array(ci_hi)
            color = axis_colors[ax_name]
            ax.plot(x, y, marker="o", color=color)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.set_title(f"{model_label} — {ax_name.capitalize()}")
            ax.set_ylabel("Mean Δcallback margin (over 60 names)")
            ax.set_xlabel("Steering strength (× mean_resid_norm)")
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Steering → callback: mean Δmargin over 60 names ± 95% CI (name-level)",
        fontsize=12,
    )
    fig.tight_layout()
    save("fig17_hiring_steering_callback")


# ---------------------------------------------------------------------------
# Figure 18 — Hiring: disparity (two panels — magnitude + direction)
# ---------------------------------------------------------------------------

def fig18_hiring_disparity(
    disparity_paths: list[Path],
    audit_paths: list[Path],
    model_labels: list[str],
) -> None:
    """Two panels.
    Panel A: race and gender gaps (Black-White / Female-Male callback margin) in
    within-model SD units; human gaps (from Gallo & Hausladen 2024) drawn as
    horizontal reference lines.
    Panel B: direction-agreement grid — does sign(model gap) match sign(human gap)?
    """
    # Human benchmark gaps (from Gallo & Hausladen 2024 name-level data):
    #   race:   Black callback 0.183, White 0.171  → gap = +0.012 (positive)
    #   gender: Female 0.1454, Male 0.1815          → gap = −0.037 (negative)
    human_race_gap   = +0.012
    human_gender_gap = -0.037

    colors_gap = {"race": "#1b7837", "gender": "#762a83"}
    n_models = len(disparity_paths)
    x_pos = np.arange(n_models)
    bar_width = 0.35

    # Collect per-model within-SD-normalised gaps
    race_gaps_sd:   list[float] = []
    gender_gaps_sd: list[float] = []
    for disp_path, audit_path in zip(disparity_paths, audit_paths):
        disp_rows  = list(csv.DictReader(open(disp_path)))
        audit_rows = list(csv.DictReader(open(audit_path)))
        # Within-model SD of callback_margin across all names
        cb_margins = np.array([float(r["callback_margin"]) for r in audit_rows])
        sigma = float(np.std(cb_margins, ddof=1))
        if sigma < 1e-9:
            race_gaps_sd.append(float("nan"))
            gender_gaps_sd.append(float("nan"))
            continue
        # Race gap: Black − White
        race_rows = {r["group"]: float(r["model_callback_margin"])
                     for r in disp_rows if r["axis"] == "race"}
        race_gap = (race_rows.get("Black", float("nan"))
                    - race_rows.get("White", float("nan"))) / sigma
        race_gaps_sd.append(race_gap)
        # Gender gap: Female − Male
        gender_rows = {r["group"]: float(r["model_callback_margin"])
                       for r in disp_rows if r["axis"] == "gender"}
        gender_gap = (gender_rows.get("Female", float("nan"))
                      - gender_rows.get("Male", float("nan"))) / sigma
        gender_gaps_sd.append(gender_gap)

    # Human gaps in the same SD units requires a cross-model SD reference, which
    # we don't have.  We show them as %-point bars separately in the legend.
    race_arr   = np.array(race_gaps_sd)
    gender_arr = np.array(gender_gaps_sd)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: magnitudes ---
    ax = axes[0]
    ax.bar(x_pos - bar_width / 2, race_arr,   bar_width,
           color=colors_gap["race"],   alpha=0.85, label="Race gap (Black−White)")
    ax.bar(x_pos + bar_width / 2, gender_arr, bar_width,
           color=colors_gap["gender"], alpha=0.85, label="Gender gap (Female−Male)")
    ax.axhline(0, color="black", linewidth=0.8)
    # Human reference as dashed lines; note these are in %-point not SD units
    ax.axhline(human_race_gap,   color=colors_gap["race"],   linestyle="--",
               linewidth=1.2, label=f"Human race gap ({human_race_gap:+.3f} pp)", alpha=0.6)
    ax.axhline(human_gender_gap, color=colors_gap["gender"], linestyle="--",
               linewidth=1.2, label=f"Human gender gap ({human_gender_gap:+.3f} pp)", alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels, rotation=15, ha="right")
    ax.set_ylabel("Gap in within-model SD units  (model) / %-points (human ref.)")
    ax.set_title("(A) Disparity magnitude — model gaps in within-model SD units")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, framealpha=0.9)

    # --- Panel B: direction agreement ---
    ax2 = axes[1]
    ax2.set_xlim(-0.5, n_models - 0.5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_labels, rotation=15, ha="right")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Gender gap\n(human −)", "Race gap\n(human +)"])
    ax2.set_title("(B) Direction agreement with human benchmark")
    ax2.grid(False)
    ax2.set_facecolor("#f7f7f7")
    agree_color = "#1b7837"
    disagree_color = "#d6604d"
    for i, (rg, gg) in enumerate(zip(race_arr, gender_arr)):
        # Race: human gap positive (+0.012); model agrees if rg > 0
        race_agree = (not np.isnan(rg)) and (rg > 0)
        ax2.scatter(i, 1, s=300,
                    color=agree_color if race_agree else disagree_color,
                    zorder=3)
        ax2.text(i, 1.25, "✓" if race_agree else "✗",
                 ha="center", va="center", fontsize=14,
                 color=agree_color if race_agree else disagree_color)
        # Gender: human gap negative (−0.037); model agrees if gg < 0
        gender_agree = (not np.isnan(gg)) and (gg < 0)
        ax2.scatter(i, 0, s=300,
                    color=agree_color if gender_agree else disagree_color,
                    zorder=3)
        ax2.text(i, -0.25, "✓" if gender_agree else "✗",
                 ha="center", va="center", fontsize=14,
                 color=agree_color if gender_agree else disagree_color)
    # legend patches
    from matplotlib.patches import Patch
    legend_els = [
        Patch(color=agree_color,    label="Direction matches human"),
        Patch(color=disagree_color, label="Direction opposes human"),
    ]
    ax2.legend(handles=legend_els, fontsize=8, framealpha=0.9, loc="upper right")

    fig.suptitle(
        "Model callback disparity vs. human benchmark (Gallo & Hausladen, 2024)",
        fontsize=12,
    )
    fig.tight_layout()
    save("fig18_hiring_disparity")


# ---------------------------------------------------------------------------
# Figure 19 — Hiring: mediation forest plot (indirect effects + 95% CI)
# ---------------------------------------------------------------------------

def fig19_hiring_mediation_forest(
    mediation_paths: list[Path],
    model_labels: list[str],
) -> None:
    """Forest plot of bootstrap indirect effects (probe mediating name-group → callback).
    One row per (model, grouping, probe).  Significant rows are filled; null rows are open.
    Grouped by model, with a horizontal separator between models.
    """
    color_sig    = "#762a83"
    color_nonsig = "#aaaaaa"

    # Collect rows in display order
    entries: list[dict] = []
    for path, model_label in zip(mediation_paths, model_labels):
        med = json.load(open(path))
        for m in med["mediation"]:
            entries.append({
                "model":     model_label,
                "grouping":  m["grouping"],
                "probe":     m["probe"],
                "effect":    float(m["indirect_effect"]),
                "lo":        float(m["ci_95_lo"]),
                "hi":        float(m["ci_95_hi"]),
                "sig":       bool(m["significant_95"]),
            })

    n = len(entries)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.55 * n)))
    y_pos = np.arange(n)

    model_seen: set[str] = set()
    separator_ys: list[float] = []
    current_model = ""
    for i, e in enumerate(entries):
        if e["model"] != current_model:
            if current_model:
                separator_ys.append(i - 0.5)
            current_model = e["model"]

    for i, e in enumerate(entries):
        color = color_sig if e["sig"] else color_nonsig
        marker = "D" if e["sig"] else "o"
        mfc    = color if e["sig"] else "white"
        ax.plot([e["lo"], e["hi"]], [i, i], color=color, linewidth=1.2)
        ax.scatter(e["effect"], i, color=color, marker=marker,
                   facecolors=mfc, s=55, zorder=3, linewidths=1.2)
        label = f"{e['model']} | {e['grouping']} | {e['probe']}"
        ax.text(-0.23, i, label, va="center", ha="right", fontsize=7)

    for y in separator_ys:
        ax.axhline(y, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks([])
    ax.set_xlabel("Bootstrap indirect effect (95% CI)")
    ax.set_title(
        "Mediation: name group → probe activation → callback margin\n"
        "(filled = significant @95%; open = n.s.)"
    )
    ax.grid(axis="x", alpha=0.2)
    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="D", color=color_sig,    markerfacecolor=color_sig,
               markersize=7, linestyle="-", label="Significant (95% CI ∉ 0)"),
        Line2D([0], [0], marker="o", color=color_nonsig, markerfacecolor="white",
               markersize=7, linestyle="-", label="Non-significant"),
    ]
    ax.legend(handles=legend_els, fontsize=8, framealpha=0.9, loc="lower right")
    ax.set_xlim(left=ax.get_xlim()[0] - 0.05)
    fig.tight_layout()
    save("fig19_hiring_mediation_forest")


# ---------------------------------------------------------------------------
# Figure 20 — PCA denoising of neutral-variance directions
# ---------------------------------------------------------------------------

def fig20_pca_denoising(vec_dirs: list[Path], model_labels: list[str]) -> None:
    """Summarise PCA denoising against a neutral Wikipedia corpus.

    The denoising step fits PCA on neutral residual activations, removes enough
    top PCs to explain >=50% neutral variance, and compares raw vs denoised
    concept geometry and concept discriminability.
    """
    rows: list[dict] = []
    curves: list[tuple[str, np.ndarray, int]] = []

    def project(x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.einsum("ij,j->i", x, v, optimize=False)

    for vec_dir, label in zip(vec_dirs, model_labels):
        summary = json.load(open(vec_dir / "denoise_summary.json"))
        z = np.load(vec_dir / "concept_vectors_denoised.npz")
        neutral = np.load(vec_dir / "X_neutral.npy").astype(np.float64)

        pca = PCA().fit(neutral)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = int(summary["k"])
        curves.append((label, cum, k))

        raw_w = np.load(vec_dir / "warmth_vec.npy").astype(np.float64)
        raw_c = np.load(vec_dir / "competence_vec.npy").astype(np.float64)
        den_w = z["warmth"].astype(np.float64)
        den_c = z["competence"].astype(np.float64)

        hw = np.load(vec_dir / "X_high_warmth.npy").astype(np.float64)
        lw = np.load(vec_dir / "X_low_warmth.npy").astype(np.float64)
        hc = np.load(vec_dir / "X_high_competence.npy").astype(np.float64)
        lc = np.load(vec_dir / "X_low_competence.npy").astype(np.float64)

        rows.append({
            "model": label,
            "k": k,
            "variance": float(summary["variance_kept"]),
            "cos_raw": float(summary["cosine_before"]),
            "cos_denoised": float(summary["cosine_after"]),
            "d_warm_raw": cohens_d(project(hw, raw_w), project(lw, raw_w)),
            "d_warm_denoised": cohens_d(project(hw, den_w), project(lw, den_w)),
            "d_comp_raw": cohens_d(project(hc, raw_c), project(lc, raw_c)),
            "d_comp_denoised": cohens_d(project(hc, den_c), project(lc, den_c)),
            "n_neutral": neutral.shape[0],
            "d_model": neutral.shape[1],
        })

    colors = ["#2E86AB", "#7D6608", "#6B7280", "#A23B72"]
    fig = plt.figure(figsize=(9.0, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.36, wspace=0.28)
    ax_var = fig.add_subplot(gs[0, 0])
    ax_cos = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    # Panel A: cumulative neutral PCA variance.
    for i, (label, cum, k) in enumerate(curves):
        x = np.arange(1, min(len(cum), 120) + 1)
        y = cum[:len(x)]
        color = colors[i % len(colors)]
        ax_var.plot(x, y, color=color, lw=1.8, label=label)
        ax_var.scatter([k], [cum[k - 1]], color=color, s=36, zorder=3)
        ax_var.annotate(
            f"k={k}",
            xy=(k, cum[k - 1]),
            xytext=(6, 8 if i == 0 else -18),
            textcoords="offset points",
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )
    ax_var.axhline(0.50, color="#444444", lw=0.9, ls="--")
    ax_var.set_ylim(0, 1.02)
    ax_var.set_xlim(0, 120)
    ax_var.set_xlabel("Neutral PCA component")
    ax_var.set_ylabel("Cumulative variance explained")
    ax_var.set_title("A. Neutral variance removed")
    ax_var.legend(framealpha=0.9, fontsize=8, loc="lower right")
    ax_var.grid(axis="y", alpha=0.18)

    # Panel B: before/after axis cosine.
    x = np.arange(len(rows))
    width = 0.34
    raw_vals = [r["cos_raw"] for r in rows]
    den_vals = [r["cos_denoised"] for r in rows]
    ax_cos.bar(x - width / 2, raw_vals, width, color="#9CA3AF", label="Raw")
    ax_cos.bar(x + width / 2, den_vals, width, color="#2E86AB", label="Denoised")
    for xi, raw, den in zip(x, raw_vals, den_vals):
        ax_cos.plot([xi - width / 2, xi + width / 2], [raw, den], color="#444444", lw=0.9)
        ax_cos.text(xi, max(raw, den) + 0.025, f"{raw:.2f}->{den:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax_cos.set_xticks(x)
    ax_cos.set_xticklabels([r["model"] for r in rows], rotation=0)
    ax_cos.set_ylim(0, max(raw_vals) * 1.22)
    ax_cos.set_ylabel("cos(warmth, competence)")
    ax_cos.set_title("B. Shared axis geometry reduced")
    ax_cos.legend(framealpha=0.9, fontsize=8)
    ax_cos.grid(axis="y", alpha=0.18)

    # Panel C: discriminability before/after.
    labels = []
    raw_d = []
    den_d = []
    for r in rows:
        short = r["model"].replace("Gemma-3-", "")
        labels.extend([f"{short}\nWarmth", f"{short}\nCompetence"])
        raw_d.extend([r["d_warm_raw"], r["d_comp_raw"]])
        den_d.extend([r["d_warm_denoised"], r["d_comp_denoised"]])
    xd = np.arange(len(labels))
    ax_d.bar(xd - width / 2, raw_d, width, color="#9CA3AF", label="Raw")
    ax_d.bar(xd + width / 2, den_d, width, color="#F18F01", label="Denoised")
    ax_d.axhline(0, color="#333333", lw=0.8)
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(labels)
    ax_d.set_ylabel("Concept separation (Cohen's d)")
    ax_d.set_title("C. Concept signal after removing neutral PCs")
    ax_d.legend(framealpha=0.9, fontsize=8)
    ax_d.grid(axis="y", alpha=0.18)

    # Panel D: concise method/result card.
    ax_text.axis("off")
    lines = [
        "D. What the PCA step does",
        "",
        "Fit PCA on neutral Wikipedia residual activations",
        "and project the top neutral-variance directions",
        "out of warmth and competence vectors.",
        "",
    ]
    for r in rows:
        lines.append(
            f"{r['model']}: {r['n_neutral']:,} neutral texts, d={r['d_model']:,}, "
            f"k={r['k']} PCs ({r['variance']:.1%} variance)"
        )
    lines.extend([
        "",
        "Interpretation: denoising reduces shared neutral",
        "geometry, but the remaining warmth/competence",
        "cosine is not forced to zero; SCM predicts",
        "substantive correlation between the axes.",
    ])
    ax_text.text(
        0.02, 0.98, "\n".join(lines),
        ha="left", va="top", fontsize=9.2, linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.55", fc="#F8FAFC", ec="#CBD5E1", lw=1.0),
        transform=ax_text.transAxes,
    )

    fig.suptitle(
        "PCA denoising removes dominant neutral-variance directions from concept vectors",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.88, hspace=0.42, wspace=0.28)
    save("fig20_pca_denoising")


# ---------------------------------------------------------------------------
# paper_figure1 — Warmth–Competence space with oblique-basis direction arrows
# ---------------------------------------------------------------------------

def paper_figure1_axis_arrows(vec_dirs: list[Path], model_labels: list[str]) -> None:
    """2×2 panels: warmth–competence story cloud + real-angle direction arrows.

    For each model the story activations are projected onto both axes and then
    displayed in an *oblique* coordinate system whose horizontal axis is the
    warmth direction and whose second axis is the competence direction, drawn at
    the true inter-axis angle theta = arccos(cos(W,C)).  This preserves the
    geometric relationship discovered in the analysis: Gemma models have an
    elevated cos(W,C) (~0.71-0.75, i.e. ~41-45°) while Qwen/Llama are closer to
    orthogonal (~0.51-0.54, i.e. ~57-59°).

    Coordinate transform (oblique basis):
        x_plot = z_w + z_c * cos(theta)
        y_plot = z_c * sin(theta)
    where z_w and z_c are model-internal z-scores of the projections.
    """
    _style.apply()

    n_models = len(vec_dirs)
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.25))
    axes_flat = axes.flatten()

    # Color per condition (from PALETTE) + arrow colours
    ARROW_W_COLOR = "#1A5276"   # deep blue — warmth
    ARROW_C_COLOR = "#7D6608"   # deep gold — competence
    SCATTER_ALPHA = 0.55
    SCATTER_SIZE  = 18

    for i_m, (vd, label) in enumerate(zip(vec_dirs, model_labels)):
        ax = axes_flat[i_m]

        # Load vectors
        wv = unit(np.load(vd / "warmth_vec.npy").astype(np.float64))
        cv = unit(np.load(vd / "competence_vec.npy").astype(np.float64))
        axis_cosine = float(np.dot(wv, cv))
        theta = float(np.arccos(np.clip(axis_cosine, -1.0, 1.0)))
        theta_deg = np.degrees(theta)

        # Collect projections for all 200 stories
        all_proj_w, all_proj_c = [], []
        cond_projs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for cond in CONDITIONS:
            X = np.load(vd / f"X_{cond}.npy").astype(np.float64)
            pw = X @ wv
            pc = X @ cv
            all_proj_w.append(pw)
            all_proj_c.append(pc)
            cond_projs[cond] = (pw, pc)

        all_w = np.concatenate(all_proj_w)
        all_c = np.concatenate(all_proj_c)
        mu_w, sig_w = all_w.mean(), all_w.std() + 1e-12
        mu_c, sig_c = all_c.mean(), all_c.std() + 1e-12

        # Compute oblique-basis coordinates for all stories (to determine axis limits)
        all_xp, all_yp = [], []
        for cond in CONDITIONS:
            pw, pc = cond_projs[cond]
            zw = (pw - mu_w) / sig_w
            zc = (pc - mu_c) / sig_c
            all_xp.append(zw + zc * np.cos(theta))
            all_yp.append(zc * np.sin(theta))
        all_xp_arr = np.concatenate(all_xp)
        all_yp_arr = np.concatenate(all_yp)

        # Plot story clouds in oblique coordinates
        for cond in CONDITIONS:
            pw, pc = cond_projs[cond]
            zw = (pw - mu_w) / sig_w
            zc = (pc - mu_c) / sig_c
            xp = zw + zc * np.cos(theta)
            yp = zc * np.sin(theta)
            ax.scatter(
                xp, yp,
                c=_style.PALETTE[cond],
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                linewidths=0,
                label=_style.LABELS[cond],
            )

        # Direction arrows (unit length in z-score units, scaled visually)
        ARROW_LEN = 2.0
        cx_end = ARROW_LEN * np.cos(theta)
        cy_end = ARROW_LEN * np.sin(theta)

        # FIX: set axis limits BEFORE drawing arrows so the arrows are
        # guaranteed visible even when cy_end > scatter y-range.
        # Include arrow endpoints + label clearance (0.4 units top/right).
        x_lo = float(all_xp_arr.min()) - 0.3
        x_hi = max(float(all_xp_arr.max()), ARROW_LEN) + 0.6
        y_lo = float(all_yp_arr.min()) - 0.3
        y_hi = max(float(all_yp_arr.max()), cy_end) + 0.55  # room for "Competence +"
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        # Warmth arrow: along horizontal axis in oblique basis
        ax.annotate(
            "", xy=(ARROW_LEN, 0), xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=ARROW_W_COLOR,
                lw=2.4,
                mutation_scale=18,
            ),
            annotation_clip=False,   # FIX: never clip arrows at axes boundary
            zorder=5,
        )
        ax.text(ARROW_LEN + 0.12, 0.05, "Warmth +",
                color=ARROW_W_COLOR, fontsize=9, fontweight="bold",
                clip_on=False)

        # Competence arrow: drawn at oblique angle
        ax.annotate(
            "", xy=(cx_end, cy_end), xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=ARROW_C_COLOR,
                lw=2.4,
                mutation_scale=18,
            ),
            annotation_clip=False,   # FIX: never clip arrows at axes boundary
            zorder=5,
        )
        ax.text(cx_end + 0.08, cy_end + 0.12, "Competence +",
                color=ARROW_C_COLOR, fontsize=9, fontweight="bold",
                clip_on=False)

        # Arc showing angle between directions
        arc_r = 0.55
        n_arc = 60
        arc_angles = np.linspace(0, theta, n_arc)
        arc_x = arc_r * np.cos(arc_angles)
        arc_y = arc_r * np.sin(arc_angles)
        ax.plot(arc_x, arc_y, color="#555555", linewidth=1.0, linestyle="--")
        mid_angle = theta / 2
        ax.text(
            (arc_r + 0.18) * np.cos(mid_angle) + (0.32 if i_m in {0, 1} else 0.20),
            (arc_r + 0.18) * np.sin(mid_angle),
            f"{theta_deg:.0f}°",
            ha="center", va="center", fontsize=9, color="#333333",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        )

        # Origin crosshairs
        ax.axhline(0, color="lightgray", linewidth=0.6, zorder=0)
        ax.axvline(0, color="lightgray", linewidth=0.6, zorder=0)

        # Annotation box: cos and angle
        ax.text(
            0.03, 0.97,
            f"cos(W,C) = {axis_cosine:.3f}\nθ = {theta_deg:.1f}°",
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.85),
        )

        ax.set_title(label, fontsize=11, fontweight="bold")

        # FIX: x-label only on bottom row (i_m 2,3); y-label only on left column (i_m 0,2)
        ax.set_xlabel("Warmth axis (z-score)" if i_m in {2, 3} else "", fontsize=9)
        ax.set_ylabel("Competence axis (z-score, oblique)" if i_m in {0, 2} else "",
                      fontsize=9)

        # FIX: legend only on bottom-right panel (Llama, i_m == 3)
        if i_m == 3:
            ax.legend(loc="lower right", fontsize=6, framealpha=0.92,
                      markerscale=1.2, handletextpad=0.3)

    fig.suptitle(
        "Warmth and competence axes in representation space\n"
        "Arrow angle = true geometric angle between direction vectors",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save("paper_figure1_axis_arrows")


# ---------------------------------------------------------------------------
# paper_figure2 — Layer emergence: Cohen's d vs depth, 4 models (single panel)
# ---------------------------------------------------------------------------

def paper_figure2_layer_emergence(
    sweep_csv_paths: list[Path],
    model_labels: list[str],
) -> None:
    """Two side-by-side panels: warmth (left) and competence (right) emergence.

    Cohen's d (y) vs layer fraction (x) for all four models in each panel.
    Each curve ends with a colour-matched label showing total layer count and
    residual-stream dimension: e.g. "48L · d3840".  This contextualises the
    normalised x-axis: models with more layers spread the same frac range
    over more discrete steps.

    Note: the number of activation vectors extracted per layer is constant at
    200 (4 conditions × 50 stories) for every model and every layer.  Only
    d_model (the vector dimension) varies across models and is shown in the label.

    Probe layer at frac=0.66 is labelled directly on the figure (not in legend).
    No large-effect threshold line.

    CSV columns expected (from src/layer_sweep.py):
        frac, warmth_cohens_d, comp_cohens_d, cos_wc
    """
    import csv as _csv
    import matplotlib.transforms as _mtrans

    _style.apply()

    model_colors = ["#1b7837", "#006d6d", "#762a83", "#d6604d"]
    model_ls     = ["-", (0, (3, 1, 1, 1)), "--", "-."]

    # d_model per label (residual-stream width; verified from X_*.npy shapes).
    D_MODEL: dict[str, int] = {
        "Gemma-3-12B":  3840,
        "Gemma-3-27B":  5376,
        "Qwen3-14B":    5120,
        "Llama-3.1-8B": 4096,
    }

    sweeps: list[dict[str, list]] = []
    for path in sweep_csv_paths:
        rows: dict[str, list] = {"frac": [], "warmth_d": [], "comp_d": []}
        with path.open(newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                rows["frac"].append(float(row["frac"]))
                rows["warmth_d"].append(float(row["warmth_cohens_d"]))
                rows["comp_d"].append(float(row["comp_cohens_d"]))
        sweeps.append(rows)
        print(f"  [paper_fig2] {path.name}: {len(rows['frac'])} layers")

    all_d = [v for sw in sweeps for k in ("warmth_d", "comp_d") for v in sw[k]]
    y_max = max(all_d) * 1.05

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.25, 3.4), sharey=True)

    for ax in (axL, axR):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, y_max)

    # Collect terminal (y, text, color) per panel for de-cluttered end labels.
    end_labels_W: list[tuple[float, str, str]] = []
    end_labels_C: list[tuple[float, str, str]] = []

    for i_m, (sweep, label) in enumerate(zip(sweeps, model_labels)):
        c        = model_colors[i_m % len(model_colors)]
        ls       = model_ls[i_m % len(model_ls)]
        n_layers = len(sweep["frac"])
        dm       = D_MODEL.get(label, None)
        tag      = f"{n_layers}L · d{dm}" if dm else f"{n_layers}L"

        axL.plot(sweep["frac"], sweep["warmth_d"], color=c, linestyle=ls,
                 linewidth=2.0, label=label, zorder=3)
        axR.plot(sweep["frac"], sweep["comp_d"],   color=c, linestyle=ls,
                 linewidth=2.0, zorder=3)

        end_labels_W.append((sweep["warmth_d"][-1], tag, c))
        end_labels_C.append((sweep["comp_d"][-1],   tag, c))

    def _draw_end_labels(ax, labels: list[tuple[float, str, str]]) -> None:
        """Render colour-matched line-end labels with vertical de-cluttering."""
        min_gap = y_max * 0.045
        # Sort by y ascending, then nudge upward any pair that overlaps.
        items = sorted(labels, key=lambda t: t[0])
        y_adj = [t[0] for t in items]
        for i in range(1, len(y_adj)):
            if y_adj[i] - y_adj[i - 1] < min_gap:
                y_adj[i] = y_adj[i - 1] + min_gap
        for (_, text, color), y in zip(items, y_adj):
            ax.text(1.012, y, text, color=color,
                    fontsize=7, ha="left", va="center",
                    clip_on=False, zorder=6)

    _draw_end_labels(axR, end_labels_C)

    # Probe-layer vertical line — labelled on figure, not in legend.
    for ax in (axL, axR):
        ax.axvline(0.66, color="gray", linestyle=":", linewidth=1.2, zorder=1)
        trans = _mtrans.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(0.67, 0.97, "probe layer\nfrac = 0.66",
                transform=trans, fontsize=7.5, color="gray",
                va="top", ha="left", zorder=2)
        ax.set_xlabel("Layer fraction (layer index / n_layers)", fontsize=11)

    axL.set_ylabel("Cohen's d", fontsize=11)
    axL.set_title("Warmth",     fontsize=12, fontweight="bold")
    axR.set_title("Competence", fontsize=12, fontweight="bold")
    axL.legend(fontsize=5.5, loc="upper left", ncol=1, framealpha=0.92)

    fig.suptitle(
        "Warmth and competence representations emerge with depth\n"
        "(topic-holdout CV = 1.00 at every layer; four open-weights models)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.12)
    save("paper_figure2_layer_emergence")


# ---------------------------------------------------------------------------
# paper_figure3_diverging_steering
# ---------------------------------------------------------------------------

def paper_figure3_diverging_steering(slopes_csv: Path) -> None:
    """Position + boundary chart showing causal steering of Yes/No social judgement.

    x-axis = absolute Yes/No logit margin; x=0 = decision boundary.
    Each row: bull’s-eye dot = baseline (no steering); line+arrow = steerable range
    at ±0.10 × mean residual norm. Every row crosses the boundary.

    Note: concept judgement only, NOT hiring callbacks.
    Gemma 12B & 27B; raw_dense direction; local steering regime.
    """
    import csv as _csv
    import matplotlib.transforms as _mt

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    rows: list[dict] = []
    with slopes_csv.open(newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row["direction"] == "raw_dense":
                rows.append(row)

    ORDER = [
        ("gemma3_12b", "warmth"),
        ("gemma3_12b", "competence"),
        ("gemma3_27b", "warmth"),
        ("gemma3_27b", "competence"),
    ]
    DISPLAY_LABELS = {
        ("gemma3_12b", "warmth"):      "Gemma-3-12B  ·  Warmth",
        ("gemma3_12b", "competence"):  "Gemma-3-12B  ·  Competence",
        ("gemma3_27b", "warmth"):      "Gemma-3-27B  ·  Warmth",
        ("gemma3_27b", "competence"):  "Gemma-3-27B  ·  Competence",
    }
    # Vivid, distinct colour pair: coral-red for warmth, teal for competence
    AXIS_COLOR = {
        "warmth":     "#C0392B",   # bold coral-red
        "competence": "#117A65",   # deep teal
    }

    data_map: dict[tuple, tuple[float, float]] = {}
    for row in rows:
        key = (row["label"], row["axis"])
        data_map[key] = (float(row["local_slope"]), float(row["intercept"]))

    STRENGTH = 0.10
    entries: list[tuple] = []
    for key in ORDER:
        slope, intercept = data_map[key]
        entries.append((key, intercept,
                        intercept - slope * STRENGTH,
                        intercept + slope * STRENGTH))

    # ------------------------------------------------------------------
    # 2. Single-axes figure — compact and wide-enough for labels
    # ------------------------------------------------------------------
    fig, axMain = plt.subplots(figsize=(3.9, 1.95))
    fig.subplots_adjust(left=0.30, right=0.88, top=0.84, bottom=0.20)

    n = len(entries)
    y_positions = list(range(n - 1, -1, -1))

    all_vals = [v for (_, ic, mm, mp) in entries for v in (ic, mm, mp)]
    xpad = (max(all_vals) - min(all_vals)) * 0.20
    x_lo = min(all_vals) - xpad
    x_hi = max(all_vals) + xpad

    axMain.set_xlim(x_lo, x_hi)
    axMain.set_ylim(-0.65, n - 0.35)

    # Decision boundary
    axMain.axvline(0, color="#333333", linestyle="--", linewidth=1.0, zorder=2)

    # Yes-side very subtle shade
    axMain.axvspan(0, x_hi + 1, alpha=0.04, color="#444444", zorder=0)

    # "No" / "Yes" column headers above chart
    trans_blend = _mt.blended_transform_factory(axMain.transData, axMain.transAxes)
    axMain.text(0.01, 1.04, "No",
                transform=axMain.transAxes, fontsize=9, ha="left", va="bottom",
                color="#333333", fontstyle="italic")
    axMain.text(0.99, 1.04, "Yes",
                transform=axMain.transAxes, fontsize=9, ha="right", va="bottom",
                color="#333333", fontstyle="italic")

    # Draw rows
    lpad = (x_hi - x_lo) * 0.025
    for i, ((key, intercept, m_minus, m_plus), y) in enumerate(
            zip(entries, y_positions)):
        color = AXIS_COLOR[key[1]]

        # Group separator between 12B and 27B
        if i == 2:
            axMain.axhline(y + 0.5, color="#E8E8E8", linewidth=0.7, zorder=0)

        # Steering range line
        axMain.plot([m_minus, m_plus], [y, y],
                    color=color, lw=2.2, zorder=3, solid_capstyle="butt",
                    alpha=0.85)

        # Arrowhead at +steering end
        tail_x = m_minus + (m_plus - m_minus) * 0.80
        axMain.annotate(
            "",
            xy=(m_plus, y), xytext=(tail_x, y),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=2.2, mutation_scale=12),
            zorder=4,
        )

        # Baseline dot: filled outer, white inner
        axMain.plot(intercept, y, "o", color=color, markersize=10,
                    zorder=5, markeredgewidth=0)
        axMain.plot(intercept, y, "o", color="white", markersize=5,
                    zorder=6, markeredgewidth=0)

        # End-value labels
        axMain.text(m_minus - lpad, y, f"{m_minus:.2f}",
                    color=color, fontsize=7.5, va="center", ha="right",
                    alpha=0.80)
        axMain.text(m_plus + lpad, y, f"+{m_plus:.2f}",
                    color=color, fontsize=7.5, va="center", ha="left",
                    alpha=0.80)

    # y-tick labels
    axMain.set_yticks(y_positions)
    axMain.set_yticklabels(
        [DISPLAY_LABELS[k] for (k, *_) in entries], fontsize=8.5,
    )
    axMain.tick_params(axis="y", length=0)

    # Strip x-axis decoration
    axMain.set_xticks([])
    for sp in ("top", "right", "bottom"):
        axMain.spines[sp].set_visible(False)
    axMain.spines["left"].set_color("#333333")

    axMain.set_xlabel(
        "Yes/No logit margin   (0 = decision boundary)",
        fontsize=8.5, labelpad=4, color="#555555",
    )

    save("paper_figure3_diverging_steering")


# ---------------------------------------------------------------------------
# (removed) paper_figure2_universal_representation — replaced by layer_emergence
# ---------------------------------------------------------------------------

def _paper_figure2_universal_representation_REMOVED(
    vec_dirs: list[Path],
    model_labels: list[str],
    metrics_paths: list[Path],
) -> None:  # pragma: no cover
    """Dead code — replaced by paper_figure2_layer_emergence (2026-06-24).

    The 2×2 KDE small-multiples overlapped with paper_figure1_axis_arrows
    (same story clouds, same geometric message).  The layer-emergence figure
    adds the depth dimension instead.
    """
    raise NotImplementedError("This function has been superseded.")

    _style.apply()

    def _read_metrics(path: Path) -> dict[str, float]:
        out: dict[str, float] = {}
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ax_name = row["axis"]
                out[f"cohens_d_{ax_name}"] = float(row["cohens_d"])
                out[f"cv_{ax_name}"] = float(row["cv_mean"])
        return out

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes_flat = axes.flatten()

    for i_m, (vd, label, mpath) in enumerate(zip(vec_dirs, model_labels, metrics_paths)):
        ax = axes_flat[i_m]
        wv = unit(np.load(vd / "warmth_vec.npy").astype(np.float64))
        cv = unit(np.load(vd / "competence_vec.npy").astype(np.float64))

        all_w_proj: list[np.ndarray] = []
        all_c_proj: list[np.ndarray] = []
        cond_projs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for cond in CONDITIONS:
            X = np.load(vd / f"X_{cond}.npy").astype(np.float64)
            pw = X @ wv
            pc = X @ cv
            all_w_proj.append(pw)
            all_c_proj.append(pc)
            cond_projs[cond] = (pw, pc)

        all_w = np.concatenate(all_w_proj)
        all_c = np.concatenate(all_c_proj)
        mu_w, sig_w = all_w.mean(), all_w.std() + 1e-12
        mu_c, sig_c = all_c.mean(), all_c.std() + 1e-12

        for cond in CONDITIONS:
            pw, pc = cond_projs[cond]
            zw = (pw - mu_w) / sig_w
            zc = (pc - mu_c) / sig_c
            try:
                sns.kdeplot(
                    x=zw, y=zc,
                    ax=ax,
                    color=_style.PALETTE[cond],
                    levels=5,
                    linewidths=1.6,
                    label=_style.LABELS[cond],
                )
            except Exception:
                # KDE can fail with tiny variance; fall back to scatter
                ax.scatter(zw, zc, color=_style.PALETTE[cond], s=10, alpha=0.4,
                           label=_style.LABELS[cond])

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)

        # Metrics badge
        m = _read_metrics(mpath)
        dw = m.get("cohens_d_warmth", float("nan"))
        dc = m.get("cohens_d_competence", float("nan"))
        cvw = m.get("cv_warmth", float("nan"))
        badge = (
            f"d(W) = {dw:.2f}  d(C) = {dc:.2f}\n"
            f"CV = {cvw:.2f}"
        )
        ax.text(
            0.03, 0.97, badge,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.88),
        )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Warmth projection (z-score)", fontsize=9)
        ax.set_ylabel("Competence projection (z-score)", fontsize=9)

        if i_m == 0:
            from matplotlib.lines import Line2D as _L2D
            handles = [_L2D([0], [0], color=_style.PALETTE[c], lw=2,
                            label=_style.LABELS[c]) for c in CONDITIONS]
            ax.legend(handles=handles, loc="lower right", fontsize=7.5,
                      framealpha=0.92)

    fig.suptitle(
        "Warmth and competence representations across four language models\n"
        "All panels z-scored within model; d = Cohen's d, CV = 5-fold accuracy",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save("paper_figure2_universal_representation")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global VEC_DIR, OUT_DIR  # allow reassignment from CLI

    parser = argparse.ArgumentParser(description="Generate presentation figures.")
    parser.add_argument(
        "--fig", default="all",
        help="Figure(s) to generate: 1-21, comma-separated, or 'all' (runs 1-4 only).",
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
             "(required for --fig 6/7/20). E.g. "
             "data/processed/concept_vectors,"
             "data/processed/concept_vectors_qwen3_14b,"
             "data/processed/concept_vectors_llama31_8b",
    )
    parser.add_argument(
        "--stories",
        default=None,
        help="Path to concept_stories.jsonl for fig7 text labels (optional).",
    )
    parser.add_argument(
        "--agreement-csv",
        default=None,
        help="Tracked cross-model agreement CSV for fig6 (preferred; vec-dirs fallback).",
    )
    parser.add_argument(
        "--sweep-csvs",
        default=None,
        help="Comma-separated paths to layer_sweep_<label>.csv files "
             "(required for --fig 8). Same order as --labels.",
    )
    parser.add_argument(
        "--scope-metrics",
        default=None,
        help="Comma-separated gemma_scope_metrics CSVs (required for --fig 9).",
    )
    parser.add_argument(
        "--causality-csvs",
        default=None,
        help=(
            "Comma-separated gemma_scope_causality summary CSVs "
            "(required for --fig 10/11)."
        ),
    )
    parser.add_argument(
        "--feature-matches",
        default=None,
        help="Cross-scale feature-match CSV (required for --fig 12).",
    )
    parser.add_argument(
        "--feature-match-null",
        default=None,
        help="Permutation-null summary CSV (required for --fig 12).",
    )
    parser.add_argument(
        "--steering-slopes",
        default=None,
        help="Path to gemma_scope_local_steering_slopes.csv (required for --fig p3).",
    )
    parser.add_argument(
        "--dense-csvs",
        default=None,
        help=(
            "Comma-separated steering_dense_<label>.csv summary files, one per model "
            "(required for --fig 13/14/15). Same order as --labels."
        ),
    )
    parser.add_argument(
        "--hiring-audit-csvs",
        default=None,
        help=(
            "Comma-separated hiring_audit_<label>.csv files, one per model "
            "(required for --fig 16/18). Same order as --labels."
        ),
    )
    parser.add_argument(
        "--hiring-steering-csvs",
        default=None,
        help=(
            "Comma-separated hiring_steering_raw_<label>.csv files, one per model "
            "(required for --fig 17). Same order as --labels."
        ),
    )
    parser.add_argument(
        "--hiring-disparity-csvs",
        default=None,
        help=(
            "Comma-separated hiring_disparity_<label>.csv files, one per model "
            "(required for --fig 18). Same order as --labels."
        ),
    )
    parser.add_argument(
        "--hiring-mediation-jsons",
        default=None,
        help=(
            "Comma-separated hiring_mediation_<label>.json files, one per model "
            "(required for --fig 19). Same order as --labels."
        ),
    )
    args = parser.parse_args()

    # Resolve runtime dirs
    if args.vec_dir is not None:
        VEC_DIR = Path(args.vec_dir)
    if args.out_dir is not None:
        OUT_DIR = Path(args.out_dir)

    _style.apply()

    # Parse --fig: supports integers (1-20) and paper-figure tokens p1/p2/p3.
    paper_selected: set[int] = set()
    if args.fig == "all":
        selected = {1, 2, 3, 4}
    else:
        selected = set()
        for tok in args.fig.split(","):
            tok = tok.strip()
            if tok.lower().startswith("p"):
                paper_fig = int(tok[1:])
                if paper_fig not in {1, 2, 3}:
                    parser.error(f"Unknown paper figure token '{tok}'; supported: p1, p2, p3")
                paper_selected.add(paper_fig)
            else:
                selected.add(int(tok))

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
        if not model_labels or (not args.vec_dirs and not args.agreement_csv):
            parser.error("--fig 6 requires --labels and either --agreement-csv or --vec-dirs")
        vec_dirs = [Path(p.strip()) for p in args.vec_dirs.split(",")] if args.vec_dirs else []
        if vec_dirs and len(vec_dirs) != len(model_labels):
            parser.error("--vec-dirs and --labels must have the same number of entries")
        agreement_csv = Path(args.agreement_csv) if args.agreement_csv else None
        fig6_cross_model_story_agreement(vec_dirs, model_labels, agreement_csv=agreement_csv)

    if 7 in selected:
        print("Figure 7: same-story three-model demo …")
        if not args.vec_dirs or not model_labels:
            parser.error("--fig 7 requires --vec-dirs and --labels")
        vec_dirs = [Path(p.strip()) for p in args.vec_dirs.split(",")]
        if len(vec_dirs) != len(model_labels):
            parser.error("--vec-dirs and --labels must have the same number of entries")
        stories_jsonl = Path(args.stories) if args.stories else None
        fig7_same_story_demo(vec_dirs, model_labels, stories_jsonl=stories_jsonl)

    if 8 in selected:
        print("Figure 8: layer emergence curves …")
        if not args.sweep_csvs or not model_labels:
            parser.error("--fig 8 requires --sweep-csvs and --labels")
        sweep_paths = [Path(p.strip()) for p in args.sweep_csvs.split(",")]
        if len(sweep_paths) != len(model_labels):
            parser.error("--sweep-csvs and --labels must have the same number of entries")
        fig8_layer_emergence(sweep_paths, model_labels)

    if 21 in selected:
        print("Figure 8B: enhanced Stage 3B validation …")
        if not args.sweep_csvs or not model_labels:
            parser.error("--fig 21 requires --sweep-csvs and --labels")
        sweep_paths = [Path(p.strip()) for p in args.sweep_csvs.split(",")]
        if len(sweep_paths) != len(model_labels):
            parser.error("--sweep-csvs and --labels must have the same number of entries")
        fig8b_stage3b_validation(sweep_paths, model_labels)

    if 9 in selected:
        print("Figure 9: Gemma Scope decomposition quality …")
        if not args.scope_metrics or not model_labels:
            parser.error("--fig 9 requires --scope-metrics and --labels")
        scope_metric_paths = [
            Path(path.strip()) for path in args.scope_metrics.split(",")
        ]
        if len(scope_metric_paths) != len(model_labels):
            parser.error("--scope-metrics and --labels must have equal lengths")
        fig9_gemma_scope_decomposition(scope_metric_paths, model_labels)

    if selected & {10, 11}:
        if not args.causality_csvs or not model_labels:
            parser.error("--fig 10/11 requires --causality-csvs and --labels")
        causality_paths = [
            Path(path.strip()) for path in args.causality_csvs.split(",")
        ]
        if len(causality_paths) != len(model_labels):
            parser.error("--causality-csvs and --labels must have equal lengths")
        if 10 in selected:
            print("Figure 10: Gemma Scope concept steering …")
            fig10_gemma_scope_steering(causality_paths, model_labels)
        if 11 in selected:
            print("Figure 11: Gemma Scope feature ablation …")
            fig11_gemma_scope_ablation(causality_paths, model_labels)

    if 12 in selected:
        print("Figure 12: Gemma Scope cross-scale feature matching …")
        if not args.feature_matches or not args.feature_match_null:
            parser.error(
                "--fig 12 requires --feature-matches and --feature-match-null"
            )
        fig12_gemma_scope_feature_matching(
            Path(args.feature_matches),
            Path(args.feature_match_null),
        )

    # ------------------------------------------------------------------
    # Paper-draft figures (p1, p2, p3)
    # p1 → --vec-dirs + --labels
    # p2 → --sweep-csvs + --labels  (layer emergence; does NOT need --vec-dirs)
    # p3 → --steering-slopes        (diverging dot-arrow; no vec-dirs needed)
    # ------------------------------------------------------------------
    if 1 in paper_selected:
        print("paper_figure1: axis arrows …")
        if not args.vec_dirs or not model_labels:
            parser.error("--fig p1 requires --vec-dirs and --labels")
        paper_vec_dirs = [Path(p.strip()) for p in args.vec_dirs.split(",")]
        if len(paper_vec_dirs) != len(model_labels):
            parser.error("--vec-dirs and --labels must have the same number of entries")
        paper_figure1_axis_arrows(paper_vec_dirs, model_labels)

    if 2 in paper_selected:
        print("paper_figure2: layer emergence …")
        if not args.sweep_csvs or not model_labels:
            parser.error("--fig p2 requires --sweep-csvs and --labels")
        pf2_sweeps = [Path(p.strip()) for p in args.sweep_csvs.split(",")]
        if len(pf2_sweeps) != len(model_labels):
            parser.error("--sweep-csvs and --labels must have the same number of entries")
        paper_figure2_layer_emergence(pf2_sweeps, model_labels)

    if 3 in paper_selected:
        print("paper_figure3: diverging steering …")
        if not args.steering_slopes:
            parser.error("--fig p3 requires --steering-slopes")
        paper_figure3_diverging_steering(Path(args.steering_slopes))

    if selected & {13, 14, 15}:
        if not args.dense_csvs or not model_labels:
            parser.error("--fig 13/14/15 requires --dense-csvs and --labels")
        dense_paths = [
            Path(p.strip()) for p in args.dense_csvs.split(",")
        ]
        if len(dense_paths) != len(model_labels):
            parser.error("--dense-csvs and --labels must have equal lengths")
        if 13 in selected:
            print("Figure 13: dense steering dose-response …")
            fig13_dense_steering_doseresponse(dense_paths, model_labels)
        if 14 in selected:
            print("Figure 14: dense steering normalized cross-model …")
            fig14_dense_steering_normalized(dense_paths, model_labels)
        if 15 in selected:
            print("Figure 15: dense steering signal vs. control …")
            fig15_dense_steering_signal_vs_control(dense_paths, model_labels)

    if selected & {16, 17, 18, 19}:
        if not model_labels:
            parser.error("--fig 16/17/18/19 requires --labels")
        # Resolve hiring audit paths (needed for 16 and 18)
        audit_paths: list[Path] | None = None
        if args.hiring_audit_csvs:
            audit_paths = [Path(p.strip()) for p in args.hiring_audit_csvs.split(",")]
            if len(audit_paths) != len(model_labels):
                parser.error("--hiring-audit-csvs and --labels must have equal lengths")
        # Resolve steering paths (needed for 17)
        steering_h_paths: list[Path] | None = None
        if args.hiring_steering_csvs:
            steering_h_paths = [Path(p.strip()) for p in args.hiring_steering_csvs.split(",")]
            if len(steering_h_paths) != len(model_labels):
                parser.error("--hiring-steering-csvs and --labels must have equal lengths")
        # Resolve disparity paths (needed for 18)
        disparity_paths: list[Path] | None = None
        if args.hiring_disparity_csvs:
            disparity_paths = [Path(p.strip()) for p in args.hiring_disparity_csvs.split(",")]
            if len(disparity_paths) != len(model_labels):
                parser.error("--hiring-disparity-csvs and --labels must have equal lengths")
        # Resolve mediation paths (needed for 19)
        mediation_paths: list[Path] | None = None
        if args.hiring_mediation_jsons:
            mediation_paths = [Path(p.strip()) for p in args.hiring_mediation_jsons.split(",")]
            if len(mediation_paths) != len(model_labels):
                parser.error("--hiring-mediation-jsons and --labels must have equal lengths")

        if 16 in selected:
            if not audit_paths:
                parser.error("--fig 16 requires --hiring-audit-csvs")
            print("Figure 16: hiring probe-vs-human alignment …")
            fig16_hiring_probe_vs_human(audit_paths, model_labels)
        if 17 in selected:
            if not steering_h_paths:
                parser.error("--fig 17 requires --hiring-steering-csvs")
            print("Figure 17: hiring steering → callback …")
            fig17_hiring_steering_callback(steering_h_paths, model_labels)
        if 18 in selected:
            if not disparity_paths or not audit_paths:
                parser.error("--fig 18 requires --hiring-disparity-csvs and --hiring-audit-csvs")
            print("Figure 18: hiring disparity …")
            fig18_hiring_disparity(disparity_paths, audit_paths, model_labels)
        if 19 in selected:
            if not mediation_paths:
                parser.error("--fig 19 requires --hiring-mediation-jsons")
            print("Figure 19: hiring mediation forest …")
            fig19_hiring_mediation_forest(mediation_paths, model_labels)

    if 20 in selected:
        print("Figure 20: PCA denoising summary …")
        if not args.vec_dirs or not model_labels:
            parser.error("--fig 20 requires --vec-dirs and --labels")
        vec_dirs = [Path(p.strip()) for p in args.vec_dirs.split(",")]
        if len(vec_dirs) != len(model_labels):
            parser.error("--vec-dirs and --labels must have the same number of entries")
        fig20_pca_denoising(vec_dirs, model_labels)

    print("Done.")


if __name__ == "__main__":
    main()
