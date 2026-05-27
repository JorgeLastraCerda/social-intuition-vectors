"""Analysis script for priming experiment results.

Reads a scored CSV (output of emotion_scoring.py) and produces:
  - Group-level EFS statistics by condition
  - Statistical tests (one-way ANOVA / Kruskal-Wallis + pairwise Mann-Whitney)
  - Correlation between projection score and classifier score (cross-validation)
  - Condition plots (saved to output directory)

Usage:
    python -m scripts.analyze_priming --input runs/priming/.../scored.csv
    python -m scripts.analyze_priming --input scored.csv --output reports/priming_analysis/ --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

def group_summary(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    return (
        df.groupby(group_cols)[metric]
        .agg(["count", "mean", "std", "median"])
        .round(4)
        .reset_index()
    )


def kruskal_by_valence(df: pd.DataFrame, metric: str) -> dict:
    groups = [g[metric].dropna().values for _, g in df.groupby("priming_valence")]
    if len(groups) < 2:
        return {"error": "need at least 2 groups"}
    stat, p = stats.kruskal(*groups)
    return {"kruskal_H": round(stat, 4), "p_value": round(p, 6)}


def pairwise_mannwhitney(df: pd.DataFrame, metric: str, group_col: str = "priming_valence") -> pd.DataFrame:
    groups = df.groupby(group_col)[metric].apply(lambda x: x.dropna().values).to_dict()
    keys = list(groups.keys())
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = groups[keys[i]], groups[keys[j]]
            if len(a) < 2 or len(b) < 2:
                continue
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            d = cohens_d(a, b)
            rows.append({
                "group_a": keys[i],
                "group_b": keys[j],
                "U": round(stat, 2),
                "p_value": round(p, 6),
                "cohens_d": round(d, 4),
            })
    return pd.DataFrame(rows)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_sd


def spearman_cross_validation(df: pd.DataFrame) -> dict:
    if "efs_projection" not in df.columns or "efs_classifier" not in df.columns:
        return {"error": "missing columns"}
    valid = df[["efs_projection", "efs_classifier"]].dropna()
    if len(valid) < 5:
        return {"error": "too few rows"}
    r, p = stats.spearmanr(valid["efs_projection"], valid["efs_classifier"])
    return {"spearman_r": round(r, 4), "p_value": round(p, 6), "n": len(valid)}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def make_plots(df: pd.DataFrame, out_dir: Path, metric: str = "efs_composite") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed — skipping plots", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Boxplot: by condition
    cond_col = "condition" if "condition" in df.columns else "priming_valence"
    if cond_col in df.columns:
        conditions_sorted = sorted(df[cond_col].unique())
        fig, ax = plt.subplots(figsize=(max(6, len(conditions_sorted) * 2), 5))
        vals = [df[df[cond_col] == c][metric].dropna().values for c in conditions_sorted]
        ax.boxplot(vals, patch_artist=True, medianprops=dict(color="black"))
        ax.set_xticklabels(conditions_sorted, rotation=15)
        ax.set_ylabel(metric)
        ax.set_title(f"EFS by condition")
        fig.tight_layout()
        fig.savefig(out_dir / "efs_by_condition.png", dpi=150)
        plt.close(fig)
        print(f"[plots] saved efs_by_condition.png", flush=True)

    # Scatter: projection vs classifier
    if "efs_projection" in df.columns and "efs_classifier" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        color_map = {"dark-15": "red", "dark": "red", "neutral-15": "gray", "neutral": "gray",
                     "light-15": "steelblue", "light": "steelblue", "baseline": "green"}
        valid = df[["efs_projection", "efs_classifier", cond_col]].dropna()
        for cond, group in valid.groupby(cond_col):
            ax.scatter(group["efs_projection"], group["efs_classifier"],
                       label=cond, alpha=0.6, color=color_map.get(cond, "black"), s=30)
        ax.set_xlabel("efs_projection (anchor cosine)")
        ax.set_ylabel("efs_classifier (distilroberta fear)")
        ax.set_title("Cross-validation: projection vs classifier")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "efs_crossvalidation.png", dpi=150)
        plt.close(fig)
        print(f"[plots] saved efs_crossvalidation.png", flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze priming experiment scored results.")
    parser.add_argument("--input", required=True, help="Path to scored CSV")
    parser.add_argument("--output", default=None, help="Directory for report files (default: alongside input)")
    parser.add_argument("--metric", default="efs_composite", help="Primary EFS column to analyze.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output) if args.output else input_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] loading {input_path}", flush=True)
    df = pd.read_csv(input_path)
    print(f"[analyze] {len(df)} rows", flush=True)

    metric = args.metric
    if metric not in df.columns:
        available = [c for c in df.columns if "efs" in c.lower()]
        print(f"[analyze] metric '{metric}' not found. Available EFS columns: {available}", flush=True)
        if available:
            metric = available[0]
            print(f"[analyze] falling back to: {metric}", flush=True)
        else:
            print("[analyze] no scorable columns found — exiting", flush=True)
            return

    # Determine condition column: prefer 'condition', fall back to 'priming_valence'
    condition_col = "condition" if "condition" in df.columns else "priming_valence"

    # Group summary
    group_cols = [c for c in [condition_col, "model"] if c in df.columns]
    if group_cols:
        summary = group_summary(df, group_cols, metric)
        print("\n[analyze] Group summary:", flush=True)
        print(summary.to_string(index=False), flush=True)
        summary.to_csv(out_dir / "group_summary.csv", index=False)

    # Kruskal-Wallis by condition
    pairwise = pd.DataFrame()
    if condition_col in df.columns:
        groups = [g[metric].dropna().values for _, g in df.groupby(condition_col)]
        if len(groups) >= 2:
            from scipy import stats as _stats
            stat, p = _stats.kruskal(*groups)
            kruskal = {"kruskal_H": round(stat, 4), "p_value": round(p, 6)}
        else:
            kruskal = {"error": "need at least 2 groups"}
        print(f"\n[analyze] Kruskal-Wallis ({condition_col}): {kruskal}", flush=True)

        # Pairwise Mann-Whitney
        pairwise = pairwise_mannwhitney(df, metric, condition_col)
        print(f"\n[analyze] Pairwise Mann-Whitney U:", flush=True)
        print(pairwise.to_string(index=False), flush=True)
        pairwise.to_csv(out_dir / "pairwise_mannwhitney.csv", index=False)

    # Cross-validation
    cv = spearman_cross_validation(df)
    print(f"\n[analyze] Projection vs classifier cross-validation (Spearman): {cv}", flush=True)

    # Interpretation hint
    # Manipulation check: dark-15 vs light-15 (v2 naming) or dark vs light (v1 naming)
    if "cohens_d" in pairwise.columns and not pairwise.empty:
        dark_light = pairwise[
            pairwise.apply(
                lambda r: set([r["group_a"], r["group_b"]]) in (
                    {"dark-15", "light-15"}, {"dark", "light"}
                ),
                axis=1,
            )
        ]
        if not dark_light.empty:
            d = dark_light.iloc[0]["cohens_d"]
            p = dark_light.iloc[0]["p_value"]
            sig = "significant" if p < 0.05 else "not significant"
            magnitude = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
            print(f"\n[interpret] dark vs light: Cohen's d = {d:.3f} ({magnitude} effect), p = {p:.4f} ({sig})", flush=True)
            if p < 0.05 and abs(d) >= 0.5:
                print("[interpret] MANIPULATION CHECK PASSED: dark vs light EFS difference is significant and medium+.", flush=True)
            else:
                print("[interpret] MANIPULATION CHECK NOTE: effect is small or non-significant — consider strengthening priming protocol.", flush=True)

    if not args.no_plots:
        make_plots(df, out_dir, metric)

    print(f"\n[analyze] reports written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
