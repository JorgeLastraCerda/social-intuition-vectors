"""Recompute the data-quality-audit statistics for
``paper/2026-06-27_1757_probe_human_data_audit.md`` after the flake_leasure/kline
duplication-bug fix (``src/utils/human_ratings.py``, ``step_logs/STEP_LOG.md``
2026-08-06, Step 37).

No pre-existing script or notebook cell produced this report's tables (checked
``src/`` and ``notebooks/07_hiring_audit.ipynb``); they were originally computed
ad hoc. This script reproduces and updates every number in that report so the
computation is reproducible going forward, and always reads through
``src.utils.human_ratings`` so the flake_leasure bug cannot be silently
reintroduced.

CPU-only. Loads no model; reads ``ratings/names/df_all.csv`` (raw, via the fixed
loader) and the already-computed ``results/tables/hiring_audit_<label>.csv``
files (existing ``model_warmth``/``model_competence`` columns, never
recomputed).

Prints every table in the report's existing Markdown format to stdout, so the
output can be pasted directly into the report.

Usage
-----
    python -m src.build_probe_human_data_audit_stats
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.utils.human_ratings import _load_deduplicated_raw

RAW_DATA_DIR = Path("data/raw")
LABELS = ("gemma3_12b", "gemma3_27b", "llama31_8b", "qwen3_14b")
THRESHOLDS = (0, 5, 10, 20, 100)


def per_study_table(raw: pd.DataFrame) -> None:
    print("| Study | Rows | Unique names | Unique raters |")
    print("|-------|-----:|-------------:|--------------:|")
    for study, group in raw.groupby("study"):
        print(
            f"| {study} | {len(group):,} | {group['name'].nunique()} | "
            f"{group['ResponseId'].nunique()} |"
        )
    print(
        "\n(No `flake` or `leasure` row: the real Flake/Leasure name ratings "
        "were never produced by Carina Hausladen's own data-prep code, since "
        "the `flake_leasure` block reused the Kline columns by mistake; see "
        "`src/utils/human_ratings.py` module docstring.)"
    )


def headline_stats(raw: pd.DataFrame) -> None:
    n_rows = len(raw)
    n_names = raw["name"].nunique()
    n_raters = raw["ResponseId"].nunique()
    n_studies = raw["study"].nunique()
    n_missing_competent = raw["competent"].isna().sum()
    n_missing_warm = raw["warm"].isna().sum()
    print("| Dimension | Value | Assessment |")
    print("|-----------|------:|------------|")
    print(f"| Human rating rows | {n_rows:,} | Strong |")
    print(f"| Unique names | {n_names} | Good coverage for name-level audit |")
    print(f"| Unique raters | {n_raters} | Strong |")
    print(f"| Source studies | {n_studies} | Strong provenance |")
    print("| Human warmth scale | 0-100 | Good continuous range |")
    print("| Human competence scale | 0-100 | Good continuous range |")
    print(f"| Missing `warm` values | {n_missing_warm} | Clean |")
    print(
        f"| Missing `competent` values | {n_missing_competent} / {n_rows:,} | "
        "Negligible |"
    )


def distribution_and_cross_correlation(raw: pd.DataFrame) -> None:
    print("| Variable | Min | Max | Mean | SD |")
    print("|----------|----:|----:|-----:|---:|")
    for col in ("warm", "competent"):
        s = raw[col].dropna()
        print(f"| `{col}` | {s.min():.0f} | {s.max():.0f} | {s.mean():.2f} | {s.std():.2f} |")

    per_name = raw.groupby("name").agg(warm=("warm", "mean"), competent=("competent", "mean"))
    pear_r, _ = pearsonr(per_name["warm"], per_name["competent"])
    spear_r, _ = spearmanr(per_name["warm"], per_name["competent"])
    print()
    print("| Correlation | Value |")
    print("|-------------|------:|")
    print(f"| Pearson | {pear_r:+.3f} |")
    print(f"| Spearman | {spear_r:+.3f} |")


def rating_count_imbalance(raw: pd.DataFrame) -> None:
    counts = raw.groupby("name").size()
    print("| Per-name rater count | Value |")
    print("|----------------------|------:|")
    print(f"| Mean | {counts.mean():.1f} |")
    print(f"| Median | {counts.median():.1f} |")
    print(f"| Minimum | {counts.min()} |")
    print(f"| Maximum | {counts.max()} |")
    print(f"| Names with 1 rating | {(counts == 1).sum()} |")
    print(f"| Names with <5 ratings | {(counts < 5).sum()} |")
    print(f"| Names with <10 ratings | {(counts < 10).sum()} |")
    print(f"| Names with <20 ratings | {(counts < 20).sum()} |")
    print(f"| Names with >=200 ratings | {(counts >= 200).sum()} |")


def multi_study_names(raw: pd.DataFrame) -> int:
    n_studies_per_name = raw.groupby("name")["study"].nunique()
    return int((n_studies_per_name > 1).sum())


def robustness_table(raw: pd.DataFrame) -> None:
    counts = raw.groupby("name").size().rename("n_raters")
    per_name = raw.groupby("name").agg(warm=("warm", "mean"), competent=("competent", "mean"))
    per_name = per_name.join(counts)

    for label in LABELS:
        audit = pd.read_csv(f"results/tables/hiring_audit_{label}.csv")[
            ["name", "model_warmth", "model_competence"]
        ]
        merged = audit.merge(per_name, on="name", how="inner")
        print(f"\n### {label}")
        print()
        print("| Filter | N names | Warmth rho | Competence rho |")
        print("|--------|--------:|-----------:|---------------:|")
        for k in THRESHOLDS:
            sub = merged[merged["n_raters"] >= k] if k > 0 else merged
            rw, _ = spearmanr(sub["model_warmth"], sub["warm"])
            rc, _ = spearmanr(sub["model_competence"], sub["competent"])
            filt_label = "all names" if k == 0 else f"n >= {k}"
            print(f"| {filt_label} | {len(sub)} | {rw:+.3f} | {rc:+.3f} |")


def main() -> None:
    raw = _load_deduplicated_raw(RAW_DATA_DIR)

    print("=" * 70)
    print("PER-STUDY TABLE (Section 2)")
    print("=" * 70)
    per_study_table(raw)

    print()
    print("=" * 70)
    print("STRUCTURAL AUDIT — headline stats (Section 3)")
    print("=" * 70)
    headline_stats(raw)

    print()
    print("=" * 70)
    print("STRUCTURAL AUDIT — distribution and cross-correlation (Section 3)")
    print("=" * 70)
    distribution_and_cross_correlation(raw)

    print()
    print("=" * 70)
    print("RATING-COUNT IMBALANCE (Section 4)")
    print("=" * 70)
    rating_count_imbalance(raw)

    n_multi = multi_study_names(raw)
    print(f"\nNames rated under more than one source study: {n_multi}")

    print()
    print("=" * 70)
    print("ROBUSTNESS TO BETTER-RATED NAMES (Section 5)")
    print("=" * 70)
    robustness_table(raw)


if __name__ == "__main__":
    main()
