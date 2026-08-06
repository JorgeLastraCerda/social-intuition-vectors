"""Reproducible name-level and race-by-gender R4 hiring analysis.

Join granularity
----------------
Prior versions of this script collapsed each rated name to a single row
(``.groupby("name").agg(study=("study", "first"))`` upstream in
``hiring_audit.py``) before requiring an exact (name, study) match against
Carina Hausladen's published callback data. That combination silently
dropped 37 genuine Kline names mislabeled ``flake_leasure`` by a bug in her
own data-prep code (see ``src/utils/human_ratings.py``), and arbitrarily
discarded the second study for any of the 56 names rated under more than one
real study (e.g. "aisha" under both Bertrand, callback 0.022, and Kline,
callback 0.246 — very different numbers, not interchangeable).

This version keeps every valid (name, study) pair as its own row: a name
rated under two studies contributes two rows, matched against that study's
own callback rate, never blended or arbitrarily reduced to one. Carina's own
``published_data/code.R`` groups some studies by age as well as name
(Neumark: ``group_by(name, age, gender)``; Farber: ``group_by(name, age)``),
but our ratings have no age dimension (raters were never asked to rate by
age), so age is averaged away within each (name, study) pair before the
join — there is no finer information on our side to preserve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config
from src.utils.human_ratings import (
    add_zscores,
    full_distribution_stats,
    load_name_study_ratings,
)

PREDICTORS = ("human_callback", "model_warmth", "model_competence")


def load_and_join(audit_csv: Path, raw_data_dir: Path) -> pd.DataFrame:
    """One row per matched (name, study) pair. See module docstring."""
    audit = pd.read_csv(audit_csv)[
        ["name", "model_warmth", "model_competence", "callback_margin"]
    ]

    ratings = load_name_study_ratings(raw_data_dir)
    ratings["first"] = ratings["name"].str.lower().str.split().str[0]

    human_csv = (
        raw_data_dir
        / "SocialPerceptions-Predict-Callback-main"
        / "0_data"
        / "published_data"
        / "df_all.csv"
    )
    human = pd.read_csv(human_csv)
    human["first"] = human["name"].str.lower()
    # Average within (first, study): collapses Neumark/Farber's own internal
    # age-condition rows, since our ratings carry no age dimension to match
    # them against.
    human_by_study = (
        human.groupby(["first", "study"])
        .agg(
            human_callback=("callback", "mean"),
            race=("race", "first"),
            gender=("gender", "first"),
        )
        .reset_index()
    )

    matched = ratings.merge(human_by_study, on=["first", "study"], how="inner")
    matched = matched.merge(audit, on="name", how="left")
    return matched.reset_index(drop=True)


def margin_diagnostic(values: pd.Series) -> dict:
    values = values.dropna().astype(float)
    on_eighth_grid = np.isclose(values * 8.0, np.round(values * 8.0), atol=1e-6)
    sd = float(values.std(ddof=1))
    fraction = float(on_eighth_grid.mean())
    return {
        "n": int(len(values)),
        "n_unique": int(values.nunique()),
        "sd": sd,
        "fraction_on_0.125_grid": fraction,
        "quantisation_warning": bool(fraction > 0.8 and sd < 0.25),
    }


def group_statistics(
    matched: pd.DataFrame, label: str, group_cols: list[str] | None = None
) -> pd.DataFrame:
    """Group ``matched`` by any subset of ``["race", "gender", "study"]``
    (default: all three, i.e. race x gender x study cells, not collapsed
    across study, since callback rates genuinely differ by study and
    blending them is the problem the (name, study) join redesign fixes).
    ``n_names`` counts (name, study) observations in the cell, not distinct
    names; use ``n_distinct_names`` for the latter.

    Includes human warmth/competence alongside the model's own, and callback
    margin/rate — the data for all four was already present in ``matched``
    (human warmth/competence flow in from ``load_name_study_ratings``), this
    just aggregates them instead of leaving them unused.
    """
    if group_cols is None:
        group_cols = ["race", "gender", "study"]
    grouped = (
        matched.groupby(group_cols, dropna=False)
        .agg(
            model_margin_mean=("callback_margin", "mean"),
            model_margin_se=("callback_margin", lambda values: values.sem()),
            model_warmth_mean=("model_warmth", "mean"),
            model_competence_mean=("model_competence", "mean"),
            human_warm_mean=("human_warm", "mean"),
            human_competent_mean=("human_competent", "mean"),
            human_callback=("human_callback", "mean"),
            n_names=("name", "size"),
            n_distinct_names=("name", "nunique"),
        )
        .reset_index()
    )
    grouped["model"] = label
    return grouped


def name_level_statistics(matched: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    columns = ["callback_margin", *PREDICTORS]
    work = matched[columns].dropna()
    if len(work) < 20:
        raise ValueError(f"R4 requires at least 20 matched names; got {len(work)}.")

    X = work[list(PREDICTORS)].to_numpy()
    y = work["callback_margin"].to_numpy()
    scaled = StandardScaler().fit_transform(X)
    regression = LinearRegression().fit(scaled, y)

    rows = []
    for predictor, beta in zip(PREDICTORS, regression.coef_, strict=True):
        r, p = stats.pearsonr(work[predictor], work["callback_margin"])
        rows.append(
            {
                "model": label,
                "predictor": predictor,
                "pearson_r": float(r),
                "pearson_p": float(p),
                "standardized_ols_beta": float(beta),
                "ols_r2": float(regression.score(scaled, y)),
                "n_names": int(len(work)),
            }
        )
    return pd.DataFrame(rows), margin_diagnostic(work["callback_margin"])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    table_dir = Path(cfg.paths.results) / "tables"
    log_dir = Path(cfg.paths.logs)
    audit_csv = table_dir / f"hiring_audit_{args.label}.csv"
    raw_data_dir = Path(cfg.paths.raw_data)
    matched = load_and_join(audit_csv, raw_data_dir)
    group = group_statistics(matched, args.label)
    dist_stats = full_distribution_stats(audit_csv, raw_data_dir)
    group = add_zscores(
        group,
        dist_stats,
        {
            "model_warmth": "model_warmth_mean",
            "model_competence": "model_competence_mean",
            "human_warm": "human_warm_mean",
            "human_competent": "human_competent_mean",
        },
    )
    name_level, diagnostic = name_level_statistics(matched, args.label)
    if len(group) >= 3:
        group_r, group_p = stats.pearsonr(
            group["human_callback"], group["model_margin_mean"]
        )
        group_correlation = {
            "pearson_r": float(group_r),
            "pearson_p": float(group_p),
            "n_groups": int(len(group)),
        }
    else:
        group_correlation = None

    group_path = table_dir / f"hiring_group_r4_{args.label}.csv"
    name_path = table_dir / f"hiring_name_level_{args.label}.csv"
    log_path = log_dir / f"hiring_r4_{args.label}.json"
    group.to_csv(group_path, index=False)
    name_level.to_csv(name_path, index=False)
    log_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "audit_input": str(audit_csv),
                "human_input": str(
                    raw_data_dir
                    / "SocialPerceptions-Predict-Callback-main"
                    / "0_data"
                    / "published_data"
                    / "df_all.csv"
                ),
                "join": (
                    "lowercase first name plus exact study match; one row per "
                    "(name, study) pair, no dedup on name; age averaged away "
                    "within (first, study) on the published-data side"
                ),
                "n_audit": int(len(pd.read_csv(audit_csv))),
                "n_matched": int(len(matched)),
                "n_distinct_names_matched": int(matched["name"].nunique()),
                "seed": cfg.probing.seed,
                "margin_diagnostic": diagnostic,
                "group_level_correlation": group_correlation,
                "group_output": str(group_path),
                "name_level_output": str(name_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[done] {len(matched)} matched (name, study) rows "
        f"({matched['name'].nunique()} distinct names) -> {group_path}, {name_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--label", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
