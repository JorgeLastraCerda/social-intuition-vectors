"""Reproducible name-level and race-by-gender R4 hiring analysis."""

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

PREDICTORS = ("human_callback", "model_warmth", "model_competence")


def load_and_join(audit_csv: Path, human_csv: Path) -> pd.DataFrame:
    audit = pd.read_csv(audit_csv)
    human = pd.read_csv(human_csv)
    audit["first"] = audit["name"].str.lower().str.split().str[0]
    human["first"] = human["name"].str.lower()
    merged = audit.merge(
        human[["first", "race", "gender", "callback", "study"]].rename(
            columns={"callback": "human_callback", "study": "human_study"}
        ),
        on="first",
        how="left",
    )
    matched = merged[merged["study"] == merged["human_study"]].copy()
    return matched.drop_duplicates(subset=["name"]).reset_index(drop=True)


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


def group_statistics(matched: pd.DataFrame, label: str) -> pd.DataFrame:
    grouped = (
        matched.groupby(["race", "gender"], dropna=False)
        .agg(
            model_margin_mean=("callback_margin", "mean"),
            model_margin_se=("callback_margin", lambda values: values.sem()),
            human_callback=("human_callback", "mean"),
            n_names=("name", "size"),
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
    human_csv = (
        Path(cfg.paths.raw_data)
        / "SocialPerceptions-Predict-Callback-main"
        / "0_data"
        / "published_data"
        / "df_all.csv"
    )
    matched = load_and_join(audit_csv, human_csv)
    group = group_statistics(matched, args.label)
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
                "human_input": str(human_csv),
                "join": "lowercase first name plus exact study match",
                "n_audit": int(len(pd.read_csv(audit_csv))),
                "n_matched": int(len(matched)),
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
    print(f"[done] {len(matched)} matched names -> {group_path}, {name_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--label", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
