"""Load Gallo & Hausladen name ratings, correcting a duplication bug in their
own data-preparation code.

``0_data/ratings/names/df_all.csv`` (from
https://github.com/carinahausladen/SocialPerceptions-Predict-Callback) has a
copy-paste bug in Carina Hausladen's own ``0_data/ratings/names/code.R``
(lines 160-175, verified against a fresh clone and an independent
``raw.githubusercontent.com`` download, both byte-identical to our local
copy): the block meant to build ``df_flake_leasure`` from the newly-extracted
Leasure survey columns instead reuses the earlier ``df_temp_k`` (Kline)
variable. The result is that the ``flake_leasure`` study label in the ratings
file is a byte-for-byte duplicate of ``kline`` (same 76 names, same 7,663
rows, same warm/competent values), and the five genuine Flake (2019) and
Leasure (2020) name ratings were never produced. See
``step_logs/STEP_LOG.md`` (2026-08-06) and
``paper/2026-07-20_1935_probe_human_result_tables.md`` for the full
discovery and verification trail.

Both loaders below drop ``study == "flake_leasure"`` rows before aggregating,
which is the complete fix: it removes the duplicate-weighting of Kline for
any name also rated under a second real study, and leaves every genuine
Kline name correctly labeled ``"kline"``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_RATINGS_RELATIVE_PATH = (
    "SocialPerceptions-Predict-Callback-main",
    "0_data",
    "ratings",
    "names",
    "df_all.csv",
)


def _ratings_csv_path(raw_data_dir: Path) -> Path:
    path = raw_data_dir
    for part in _RATINGS_RELATIVE_PATH:
        path = path / part
    return path


def _load_deduplicated_raw(raw_data_dir: Path) -> pd.DataFrame:
    """Read the ratings CSV and drop the flake_leasure duplicate rows."""
    raw = pd.read_csv(_ratings_csv_path(raw_data_dir))
    return raw[raw["study"] != "flake_leasure"].copy()


def load_name_study_ratings(raw_data_dir: Path) -> pd.DataFrame:
    """One row per (name, study) pair: human_warm, human_competent, n_raters.

    A name rated under more than one real study (56 of the 282 rated names,
    e.g. "aisha" under both Bertrand and Kline) yields one row per study, each
    with that study's own warm/competent mean — no study is picked as a
    "winner" and none is discarded. There is no age dimension on our side
    (raters were never asked to rate by age), so this is already the finest
    granularity our data supports; callers that need to match Carina's own
    published callback data (which does vary some studies, e.g. Neumark and
    Farber, by age) should pre-aggregate her data to (first name, study)
    before joining, since our single per-study rating has no finer structure
    to match against.
    """
    raw = _load_deduplicated_raw(raw_data_dir)
    return (
        raw.groupby(["name", "study"])
        .agg(
            human_warm=("warm", "mean"),
            human_competent=("competent", "mean"),
            n_raters=("warm", "size"),
        )
        .reset_index()
    )


def load_name_ratings_collapsed(raw_data_dir: Path) -> pd.DataFrame:
    """One row per name (282 total): human_warm/human_competent averaged
    across all of that name's real (non-flake_leasure) studies, with a single
    representative ``study`` label (alphabetically first among that name's
    studies). Use this where per-study granularity is not needed: picking the
    GPU-evaluation name roster (the model only ever sees the bare name, never
    the study), or the probe-vs-human correlation (not a callback comparison,
    so there is no structural reason to split it by study).
    """
    raw = _load_deduplicated_raw(raw_data_dir)
    return (
        raw.groupby("name")
        .agg(
            human_warm=("warm", "mean"),
            human_competent=("competent", "mean"),
            study=("study", "first"),
            n_raters=("warm", "size"),
        )
        .reset_index()
    )


def full_distribution_stats(audit_csv: Path, raw_data_dir: Path) -> dict[str, tuple[float, float]]:
    """(mean, SD) for ``model_warmth``, ``model_competence`` (over the full
    282-name audit for one model) and ``human_warm``, ``human_competent``
    (over the full 282-name collapsed rating set, the same for every model),
    for z-scoring group-level means onto a comparable scale.

    Human ratings are 0-100 Likert-style averages; model warmth/competence
    are unbounded raw residual-stream projections (e.g. tens of thousands in
    magnitude) — not the same units, and not directly comparable as raw
    numbers. Standardizing both to "SD above/below the full 282-name mean"
    mirrors the callback-margin standardization already used elsewhere in
    the paper ("standardized by the within-model standard deviation... so
    that models with different logit scales can be compared on equal
    footing") and makes "does the model lean the same direction as human
    perception for this group" a direct sign/magnitude comparison.
    """
    audit = pd.read_csv(audit_csv)
    ratings = load_name_ratings_collapsed(raw_data_dir)
    return {
        "model_warmth": (
            float(audit["model_warmth"].mean()),
            float(audit["model_warmth"].std()),
        ),
        "model_competence": (
            float(audit["model_competence"].mean()),
            float(audit["model_competence"].std()),
        ),
        "human_warm": (
            float(ratings["human_warm"].mean()),
            float(ratings["human_warm"].std()),
        ),
        "human_competent": (
            float(ratings["human_competent"].mean()),
            float(ratings["human_competent"].std()),
        ),
    }


def add_zscores(
    grouped: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    columns: dict[str, str],
) -> pd.DataFrame:
    """Add a ``<key>_z`` column for each ``key -> column_name`` pair in
    ``columns``, standardizing ``grouped[column_name]`` against the
    ``(mean, sd)`` in ``stats[key]`` (from :func:`full_distribution_stats`).
    Does not modify existing columns.
    """
    grouped = grouped.copy()
    for key, col in columns.items():
        mean, sd = stats[key]
        grouped[f"{key}_z"] = (grouped[col] - mean) / sd
    return grouped
