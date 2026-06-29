"""Hiring disparity analysis with bootstrap mediation. CPU-only; no model load.

Implements the gated Phase 7 analysis (D-Phase7-A/B/C):
  A — human benchmark: name-level real callback rates from published_data/df_all.csv
  B — grouping: race (Black/White, primary) and gender (Female/Male, secondary),
      mirroring Gallo & Hausladen's own published_data/code.R grouping convention
  C — bootstrap mediation: name group → model probe score → callback margin

Input
-----
results/tables/hiring_audit_<label>.csv   (produced by src/hiring_audit.py)
data/raw/.../0_data/published_data/df_all.csv

Join key: first token of name (lowercased) ↔ published name (lowercased).
The published data uses single-token first names (e.g. "aisha", "allison");
our rated names use full names (e.g. "aisha", "allison johnson").
~190 of 282 rated names are expected to join.

Outputs
-------
results/tables/hiring_disparity_<label>.csv
    Per-group mean model callback margin and mean human callback rate,
    for each axis (race, gender).
results/logs/hiring_mediation_<label>.json
    Bootstrap mediation results (indirect effect, 95% CI, proportion mediated)
    for race × warmth and race × competence (and gender × warmth/competence).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Bootstrap mediation helpers
# ---------------------------------------------------------------------------

def _ols_coef(X: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y on X (single predictor, adds intercept)."""
    X_ = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X_, y, rcond=None)
    return float(coef[1])


def bootstrap_mediation(
    x: np.ndarray,
    m: np.ndarray,
    y: np.ndarray,
    n_boot: int = 5000,
    seed: int = 20260527,
) -> dict:
    """
    Baron–Kenny / Preacher–Hayes bootstrap mediation.

    x — binary treatment (0/1), m — mediator, y — outcome; all standardised.
    indirect = a * b  where  a = coef(M ~ X), b = coef(Y ~ M + X).
    Returns dict with point estimate, 95% CI, proportion mediated, n.
    """
    # standardise
    def _std(v: np.ndarray) -> np.ndarray:
        sd = v.std(ddof=1)
        return (v - v.mean()) / sd if sd > 0 else v - v.mean()

    xs, ms, ys = _std(x), _std(m), _std(y)
    n = len(xs)

    # observed coefficients
    a = _ols_coef(xs, ms)
    # partial regression of y on m controlling for x
    Xmy = np.column_stack([np.ones(n), ms, xs])
    coef_b, _, _, _ = np.linalg.lstsq(Xmy, ys, rcond=None)
    b = float(coef_b[1])
    indirect_obs = a * b

    total = _ols_coef(xs, ys)
    prop = indirect_obs / total if abs(total) > 1e-10 else float("nan")

    # bootstrap
    rng = np.random.default_rng(seed)
    boot_indirect: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, mb, yb = xs[idx], ms[idx], ys[idx]
        a_b = _ols_coef(xb, mb)
        Xmyb = np.column_stack([np.ones(n), mb, xb])
        cb, _, _, _ = np.linalg.lstsq(Xmyb, yb, rcond=None)
        boot_indirect.append(a_b * float(cb[1]))

    arr = np.array(boot_indirect)
    ci_lo = float(np.percentile(arr, 2.5))
    ci_hi = float(np.percentile(arr, 97.5))
    significant = ci_lo > 0 or ci_hi < 0

    return {
        "a": round(a, 4),
        "b": round(b, 4),
        "indirect_effect": round(indirect_obs, 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
        "total_effect": round(total, 4),
        "proportion_mediated": round(prop, 4) if not np.isnan(prop) else None,
        "significant_95": bool(significant),
        "n_boot": n_boot,
        "n": n,
    }


def main() -> None:
    args = parse_args()
    cfg_path = args.config

    # read config just for paths (no model needed)
    from src.utils.config import load_config
    cfg = load_config(cfg_path)

    table_dir = Path(cfg.paths.results) / "tables"
    log_dir = Path(cfg.paths.logs)
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- load model audit table ---
    audit_csv = table_dir / f"hiring_audit_{args.label}.csv"
    if not audit_csv.exists():
        raise FileNotFoundError(
            f"{audit_csv} not found. Run src/hiring_audit.py --label {args.label} first."
        )
    audit = pd.read_csv(audit_csv)
    print(f"[input] audit table: {len(audit)} rows from {audit_csv}", flush=True)

    # --- load published human callback data ---
    pub_csv = (
        Path(cfg.paths.raw_data)
        / "SocialPerceptions-Predict-Callback-main"
        / "0_data"
        / "published_data"
        / "df_all.csv"
    )
    pub = pd.read_csv(pub_csv)
    # collapse to one row per first-name × race × gender
    pub["first"] = pub["name"].str.split().str[0].str.lower()
    pub_agg = (
        pub.groupby("first")
        .agg(human_callback=("callback", "mean"), race=("race", "first"), gender=("gender", "first"))
        .reset_index()
    )
    print(
        f"[human] published_data: {len(pub)} rows → {len(pub_agg)} unique first-names",
        flush=True,
    )

    # --- join on first name ---
    audit["first"] = audit["name"].str.split().str[0].str.lower()
    merged = audit.merge(pub_agg, on="first", how="inner")
    print(
        f"[join] {len(merged)} / {len(audit)} rated names matched to published callback data",
        flush=True,
    )
    if len(merged) == 0:
        raise ValueError(
            "Join produced 0 rows. Check name formats in audit table and published_data."
        )

    # --- disparity tables ---
    disparity_rows: list[dict] = []

    for axis_col, axis_label in [
        ("race", "race (Black/White)"),
        ("gender", "gender (Female/Male)"),
    ]:
        grp = (
            merged.dropna(subset=[axis_col])
            .groupby(axis_col)
            .agg(
                model_callback_margin=("callback_margin", "mean"),
                model_warmth=("model_warmth", "mean"),
                model_competence=("model_competence", "mean"),
                human_callback=("human_callback", "mean"),
                n=("name", "size"),
            )
            .reset_index()
        )
        for _, row in grp.iterrows():
            disparity_rows.append(
                {
                    "axis": axis_col,
                    "group": row[axis_col],
                    "n": int(row["n"]),
                    "model_callback_margin": round(float(row["model_callback_margin"]), 4),
                    "model_warmth": round(float(row["model_warmth"]), 4),
                    "model_competence": round(float(row["model_competence"]), 4),
                    "human_callback": round(float(row["human_callback"]), 4),
                }
            )
        print(f"\n[disparity] {axis_label}:", flush=True)
        for _, row in grp.iterrows():
            print(
                f"  {row[axis_col]:10s}  n={row['n']:3.0f}  "
                f"model_margin={row['model_callback_margin']:+.3f}  "
                f"human_callback={row['human_callback']:.3f}",
                flush=True,
            )

    disp_df = pd.DataFrame(disparity_rows)
    disp_path = table_dir / f"hiring_disparity_{args.label}.csv"
    disp_df.to_csv(disp_path, index=False)
    print(f"\n[done] {disp_path}", flush=True)

    # --- bootstrap mediation ---
    # primary axis: race (Black=1, White=0)
    race_df = merged.dropna(subset=["race"]).copy()
    race_df = race_df[race_df["race"].isin(["Black", "White"])]
    if len(race_df) < 20:
        print(
            f"[warn] only {len(race_df)} names with race∈{{Black,White}} — "
            "mediation results may be unstable",
            flush=True,
        )

    race_x = (race_df["race"] == "Black").astype(float).to_numpy()

    mediation_results: list[dict] = []
    for probe_col, probe_label in [
        ("model_warmth", "warmth"),
        ("model_competence", "competence"),
    ]:
        for grouping, grp_label, x_arr, df_ in [
            ("race", "race(Black=1)", race_x, race_df),
            ("gender", "gender(Female=1)",
             (merged.dropna(subset=["gender"])["gender"] == "Female").astype(float).to_numpy(),
             merged.dropna(subset=["gender"])),
        ]:
            m = df_[probe_col].to_numpy()
            y = df_["callback_margin"].to_numpy()
            x = (df_[grouping] == ("Black" if grouping == "race" else "Female")).astype(float).to_numpy()
            if len(np.unique(x)) < 2:
                print(f"[mediation] skipping {grouping}×{probe_label}: only one group", flush=True)
                continue
            res = bootstrap_mediation(x, m, y, n_boot=args.n_boot, seed=args.seed)
            res.update(
                {
                    "grouping": grp_label,
                    "probe": probe_label,
                    "treatment_encoding": f"{grouping}=1 for Black/Female, 0 for White/Male",
                }
            )
            mediation_results.append(res)
            sig = "✓ significant" if res["significant_95"] else "✗ n.s."
            print(
                f"[mediation] {grp_label:25s} × {probe_label:12s}  "
                f"indirect={res['indirect_effect']:+.4f}  "
                f"95%CI=[{res['ci_95_lo']:+.4f},{res['ci_95_hi']:+.4f}]  {sig}",
                flush=True,
            )

    med_log = {
        "label": args.label,
        "n_matched": len(merged),
        "n_race_subset": int(len(race_df)),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "mediation": mediation_results,
        "disparity_output": str(disp_path),
    }
    med_path = log_dir / f"hiring_mediation_{args.label}.json"
    med_path.write_text(json.dumps(med_log, indent=2), encoding="utf-8")
    print(f"[done] {med_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hiring disparity analysis with bootstrap mediation. "
            "CPU-only — does not load a language model. "
            "Requires src/hiring_audit.py output to exist first."
        )
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--label",
        required=True,
        help="Short identifier matching the hiring_audit run (e.g. gemma3_12b).",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for mediation (default 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260527,
        help="Random seed for bootstrap (default matches cfg.probing.seed).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
