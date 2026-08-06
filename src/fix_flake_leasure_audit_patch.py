"""One-time patch: correct human_warm/human_competent/study/n_raters in the
nine canonical hiring_audit_<label>.csv files after fixing the flake_leasure
duplication bug (see src/utils/human_ratings.py, step_logs/STEP_LOG.md
2026-08-06).

CPU-only. Does not touch model_warmth, model_competence, or callback_margin
(GPU-derived, proven unaffected: same 282 names, same prompts, study label
never reaches the model). Asserts those three columns and `name` are
byte-identical before overwriting each file, so a bug in this script cannot
silently corrupt GPU-derived data.

Also recomputes the matching results/logs/hiring_probe_vs_human_<label>.json
(Spearman/Pearson correlations) from the patched CSV, since the correlation
depends on human_warm/human_competent.

Usage
-----
    python -m src.fix_flake_leasure_audit_patch --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.utils.config import load_config
from src.utils.human_ratings import load_name_ratings_collapsed

LABELS = (
    "gemma3_12b",
    "gemma3_27b",
    "llama31_8b",
    "gemma4_12b",
    "gemma4_26b_a4b",
    "gemma4_31b",
    "qwen3_14b",
    "qwen36_27b",
    "qwen36_35b_a3b",
)

UNCHANGED_COLUMNS = ("name", "model_warmth", "model_competence", "callback_margin")


def patch_one(label: str, table_dir: Path, log_dir: Path, raw_data_dir: Path) -> None:
    audit_csv = table_dir / f"hiring_audit_{label}.csv"
    old = pd.read_csv(audit_csv)

    fixed_ratings = load_name_ratings_collapsed(raw_data_dir)
    new = old[list(UNCHANGED_COLUMNS)].merge(
        fixed_ratings[["name", "human_warm", "human_competent", "study", "n_raters"]],
        on="name",
        how="left",
    )
    # column order matches the original file
    new = new[
        [
            "name",
            "human_warm",
            "human_competent",
            "study",
            "n_raters",
            "model_warmth",
            "model_competence",
            "callback_margin",
        ]
    ]

    for col in UNCHANGED_COLUMNS:
        if not old[col].equals(new[col]):
            raise AssertionError(
                f"{label}: column {col!r} changed during patch; this must never happen."
            )
    if new.isna().any().any():
        raise AssertionError(f"{label}: patch introduced NaN values.")
    if len(new) != len(old):
        raise AssertionError(f"{label}: row count changed ({len(old)} -> {len(new)}).")

    changed_human_warm = int((~old["human_warm"].round(6).eq(new["human_warm"].round(6))).sum())
    changed_study = int((old["study"] != new["study"]).sum())
    new.to_csv(audit_csv, index=False)
    print(
        f"[{label}] patched {audit_csv}: "
        f"human_warm changed for {changed_human_warm}/{len(new)} names, "
        f"study label changed for {changed_study}/{len(new)} names"
    )

    # --- recompute probe-vs-human correlations from the patched CSV ---
    pairs = [
        ("model_warmth", "human_warm", "warmth"),
        ("model_competence", "human_competent", "competence"),
        ("callback_margin", "model_warmth", "callback_vs_model_warmth"),
        ("callback_margin", "model_competence", "callback_vs_model_competence"),
        ("callback_margin", "human_warm", "callback_vs_human_warm"),
        ("callback_margin", "human_competent", "callback_vs_human_competent"),
    ]
    corr_results = []
    for col_x, col_y, pair_label in pairs:
        x = new[col_x].to_numpy()
        y = new[col_y].to_numpy()
        rho, p_s = spearmanr(x, y)
        r, p_p = pearsonr(x, y)
        corr_results.append(
            {
                "pair": pair_label,
                "col_x": col_x,
                "col_y": col_y,
                "spearman_rho": round(float(rho), 4),
                "spearman_p": float(p_s),
                "pearson_r": round(float(r), 4),
                "pearson_p": float(p_p),
                "n": len(x),
            }
        )

    log_path = log_dir / f"hiring_probe_vs_human_{label}.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))
    old_corr = {c["pair"]: c["spearman_rho"] for c in log["correlations"]}
    log["correlations"] = corr_results
    log["flake_leasure_fix_note"] = (
        "human_warm/human_competent/study/n_raters recomputed after dropping "
        "the flake_leasure duplicate rows (kline mislabeled bug in Carina "
        "Hausladen's own ratings/names/code.R); model_warmth/model_competence/"
        "callback_margin unchanged (GPU-derived, verified byte-identical)."
    )
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    new_corr = {c["pair"]: c["spearman_rho"] for c in corr_results}
    print(
        f"[{label}] warmth rho: {old_corr['warmth']:+.4f} -> {new_corr['warmth']:+.4f}   "
        f"competence rho: {old_corr['competence']:+.4f} -> {new_corr['competence']:+.4f}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    table_dir = Path(cfg.paths.results) / "tables"
    log_dir = Path(cfg.paths.logs)
    raw_data_dir = Path(cfg.paths.raw_data)

    for label in LABELS:
        patch_one(label, table_dir, log_dir, raw_data_dir)
    print(f"[done] patched {len(LABELS)} labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main()
