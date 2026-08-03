"""Build three LaTeX result tables for the manuscript from existing artifacts.

CPU-only. Loads no model and runs no forward pass; every number here is read
from artifacts already on disk (``results/logs/hiring_probe_vs_human_*.json``,
``results/tables/hiring_disparity_*.csv``, ``results/tables/hiring_audit_*.csv``,
``data/processed/<vectors_subdir>/meta.json``) or, for Table 3, recomputed by
re-joining ``hiring_audit_<label>.csv`` against the published human callback
data with the same join used by ``src/hiring_r4.py``.

Outputs (booktabs style, ``\\input``-able from the manuscript)
----------------------------------------------------------------
results/tables/probe_human_correlation_9model.tex
    Table 1 (main text): per-model probe layer and Spearman correlations
    between the model's warmth/competence probe projection and human
    warmth/competence ratings, and between the probe projection and the
    model's own callback margin.
results/tables/hiring_disparity_marginal_9model.tex
    Table 2 (appendix): per-model race (Black/White) and gender
    (Female/Male) marginal group means, from the existing
    ``hiring_disparity_<label>.csv`` tables.
results/tables/hiring_disparity_crossed_9model.tex
    Table 3 (appendix): per-model crossed race x gender group means
    (Black-Female / Black-Male / White-Female / White-Male), re-derived by
    joining ``hiring_audit_<label>.csv`` against
    ``published_data/df_all.csv`` for all nine models. A regression gate
    checks the recomputed callback margin against the five pre-existing
    ``hiring_group_r4_<label>.csv`` files before writing the table.

Usage
-----
    python -m src.build_paper_probe_tables --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.hiring_r4 import load_and_join
from src.utils.config import load_config

# Canonical model order, matching the existing steering-transition census
# table (results/tables/hiring_steering_transition_summary_9model.tex).
MODEL_ORDER = (
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

DISPLAY_NAME = {
    "gemma3_12b": "Gemma-3-12B",
    "gemma3_27b": "Gemma-3-27B",
    "llama31_8b": "Llama-3.1-8B",
    "gemma4_12b": "Gemma-4-12B",
    "gemma4_26b_a4b": "Gemma-4-26B-A4B",
    "gemma4_31b": "Gemma-4-31B",
    "qwen3_14b": "Qwen3-14B",
    "qwen36_27b": "Qwen3.6-27B",
    "qwen36_35b_a3b": "Qwen3.6-35B-A3B",
}

# The five labels for which a pre-existing crossed race x gender table
# (results/tables/hiring_group_r4_<label>.csv) already exists. Used only as
# a regression gate for Table 3; not a scope restriction.
EXISTING_R4_LABELS = (
    "gemma4_12b",
    "gemma4_26b_a4b",
    "gemma4_31b",
    "qwen36_27b",
    "qwen36_35b_a3b",
)


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return " n.s."


def fmt_rho(rho: float, p: float) -> str:
    sign = "+" if rho >= 0 else ""
    return f"{sign}{rho:.3f}{stars(p)}"


# ---------------------------------------------------------------------------
# Table 1 — probe-vs-human correlation, main text
# ---------------------------------------------------------------------------

def build_table1(cfg, log_dir: Path, out_path: Path) -> None:
    rows = []
    n_names = None
    for label in MODEL_ORDER:
        log_path = log_dir / f"hiring_probe_vs_human_{label}.json"
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        corr = {c["pair"]: c for c in payload["correlations"]}

        vectors_subdir = payload["vectors_subdir"]
        meta_path = Path(cfg.paths.processed) / vectors_subdir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        layer = meta["probe_layer"]
        # Always derive the resolved fraction from layer / n_layers rather
        # than reading meta["probe_layer_frac"] (where present, that field
        # stores the config *target* of 0.66, not the integer-rounded
        # per-model result) so the column is consistent and comparable
        # across all nine models.
        frac = layer / meta["n_layers"]

        warmth = corr["warmth"]
        comp = corr["competence"]
        cb_w = corr["callback_vs_model_warmth"]
        cb_c = corr["callback_vs_model_competence"]
        n_names = warmth["n"]

        rows.append(
            {
                "model": DISPLAY_NAME[label],
                "layer": f"{layer} ({frac:.2f})",
                "warmth": fmt_rho(warmth["spearman_rho"], warmth["spearman_p"]),
                "comp": fmt_rho(comp["spearman_rho"], comp["spearman_p"]),
                "cb_warmth": fmt_rho(cb_w["spearman_rho"], cb_w["spearman_p"]),
                "cb_comp": fmt_rho(cb_c["spearman_rho"], cb_c["spearman_p"]),
            }
        )

    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/logs/hiring_probe_vs_human_<label>.json, "
        "data/processed/<vectors_subdir>/meta.json.",
        "% table* spans both columns of the twocolumn body (Section 1 of",
        "% Ulu_Lastra.tex); a single-column table with \\resizebox{\\textwidth}",
        "% would overflow into the adjacent column.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Name-level correlation between probed warmth/competence and "
        r"human ratings.} For each of the "
        f"{n_names}"
        r" rated applicant names, the model's residual-stream activation at "
        r"the probe layer is projected onto the warmth and competence direction "
        r"vectors. ``Probe layer'' gives the selected layer index and its resolved "
        r"fraction of total depth (targeted $\mathrm{probe\_layer\_frac}=0.66$; the "
        r"reported fraction is the integer-rounded layer divided by model depth). "
        r"Warmth/competence "
        r"columns correlate the probe projection with human warmth/competence "
        r"ratings; callback columns correlate the probe projection with the "
        r"model's own unsteered callback margin. All correlations are Spearman "
        r"$\rho$. $^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$, n.s. not significant.}",
        r"\label{tab:probe_human}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Model & Probe layer (frac) & $\rho$(warmth, human) & $\rho$(competence, human) "
        r"& $\rho$(callback, warmth) & $\rho$(callback, competence) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['layer']} & {row['warmth']} & {row['comp']} & "
            f"{row['cb_warmth']} & {row['cb_comp']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table1] wrote {out_path} ({len(rows)} models)")


# ---------------------------------------------------------------------------
# Table 2 — marginal race/gender disparity, appendix
# ---------------------------------------------------------------------------

def build_table2(table_dir: Path, out_path: Path) -> None:
    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/hiring_disparity_<label>.csv.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Name-level warmth/competence and callback margin by marginal "
        r"demographic group.} Race (Black/White) and gender (Female/Male) are treated as "
        r"separate marginal axes, following Gallo and Hausladen's own grouping convention. "
        r"Model warmth/competence are raw (unnormalized) projections onto the concept "
        r"direction and are not comparable in magnitude across models; only within-model "
        r"group differences are meaningful.}",
        r"\label{tab:disparity_marginal}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Model & Group & $n$ & Human callback & Model margin & Model warmth / competence \\",
        r"\midrule",
    ]
    for label in MODEL_ORDER:
        df = pd.read_csv(table_dir / f"hiring_disparity_{label}.csv")
        for i, row in df.iterrows():
            model_cell = DISPLAY_NAME[label] if i == 0 else ""
            lines.append(
                f"{model_cell} & {row['group']} & {int(row['n'])} & "
                f"{row['human_callback']:.3f} & {row['model_callback_margin']:.3f} & "
                f"{row['model_warmth']:.2f} / {row['model_competence']:.2f} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table2] wrote {out_path} ({len(MODEL_ORDER)} models)")


# ---------------------------------------------------------------------------
# Table 3 — crossed race x gender disparity, appendix (re-derived)
# ---------------------------------------------------------------------------

def build_table3(cfg, table_dir: Path, out_path: Path) -> None:
    human_csv = (
        Path(cfg.paths.raw_data)
        / "SocialPerceptions-Predict-Callback-main"
        / "0_data"
        / "published_data"
        / "df_all.csv"
    )

    all_rows = []
    for label in MODEL_ORDER:
        audit_csv = table_dir / f"hiring_audit_{label}.csv"
        matched = load_and_join(audit_csv, human_csv)
        grouped = (
            matched.groupby(["race", "gender"], dropna=False)
            .agg(
                n=("name", "size"),
                human_callback=("human_callback", "mean"),
                model_margin=("callback_margin", "mean"),
                model_warmth=("model_warmth", "mean"),
                model_competence=("model_competence", "mean"),
            )
            .reset_index()
        )
        grouped["label"] = label
        all_rows.append(grouped)

    combined = pd.concat(all_rows, ignore_index=True)

    # Regression gate: recomputed margins for the five labels that already
    # have a hiring_group_r4_<label>.csv must match the pre-existing values.
    mismatches = []
    for label in EXISTING_R4_LABELS:
        existing = pd.read_csv(table_dir / f"hiring_group_r4_{label}.csv")
        new = combined[combined["label"] == label]
        for _, erow in existing.iterrows():
            nrow = new[(new["race"] == erow["race"]) & (new["gender"] == erow["gender"])]
            if nrow.empty:
                mismatches.append(f"{label}: missing group {erow['race']}/{erow['gender']}")
                continue
            got = float(nrow.iloc[0]["model_margin"])
            want = float(erow["model_margin_mean"])
            if abs(got - want) > 1e-6:
                mismatches.append(
                    f"{label} {erow['race']}/{erow['gender']}: "
                    f"recomputed margin {got:.6f} != existing {want:.6f}"
                )
    if mismatches:
        raise RuntimeError(
            "Table 3 regression gate failed against hiring_group_r4_<label>.csv:\n"
            + "\n".join(mismatches)
        )
    print(f"[table3] regression gate passed for {len(EXISTING_R4_LABELS)} labels")

    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/hiring_audit_<label>.csv joined with "
        "data/raw/.../published_data/df_all.csv (src.hiring_r4.load_and_join).",
        "% Regression-gated against results/tables/hiring_group_r4_<label>.csv "
        "for the five labels where that file already existed.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Name-level warmth/competence and callback margin by crossed "
        r"race $\times$ gender group.} Applicant names are joined to "
        r"\citet{gallo2024warmth}'s published race/gender labels by lowercase first name "
        r"and matching study (\texttt{src/hiring\_r4.py}). Model warmth/"
        r"competence are raw projections and are not comparable in magnitude across "
        r"models; only within-model comparisons across the four cells are meaningful.}",
        r"\label{tab:disparity_crossed}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Model & Race $\times$ Gender & $n$ & Human callback & Model margin & "
        r"Model warmth / competence \\",
        r"\midrule",
    ]
    group_order = [
        ("Black", "Female"),
        ("Black", "Male"),
        ("White", "Female"),
        ("White", "Male"),
    ]
    for label in MODEL_ORDER:
        sub = combined[combined["label"] == label]
        for i, (race, gender) in enumerate(group_order):
            row = sub[(sub["race"] == race) & (sub["gender"] == gender)]
            model_cell = DISPLAY_NAME[label] if i == 0 else ""
            if row.empty:
                lines.append(f"{model_cell} & {race}-{gender} & -- & -- & -- & -- \\\\")
                continue
            r = row.iloc[0]
            lines.append(
                f"{model_cell} & {race}-{gender} & {int(r['n'])} & "
                f"{r['human_callback']:.3f} & {r['model_margin']:.3f} & "
                f"{r['model_warmth']:.2f} / {r['model_competence']:.2f} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table3] wrote {out_path} ({len(MODEL_ORDER)} models)")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    table_dir = Path(cfg.paths.results) / "tables"
    log_dir = Path(cfg.paths.logs)

    build_table1(cfg, log_dir, table_dir / "probe_human_correlation_9model.tex")
    build_table2(table_dir, table_dir / "hiring_disparity_marginal_9model.tex")
    build_table3(cfg, table_dir, table_dir / "hiring_disparity_crossed_9model.tex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main()
