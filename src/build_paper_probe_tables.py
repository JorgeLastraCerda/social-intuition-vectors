"""Build the manuscript's LaTeX result tables from existing artifacts.

CPU-only. Loads no model and runs no forward pass; every number here is read
from artifacts already on disk (``results/logs/hiring_probe_vs_human_*.json``,
``results/tables/hiring_disparity_*.csv``, ``results/tables/hiring_audit_*.csv``,
``data/processed/<vectors_subdir>/meta.json``) or recomputed by re-joining
``hiring_audit_<label>.csv`` against the published human callback data with
the same join used by ``src/hiring_r4.py``.

Outputs (booktabs/longtable style, ``\\input``-able from the manuscript)
--------------------------------------------------------------------------
results/tables/probe_human_correlation_9model.tex
    Table 1 (main text): per-model probe layer and Spearman correlations
    between the model's warmth/competence probe projection and human
    warmth/competence ratings, and between the probe projection and the
    model's own callback margin.
results/tables/hiring_disparity_marginal_9model.tex
    Table 2 (main text): per-model race (Black/White) and gender
    (Female/Male) marginal groups, model warmth/competence and human
    warmth/competence both standardized to z-scores for direct comparison.
results/tables/hiring_disparity_race_gender_9model.tex
    Table 2c (main text): the same idea, crossed race x gender groups
    (collapsed across source study), z-scored.
results/tables/hiring_disparity_marginal_raw_9model.tex
    Table S.2b (appendix): raw (unstandardized) companion to Table 2, broken
    down by source study instead of collapsed across it.
results/tables/hiring_disparity_crossed_9model.tex
    Table S.3 (appendix, longtable): per-model crossed race x gender x study
    group means (e.g. Black-Female x Kline, Black-Female x Bertrand, ...),
    raw values, re-derived by joining ``hiring_audit_<label>.csv`` against
    ``published_data/df_all.csv`` for all nine models, one row per
    (name, study) pair (src.hiring_r4.load_and_join; no study is picked as a
    "winner" for names rated under more than one study). A regression gate
    checks the recomputed callback margin against the nine
    ``hiring_group_r4_<label>.csv`` files before writing the table. Uses
    ``longtable`` (not ``table``) since it no longer fits one page.

Usage
-----
    python -m src.build_paper_probe_tables --config config/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.hiring_r4 import group_statistics, load_and_join
from src.utils.config import load_config
from src.utils.human_ratings import add_zscores, full_distribution_stats

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

# Display order for the study column within each race-gender group.
STUDY_ORDER = ("bertrand", "kline", "farber", "neumark")


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
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
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
        r"\end{table*}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table1] wrote {out_path} ({len(rows)} models)")


# ---------------------------------------------------------------------------
# Table 2 — marginal race/gender bias, main text (z-scored model-vs-human)
# ---------------------------------------------------------------------------

def build_table2(table_dir: Path, out_path: Path) -> None:
    """Race (Black/White) and gender (Female/Male) marginal groups, model
    warmth/competence and human warmth/competence both standardized to
    z-scores (SD above/below the full 282-name mean) so the two are directly
    comparable despite living on different raw scales (human ratings are
    0-100 Likert averages; model warmth/competence are unbounded raw
    residual-stream projections). Raw values are kept in the appendix
    (Table S.2b) for reference.
    """
    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/hiring_disparity_<label>.csv (race/gender rows,",
        "% human_warm/human_competent and z-score columns added by src/hiring_disparity.py).",
        "% table* + resizebox: this table sits in the twocolumn Results body (like Table 1),",
        "% and a plain single-column table environment overflows into the adjacent column.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Model & Group & $n$ & Model warmth/competence ($z$) & Human warmth/competence ($z$) "
        r"& Human callback & Model margin \\",
        r"\midrule",
    ]
    for label in MODEL_ORDER:
        df = pd.read_csv(table_dir / f"hiring_disparity_{label}.csv")
        for i, row in df.iterrows():
            model_cell = DISPLAY_NAME[label] if i == 0 else ""
            lines.append(
                f"{model_cell} & {row['group']} & {int(row['n'])} & "
                f"{row['model_warmth_z']:+.2f} / {row['model_competence_z']:+.2f} & "
                f"{row['human_warm_z']:+.2f} / {row['human_competent_z']:+.2f} & "
                f"{row['human_callback']:.3f} & {row['model_callback_margin']:.3f} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Warmth/competence bias by marginal demographic group, model "
        r"versus human, standardized.} Race (Black/White) and gender (Female/Male) are "
        r"treated as separate marginal axes, following Gallo and Hausladen's own grouping "
        r"convention. ``Model margin'' is the model's own Yes/No logit difference on the "
        r"hiring prompt (unsteered); ``Human callback'' is the real observed callback rate "
        r"from the correspondence-study benchmark. Model and human warmth/competence are "
        r"each standardized to $z$-scores against the full 282-name distribution (SD above "
        r"or below the overall mean), the same convention already used for callback-margin "
        r"standardization elsewhere in this paper, so a positive value in both the model "
        r"and human column for a group indicates the model's internal representation leans "
        r"the same direction as human perception; raw (unstandardized) values are given in "
        r"\autoref{tab:disparity_marginal_raw}.}",
        r"\label{tab:disparity_marginal}",
        r"\end{table*}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table2] wrote {out_path} ({len(MODEL_ORDER)} models)")


# ---------------------------------------------------------------------------
# Table S.2b — marginal race/gender bias, raw values by study, appendix
# ---------------------------------------------------------------------------

def build_table2_raw_by_study(cfg, table_dir: Path, out_path: Path) -> None:
    """Race x study and gender x study raw values (not z-scored), appendix
    companion to Table 2: the same marginal groups, broken down by source
    study instead of blended across studies, using the fixed (name, study)
    join from src/hiring_r4.py.
    """
    raw_data_dir = Path(cfg.paths.raw_data)
    # longtable, not table: ~118 data rows (race+gender, each x study, x nine
    # models) do not fit a single-page float.
    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/hiring_audit_<label>.csv joined via src.hiring_r4.load_and_join,",
        "% grouped by (race, study) and (gender, study) separately -- raw values, companion to Table 2.",
        "% longtable, not table: ~118 rows do not fit a single-page float.",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{@{}lllrrrr@{}}",
        r"\caption{\textbf{Warmth/competence bias by marginal demographic group, broken "
        r"down by source study, raw values.} Companion to \autoref{tab:disparity_marginal}: "
        r"the same race and gender marginal groups, not blended across studies. Model "
        r"warmth/competence are raw (unnormalized) projections and are not comparable in "
        r"magnitude across models or against the 0-100 human warmth/competence scale; see "
        r"\autoref{tab:disparity_marginal} for the standardized ($z$-score) comparison.} "
        r"\label{tab:disparity_marginal_raw} \\",
        r"\toprule",
        r"Model & Axis & Group $\times$ Study & $n$ & Model warmth/competence & "
        r"Human warmth/competence & Human callback \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{l}{\textit{\autoref{tab:disparity_marginal_raw} continued}} \\",
        r"\toprule",
        r"Model & Axis & Group $\times$ Study & $n$ & Model warmth/competence & "
        r"Human warmth/competence & Human callback \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for label in MODEL_ORDER:
        audit_csv = table_dir / f"hiring_audit_{label}.csv"
        matched = load_and_join(audit_csv, raw_data_dir)
        first_row_of_model = True
        for axis_col, axis_label, values in [
            ("race", "Race", ("Black", "White")),
            ("gender", "Gender", ("Female", "Male")),
        ]:
            grouped = group_statistics(matched, label, [axis_col, "study"])
            for value in values:
                cell = grouped[grouped[axis_col] == value]
                present_studies = [s for s in STUDY_ORDER if s in set(cell["study"])]
                first_row_of_value = True
                for study in present_studies:
                    row = cell[cell["study"] == study].iloc[0]
                    model_cell = DISPLAY_NAME[label] if first_row_of_model else ""
                    axis_cell = axis_label if first_row_of_model else ""
                    lines.append(
                        f"{model_cell} & {axis_cell if first_row_of_value else ''} & "
                        f"{value} / {study.capitalize()} & {int(row['n_names'])} & "
                        f"{row['model_warmth_mean']:.2f} / {row['model_competence_mean']:.2f} & "
                        f"{row['human_warm_mean']:.2f} / {row['human_competent_mean']:.2f} & "
                        f"{row['human_callback']:.3f} \\\\"
                    )
                    first_row_of_model = False
                    first_row_of_value = False
        lines.append(r"\addlinespace")
    lines += [
        r"\end{longtable}",
        r"\endgroup",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table2_raw] wrote {out_path} ({len(MODEL_ORDER)} models)")


# ---------------------------------------------------------------------------
# Table 2c — crossed race x gender bias, main text (z-scored, no study)
# ---------------------------------------------------------------------------

def build_table_race_gender(cfg, table_dir: Path, out_path: Path) -> None:
    """Race x gender crossed groups (no study breakdown), z-scored model vs.
    human warmth/competence, same convention as Table 2. Companion main-text
    table to Table 2 (marginal); the study-broken version is
    \\autoref{tab:disparity_crossed} in the appendix.
    """
    raw_data_dir = Path(cfg.paths.raw_data)
    group_order = [("Black", "Female"), ("Black", "Male"), ("White", "Female"), ("White", "Male")]

    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/hiring_audit_<label>.csv joined via src.hiring_r4.load_and_join,",
        "% grouped by (race, gender), collapsed across study; z-scored, companion to Table 2.",
        "% table* + resizebox: same twocolumn-body overflow reason as Table 2 above.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Model & Race $\times$ Gender & $n$ & Model warmth/competence ($z$) & "
        r"Human warmth/competence ($z$) & Human callback & Model margin \\",
        r"\midrule",
    ]
    for label in MODEL_ORDER:
        audit_csv = table_dir / f"hiring_audit_{label}.csv"
        matched = load_and_join(audit_csv, raw_data_dir)
        grouped = group_statistics(matched, label, ["race", "gender"])
        dist_stats = full_distribution_stats(audit_csv, raw_data_dir)
        grouped = add_zscores(
            grouped,
            dist_stats,
            {
                "model_warmth": "model_warmth_mean",
                "model_competence": "model_competence_mean",
                "human_warm": "human_warm_mean",
                "human_competent": "human_competent_mean",
            },
        )
        first_row_of_model = True
        for race, gender in group_order:
            row = grouped[(grouped["race"] == race) & (grouped["gender"] == gender)].iloc[0]
            model_cell = DISPLAY_NAME[label] if first_row_of_model else ""
            lines.append(
                f"{model_cell} & {race}-{gender} & {int(row['n_names'])} & "
                f"{row['model_warmth_z']:+.2f} / {row['model_competence_z']:+.2f} & "
                f"{row['human_warm_z']:+.2f} / {row['human_competent_z']:+.2f} & "
                f"{row['human_callback']:.3f} & {row['model_margin_mean']:.3f} \\\\"
            )
            first_row_of_model = False
        lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Warmth/competence bias by crossed race $\times$ gender group, "
        r"model versus human, standardized.} The same four groups as "
        r"\autoref{tab:disparity_crossed}, collapsed across source study and standardized "
        r"to $z$-scores as in \autoref{tab:disparity_marginal}. Raw, study-broken values are "
        r"in \autoref{tab:disparity_crossed}.}",
        r"\label{tab:disparity_race_gender}",
        r"\end{table*}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table_race_gender] wrote {out_path} ({len(MODEL_ORDER)} models)")


# ---------------------------------------------------------------------------
# Table 3 — crossed race x gender disparity, appendix (re-derived)
# ---------------------------------------------------------------------------

def build_table3(cfg, table_dir: Path, out_path: Path) -> None:
    raw_data_dir = Path(cfg.paths.raw_data)

    all_rows = []
    for label in MODEL_ORDER:
        audit_csv = table_dir / f"hiring_audit_{label}.csv"
        matched = load_and_join(audit_csv, raw_data_dir)
        grouped = group_statistics(matched, label, ["race", "gender", "study"])
        grouped["label"] = label
        all_rows.append(grouped)

    combined = pd.concat(all_rows, ignore_index=True)

    # Regression gate: recomputed (race, gender, study) margins must match
    # hiring_r4.py's own saved hiring_group_r4_<label>.csv for all nine
    # models (both build the same join independently; they must agree).
    mismatches = []
    for label in MODEL_ORDER:
        existing = pd.read_csv(table_dir / f"hiring_group_r4_{label}.csv")
        new = combined[combined["label"] == label]
        for _, erow in existing.iterrows():
            nrow = new[
                (new["race"] == erow["race"])
                & (new["gender"] == erow["gender"])
                & (new["study"] == erow["study"])
            ]
            if nrow.empty:
                mismatches.append(
                    f"{label}: missing group {erow['race']}/{erow['gender']}/{erow['study']}"
                )
                continue
            got = float(nrow.iloc[0]["model_margin_mean"])
            want = float(erow["model_margin_mean"])
            if abs(got - want) > 1e-6:
                mismatches.append(
                    f"{label} {erow['race']}/{erow['gender']}/{erow['study']}: "
                    f"recomputed margin {got:.6f} != existing {want:.6f}"
                )
    if mismatches:
        raise RuntimeError(
            "Table 3 regression gate failed against hiring_group_r4_<label>.csv:\n"
            + "\n".join(mismatches)
        )
    print(f"[table3] regression gate passed for {len(MODEL_ORDER)} labels")

    # longtable (not table): with human warmth/competence added, this table
    # runs to roughly 100 rows across nine models and no longer fits a
    # single page even in the appendix's one-column layout; longtable lets
    # it break across pages with a repeated header, which a plain `table`
    # float cannot do.
    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/hiring_audit_<label>.csv joined with "
        "data/raw/.../published_data/df_all.csv (src.hiring_r4.load_and_join),",
        "% one row per (name, study) pair -- see src/hiring_r4.py module docstring.",
        "% Regression-gated against results/tables/hiring_group_r4_<label>.csv "
        "for all nine models.",
        "% longtable, not table: this now spans more than one page (human",
        "% warmth/competence columns added on top of the model columns).",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{longtable}{@{}lllrrrrr@{}}",
        r"\caption{\textbf{Name-level warmth/competence and callback margin by crossed "
        r"race $\times$ gender group, broken down by source study, raw values.} "
        r"Applicant names are joined to \citet{gallo2024warmth}'s published "
        r"race/gender/callback labels by lowercase first name and matching study "
        r"(\texttt{src/hiring\_r4.py}); a name rated under more than one source study "
        r"(e.g. Bertrand and Kline) contributes one row per study rather than being "
        r"collapsed to a single value, since callback rates differ meaningfully by study. "
        r"Model warmth/competence are raw projections and are not comparable in magnitude "
        r"across models or against the 0-100 human warmth/competence scale; see "
        r"\autoref{tab:disparity_race_gender} for the standardized ($z$-score), "
        r"study-collapsed comparison.} "
        r"\label{tab:disparity_crossed} \\",
        r"\toprule",
        r"Model & Race $\times$ Gender & Study & $n$ & Model warmth/competence & "
        r"Human warmth/competence & Human callback & Model margin \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{l}{\textit{\autoref{tab:disparity_crossed} continued}} \\",
        r"\toprule",
        r"Model & Race $\times$ Gender & Study & $n$ & Model warmth/competence & "
        r"Human warmth/competence & Human callback & Model margin \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    group_order = [
        ("Black", "Female"),
        ("Black", "Male"),
        ("White", "Female"),
        ("White", "Male"),
    ]
    for label in MODEL_ORDER:
        sub = combined[combined["label"] == label]
        first_row_of_model = True
        for race, gender in group_order:
            cell = sub[(sub["race"] == race) & (sub["gender"] == gender)]
            present_studies = [s for s in STUDY_ORDER if s in set(cell["study"])]
            first_row_of_group = True
            for study in present_studies:
                row = cell[cell["study"] == study].iloc[0]
                model_cell = DISPLAY_NAME[label] if first_row_of_model else ""
                group_cell = f"{race}-{gender}" if first_row_of_group else ""
                lines.append(
                    f"{model_cell} & {group_cell} & {study.capitalize()} & "
                    f"{int(row['n_names'])} & "
                    f"{row['model_warmth_mean']:.2f} / {row['model_competence_mean']:.2f} & "
                    f"{row['human_warm_mean']:.2f} / {row['human_competent_mean']:.2f} & "
                    f"{row['human_callback']:.3f} & {row['model_margin_mean']:.3f} \\\\"
                )
                first_row_of_model = False
                first_row_of_group = False
        lines.append(r"\addlinespace")
    lines += [
        r"\end{longtable}",
        r"\endgroup",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table3] wrote {out_path} ({len(MODEL_ORDER)} models)")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    table_dir = Path(cfg.paths.results) / "tables"
    log_dir = Path(cfg.paths.logs)

    build_table1(cfg, log_dir, table_dir / "probe_human_correlation_9model.tex")
    build_table2(table_dir, table_dir / "hiring_disparity_marginal_9model.tex")
    build_table2_raw_by_study(
        cfg, table_dir, table_dir / "hiring_disparity_marginal_raw_9model.tex"
    )
    build_table_race_gender(
        cfg, table_dir, table_dir / "hiring_disparity_race_gender_9model.tex"
    )
    build_table3(cfg, table_dir, table_dir / "hiring_disparity_crossed_9model.tex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main()
