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

# Canonical per-model concept-steering summary file, one row set per model.
# Mirrors the file list independently verified in
# paper/2026-07-20_0919_nine_model_normalized_steerability.md (the same
# mapping backs concept_steerability_normalized_9model.csv). The five
# "_calibrated*" files carry a richer schema (direction_type, 99 SD-matched
# random directions) than the four legacy files (single `random` direction);
# see build_table_concept_signal_vs_control, which discloses this explicitly.
STEERING_DENSE_CSV = {
    "gemma3_12b": "steering_dense_gemma3_12b.csv",
    "gemma3_27b": "steering_dense_gemma3_27b.csv",
    "llama31_8b": "steering_dense_llama31_8b.csv",
    "qwen3_14b": "steering_dense_qwen3_14b.csv",
    "gemma4_12b": "steering_dense_gemma4_12b_calibrated_ccu_h100.csv",
    "gemma4_26b_a4b": "steering_dense_gemma4_26b_a4b_calibrated_scckn_rtx6000.csv",
    "gemma4_31b": "steering_dense_gemma4_31b_calibrated_scckn_rtx6000.csv",
    "qwen36_27b": "steering_dense_qwen36_27b_calibrated_topicfix_scckn_rtx6000.csv",
    "qwen36_35b_a3b": "steering_dense_qwen36_35b_a3b_calibrated_scckn_rtx6000.csv",
}
# Labels have "_calibrated"/etc. suffixes stripped; these five use the richer schema.
CALIBRATED_LABELS = {"gemma4_12b", "gemma4_26b_a4b", "gemma4_31b", "qwen36_27b", "qwen36_35b_a3b"}

GEMMA_SCOPE_MODELS = ("gemma3_12b", "gemma3_27b")

# Canonical per-model broad-grid (alpha in {-0.5,-0.25,0,0.25,0.5}) hiring
# steering CSV, one per model, all on the same strength scale (unlike the
# mixed local/broad regime already disclosed in tab:hiring_transition_census).
HIRING_STEERING_BROAD_CSV = {
    "gemma3_12b": "hiring_steering_raw_gemma3_12b.csv",
    "gemma3_27b": "hiring_steering_raw_gemma3_27b.csv",
    "llama31_8b": "hiring_steering_raw_llama31_8b.csv",
    "qwen3_14b": "hiring_steering_raw_qwen3_14b.csv",
    "gemma4_12b": "hiring_steering_raw_gemma4_12b_broad.csv",
    "gemma4_26b_a4b": "hiring_steering_raw_gemma4_26b_a4b_broad.csv",
    "gemma4_31b": "hiring_steering_raw_gemma4_31b_broad.csv",
    "qwen36_27b": "hiring_steering_raw_qwen36_27b_broad.csv",
    "qwen36_35b_a3b": "hiring_steering_raw_qwen36_35b_a3b_broad.csv",
}

DIRECTION_DISPLAY = {
    "raw_dense": "Dense target",
    "sae_reconstructed": "SAE decoded",
    "axis_specific": "Axis-specific",
    "shared": "Shared",
    "other_axis": "Opposing axis",
    "random": "Random",
}
DIRECTION_ORDER = ("raw_dense", "sae_reconstructed", "axis_specific", "shared", "other_axis", "random")

ABLATION_DIRECTION_DISPLAY = {
    "target_axis": "Target axis",
    "shared": "Shared",
    "other_axis": "Opposing axis",
    "random_features": "Random",
}
ABLATION_DIRECTION_ORDER = ("target_axis", "shared", "other_axis", "random_features")


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


# ---------------------------------------------------------------------------
# Table: probe validation, all nine models (main text)
# ---------------------------------------------------------------------------

def _validate_probes_path(log_dir: Path, label: str) -> Path:
    # The default model (gemma3_12b) was the first one run and its log kept
    # the pre-nine-model filename; every later model follows the
    # validate_probes_<label>.json convention.
    if label == "gemma3_12b":
        return log_dir / "validate_probes_default.json"
    return log_dir / f"validate_probes_{label}.json"


def build_table_probe_validation(log_dir: Path, out_path: Path) -> None:
    rows = []
    for label in MODEL_ORDER:
        vp = json.loads(_validate_probes_path(log_dir, label).read_text(encoding="utf-8"))
        sh = json.loads(
            (log_dir / f"split_half_stability_{label}.json").read_text(encoding="utf-8")
        )
        rb = json.loads(
            (log_dir / f"random_baseline_{label}.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "model": DISPLAY_NAME[label],
                "cohens_d": f"{vp['warmth']['cohens_d']:.2f} / {vp['competence']['cohens_d']:.2f}",
                "null_z": f"{rb['warmth']['z_score']:.1f} / {rb['competence']['z_score']:.1f}",
                "split_half": f"{sh['split_half_cosine_warmth']:.2f} / "
                f"{sh['split_half_cosine_competence']:.2f}",
                "cross_axis": f"{vp['cross_warmth_on_competence_calibrated_cv']:.2f} / "
                f"{vp['cross_competence_on_warmth_calibrated_cv']:.2f}",
            }
        )

    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/logs/validate_probes_<label>.json (Cohen's d, "
        "calibrated cross-axis accuracy),",
        "% results/logs/split_half_stability_<label>.json, "
        "results/logs/random_baseline_<label>.json (random-null z-score,",
        "% written by paper/figures/generate_figures.py::fig2_random_baseline).",
        "% 5-fold CV and topic-holdout accuracy are 1.00/1.00 for every model on",
        "% both axes (no variation to tabulate) and are reported in prose instead.",
        "% table* + resizebox: same twocolumn-body overflow reason as Table 1.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Model & Cohen's $d$ (W / C) & Random-null $z$ (W / C) & "
        r"Split-half cosine (W / C) & Cross-axis accuracy (W$\to$C / C$\to$W) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['cohens_d']} & {row['null_z']} & "
            f"{row['split_half']} & {row['cross_axis']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Probe validation, all nine models.} Warmth (W) and "
        r"competence (C) values in each cell. Cohen's $d$ is the effect size "
        r"separating high- and low-condition stories on the raw dense direction. "
        r"Random-null $z$ standardizes that same $d$ against 1,000 random "
        r"directions' own high/low separation on the same stories; in every "
        r"model and axis, 0 of 1,000 random directions matched or exceeded the "
        r"real direction. Split-half cosine is the cosine between two "
        r"directions independently built from non-overlapping 25-story halves "
        r"of each condition. Cross-axis accuracy (calibrated) is how well a "
        r"story's projection onto one axis's direction predicts its label on "
        r"the other axis. 5-fold cross-validated and topic-holdout accuracy "
        r"are omitted here since both are 1.00 for every model on both axes.}",
        r"\label{tab:probe_validation}",
        r"\end{table*}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table_probe_validation] wrote {out_path} ({len(rows)} models)")


# ---------------------------------------------------------------------------
# Table: cross-model Spearman agreement, all 36 pairs (appendix)
# ---------------------------------------------------------------------------

def build_table_cross_model_agreement(table_dir: Path, out_path: Path) -> None:
    agreement_csv = table_dir / "cross_model_agreement_9model.csv"
    df = pd.read_csv(agreement_csv)
    order_index = {DISPLAY_NAME[label]: i for i, label in enumerate(MODEL_ORDER)}
    df["_a_idx"] = df["model_a"].map(order_index)
    df["_b_idx"] = df["model_b"].map(order_index)
    df = df.sort_values(["axis", "_a_idx", "_b_idx"]).reset_index(drop=True)

    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/cross_model_agreement_9model.csv "
        "(src/validate_cross_model_agreement.py; all C(9,2)=36 pairs x 2 axes).",
        "% Only overall and within-condition rho are shown here; the four",
        "% per-condition columns (high/low warmth/competence rho) are in the",
        "% source CSV. longtable: 72 rows do not fit one page.",
        r"\begingroup",
        r"\small",
        r"\begin{longtable}{@{}llrr@{}}",
        r"\caption{\textbf{Cross-model Spearman story-ranking agreement, all "
        r"36 model pairs.} For each pair and axis, every one of the 200 "
        r"concept stories is projected onto both models' own warmth or "
        r"competence direction, and the two resulting score lists are "
        r"correlated with Spearman's $\rho$. ``Overall'' includes the "
        r"high/low condition separation itself; ``within-condition'' "
        r"restricts the correlation to stories sharing one condition, so it "
        r"reflects agreement on finer-grained story ordering rather than "
        r"the coarse high-versus-low split. The four per-condition $\rho$ "
        r"values behind the within-condition summary are in "
        r"\texttt{results/tables/cross\_model\_agreement\_9model.csv}.} "
        r"\label{tab:cross_model_agreement} \\",
        r"\toprule",
        r"Model A & Model B & Overall $\rho$ & Within-condition $\rho$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{4}{l}{\textit{\autoref{tab:cross_model_agreement} continued}} \\",
        r"\toprule",
        r"Model A & Model B & Overall $\rho$ & Within-condition $\rho$ \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for axis in ("warmth", "competence"):
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{axis.title()}}}}} \\")
        sub = df[df["axis"] == axis]
        for _, row in sub.iterrows():
            lines.append(
                f"{row['model_a']} & {row['model_b']} & "
                f"{row['overall_rho']:.3f} & {row['within_condition_rho']:.3f} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [
        r"\end{longtable}",
        r"\endgroup",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table_cross_model_agreement] wrote {out_path} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Table: neutral-corpus PCA denoising, all nine models (appendix)
# ---------------------------------------------------------------------------

def build_table_pca_denoising(cfg, out_path: Path) -> None:
    processed_dir = Path(cfg.paths.processed)
    rows = []
    for label in MODEL_ORDER:
        subdir = "concept_vectors" if label == "gemma3_12b" else f"concept_vectors_{label}"
        summary_path = processed_dir / subdir / "denoise_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": DISPLAY_NAME[label],
                "k": summary["k"],
                "variance_kept": summary["variance_kept"],
                "cos_before": summary["cosine_before"],
                "cos_after": summary["cosine_after"],
            }
        )

    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: data/processed/concept_vectors<_label>/denoise_summary.json"
        " for all nine models",
        "% (src/denoise_vectors.py). Verified against results/logs/validate_probes_*.json's",
        "% independently-computed axis_cosine and re-derived from scratch for",
        "% Gemma-3-12B; see paper/2026-08-04_1610_pca_denoising_verification_and_gap_closure.md.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Model & $k$ (PCs removed) & Variance kept & cos(W,C) before & cos(W,C) after \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['k']} & {row['variance_kept']:.1%}".replace("%", r"\%")
            + f" & {row['cos_before']:.3f} & {row['cos_after']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Neutral-corpus PCA denoising, all nine models.} "
        r"$k$ is the number of leading principal components of 1,500 neutral "
        r"Wikipedia activations removed from both direction vectors, the "
        r"smallest $k$ crossing the 50\% neutral-variance threshold. "
        r"cos(W,C) before/after is the cosine between the warmth and "
        r"competence directions, measured before and after removing those "
        r"components. Denoising reduces cos(W,C) in every model but never "
        r"eliminates it.}",
        r"\label{tab:pca_denoising}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table_pca_denoising] wrote {out_path} ({len(rows)} models)")


# ---------------------------------------------------------------------------
# Table: concept-steering saturation at wide strengths (main text)
# ---------------------------------------------------------------------------

def build_table_concept_saturation(table_dir: Path, out_path: Path) -> None:
    strengths = ["-0.5", "-0.25", "0.0", "0.25", "0.5"]
    rows = []
    for label in GEMMA_SCOPE_MODELS:
        df = pd.read_csv(table_dir / f"gemma_scope_causality_{label}.csv")
        for axis in ("warmth", "competence"):
            sub = df[
                (df["mode"] == "steering")
                & (df["direction"] == "raw_dense")
                & (df["axis"] == axis)
            ].copy()
            sub["strength"] = sub["strength"].astype(str)
            sub = sub.set_index("strength")
            cells = [f"{sub.loc[s, 'effect']:+.2f}" for s in strengths]
            rows.append({"model": DISPLAY_NAME[label], "axis": axis.capitalize(), "cells": cells})

    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/gemma_scope_causality_<label>.csv "
        "(mode==steering, direction==raw_dense), Gemma-3-12B/27B only.",
        "% This is the wide-strength sweep the Methods 'Steering the Concept",
        "% Vectors' saturation claim refers to; the narrow +/-0.10 grid used",
        "% for all nine models is in tab:probe_validation-adjacent Results text.",
        "% table* + resizebox: 7 columns overflow the twocolumn body's single",
        "% column width even at \\small (confirmed overfull otherwise).",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{0.6\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Model & Axis & $\alpha=-0.50$ & $-0.25$ & $0$ & $+0.25$ & $+0.50$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(f"{row['model']} & {row['axis']} & " + " & ".join(row["cells"]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Concept-level steering at wide strengths.} Change "
        r"in the held-out Yes-versus-No logit margin under dense-target "
        r"steering, Gemma-3-12B and Gemma-3-27B, the two models where the "
        r"wide strength grid was tested. The step from $+0.25$ to $+0.50$ is "
        r"smaller than the step before it for three of the four rows, "
        r"consistent with the saturating breakdown Methods describes; "
        r"Gemma-3-27B warmth is the exception, still accelerating at "
        r"$+0.50$.}",
        r"\label{tab:concept_saturation}",
        r"\end{table*}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table_concept_saturation] wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Table: six-direction specificity comparison (main text)
# ---------------------------------------------------------------------------

def build_table_concept_direction_specificity(table_dir: Path, out_path: Path) -> None:
    # Local regime (alpha=+0.10), matching the primary strength scale used
    # everywhere else in this study, not the wide/saturating endpoint used
    # by build_table_concept_saturation. Uses the _local companion file.
    rows = []
    for label in GEMMA_SCOPE_MODELS:
        df = pd.read_csv(table_dir / f"gemma_scope_causality_{label}_local.csv")
        for axis in ("warmth", "competence"):
            sub = df[
                (df["mode"] == "steering")
                & (df["axis"] == axis)
                & (df["strength"].astype(str) == "0.1")
            ].set_index("direction")
            cells = [f"{sub.loc[d, 'effect']:+.2f}" for d in DIRECTION_ORDER]
            dense_is_max = sub.loc["raw_dense", "effect"] == sub["effect"].max()
            rows.append(
                {
                    "model": DISPLAY_NAME[label],
                    "axis": axis.capitalize(),
                    "cells": cells,
                    "dense_is_max": dense_is_max,
                }
            )

    header_cols = " & ".join(DIRECTION_DISPLAY[d] for d in DIRECTION_ORDER)
    n_dense_max = sum(1 for r in rows if r["dense_is_max"])
    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/gemma_scope_causality_<label>_local.csv "
        "(mode==steering, alpha=+0.10), Gemma-3-12B/27B only.",
        "% table* + resizebox: 6 direction columns overflow a single column.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}ll" + "r" * len(DIRECTION_ORDER) + r"@{}}",
        r"\toprule",
        rf"Model & Axis & {header_cols} \\",
        r"\midrule",
    ]
    for row in rows:
        cells = list(row["cells"])
        if row["dense_is_max"]:
            cells[0] = r"\textbf{" + cells[0] + "}"
        lines.append(f"{row['model']} & {row['axis']} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Direction specificity at the local endpoint "
        r"($\alpha=+0.10$).} Change in the held-out Yes-versus-No logit "
        r"margin for six directions, Gemma-3-12B and Gemma-3-27B. Dense "
        r"target is the same mean-difference direction used everywhere else "
        r"in this study; SAE decoded, axis-specific, and shared are built "
        r"from a Gemma Scope~2 sparse decomposition; opposing axis steers "
        r"along the other concept's direction; random is a single "
        r"orthogonalized control. Bold marks the row's largest effect. Dense "
        rf"target is largest in only {n_dense_max} of the four model-axis "
        r"rows; the other three are matched or exceeded by at least one "
        r"unrelated direction, most often the SAE-decoded or axis-specific "
        r"component, which does not support a clean specificity reading and "
        r"is discussed in the text alongside the shared warmth-competence "
        r"overlap already noted in PCA denoising.}",
        r"\label{tab:concept_direction_specificity}",
        r"\end{table*}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table_concept_direction_specificity] wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Table: signal vs. random control, all nine models (main text)
# ---------------------------------------------------------------------------

def build_table_concept_signal_vs_control(table_dir: Path, out_path: Path) -> None:
    rows = []
    for label in MODEL_ORDER:
        df = pd.read_csv(table_dir / STEERING_DENSE_CSV[label])
        calibrated = label in CALIBRATED_LABELS
        for axis in ("warmth", "competence"):
            if calibrated:
                # direction_type is not present in every "_calibrated*" file
                # (the Qwen3.6 exports omit it), so identify target vs.
                # random rows from `direction` itself, which is consistent
                # across all five calibrated files: the target row's
                # direction equals the axis name, random rows are
                # random_000..random_098.
                sub = df[(df["axis"] == axis) & (df["intervention"] == "additive")]
                target = sub[(sub["direction"] == axis) & (sub["strength"].astype(str) == "0.1")]
                target_effect = float(target["effect"].iloc[0])
                rand = sub[
                    sub["direction"].str.startswith("random_") & (sub["strength"].astype(str) == "0.1")
                ]
                rand_mean = rand["effect"].mean()
                rand_sd = rand["effect"].std(ddof=1)
                rand_cell = f"{rand_mean:+.2f} $\\pm$ {1.96 * rand_sd:.2f}"
                basis = "99 SD-matched dirs."
            else:
                sub = df[(df["mode"] == "steering") & (df["axis"] == axis) & (df["strength"].astype(str) == "0.1")]
                target_effect = float(sub[sub["direction"] == "raw_dense"]["effect"].iloc[0])
                rand_row = sub[sub["direction"] == "random"].iloc[0]
                rand_cell = f"{rand_row['effect']:+.2f} [{rand_row['ci_low']:+.2f}, {rand_row['ci_high']:+.2f}]"
                basis = "1 direction"
            rows.append(
                {
                    "model": DISPLAY_NAME[label],
                    "axis": axis.capitalize(),
                    "target": f"{target_effect:+.2f}",
                    "random": rand_cell,
                    "basis": basis,
                }
            )

    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/steering_dense_<label>*.csv, one file per",
        "% model per STEERING_DENSE_CSV (same canonical mapping verified in",
        "% paper/2026-07-20_0919_nine_model_normalized_steerability.md).",
        "% Random-control basis is genuinely heterogeneous across models: five",
        "% checkpoints have 99 SD-matched random directions (mean +/- 95% CI",
        "% shown), four have a single random direction (bootstrap CI shown).",
        "% This is disclosed in the Control basis column and in the caption,",
        "% not silently normalized away. See STEP_LOG for the deferred-GPU-work",
        "% decision behind this heterogeneity.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrlc@{}}",
        r"\toprule",
        r"Model & Axis & Dense target ($\alpha=+0.10$) & Random control & Control basis \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['axis']} & {row['target']} & {row['random']} & {row['basis']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Dense-target steering versus a random-direction "
        r"control, all nine models.} The random-control basis is not uniform: "
        r"Gemma-4-12B, Gemma-4-26B-A4B, Gemma-4-31B, Qwen3.6-27B, and "
        r"Qwen3.6-35B-A3B use 99 SD-matched random directions per model, "
        r"reported as mean $\pm$ 95\% CI; Gemma-3-12B, Gemma-3-27B, "
        r"Llama-3.1-8B, and Qwen3-14B use a single random direction, reported "
        r"with its own bootstrap CI. Extending the calibrated 99-direction "
        r"control to the remaining four models requires new GPU runs and is "
        r"left for future work rather than closed here.}",
        r"\label{tab:concept_signal_vs_control}",
        r"\end{table*}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table_concept_signal_vs_control] wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Table: Gemma Scope SAE reconstruction quality (appendix)
# ---------------------------------------------------------------------------

def build_table_gemma_scope_sae_quality(table_dir: Path, out_path: Path) -> None:
    width_order = ["16k", "65k", "262k"]
    rows = []
    for label in GEMMA_SCOPE_MODELS:
        df = pd.read_csv(table_dir / f"gemma_scope_metrics_{label}.csv").set_index("width")
        for width in width_order:
            r = df.loc[width]
            rows.append(
                {
                    "model": DISPLAY_NAME[label],
                    "width": width,
                    "recon": f"{r['reconstruction_cosine_mean']:.3f}",
                    "sparsity": f"{r['active_features_mean']:.1f}",
                    "topic_cv": f"{r['warmth_topic_cv']:.2f} / {r['competence_topic_cv']:.2f}",
                    "decoded_cos": f"{r['decoded_warmth_alignment']:.2f} / {r['decoded_competence_alignment']:.2f}",
                }
            )

    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/gemma_scope_metrics_<label>.csv, Gemma-3-12B/27B only.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Model & Width & Recon. cosine & Active feats. & Topic-CV (W/C) & Decoded cosine (W/C) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['model']} & {row['width']} & {row['recon']} & {row['sparsity']} & "
            f"{row['topic_cv']} & {row['decoded_cos']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Gemma Scope~2 SAE reconstruction quality.} Recon. "
        r"cosine is the cosine between the original activation and its SAE "
        r"reconstruction; active features is the mean number of nonzero SAE "
        r"features per story; topic-CV is 5-fold topic-holdout accuracy for "
        r"warmth/competence classification inside the sparse feature space; "
        r"decoded cosine is the alignment between the decoded SAE direction "
        r"and the raw dense direction.}",
        r"\label{tab:gemma_scope_sae_quality}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table_gemma_scope_sae_quality] wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Table: Gemma Scope feature ablation (appendix)
# ---------------------------------------------------------------------------

def build_table_gemma_scope_ablation(table_dir: Path, out_path: Path) -> None:
    rows = []
    for label in GEMMA_SCOPE_MODELS:
        df = pd.read_csv(table_dir / f"gemma_scope_causality_{label}.csv")
        for axis in ("warmth", "competence"):
            sub = df[(df["mode"] == "ablation") & (df["axis"] == axis)].set_index("direction")
            cells = [f"{sub.loc[d, 'effect']:+.3f}" for d in ABLATION_DIRECTION_ORDER]
            rows.append({"model": DISPLAY_NAME[label], "axis": axis.capitalize(), "cells": cells})

    header_cols = " & ".join(ABLATION_DIRECTION_DISPLAY[d] for d in ABLATION_DIRECTION_ORDER)
    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/gemma_scope_causality_<label>.csv "
        "(mode==ablation), Gemma-3-12B/27B only.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}ll" + "r" * len(ABLATION_DIRECTION_ORDER) + r"@{}}",
        r"\toprule",
        rf"Model & Axis & {header_cols} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(f"{row['model']} & {row['axis']} & " + " & ".join(row["cells"]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Feature ablation, Gemma-3-12B/27B.} Change in the "
        r"high-versus-low margin gap when the named group of SAE features is "
        r"zeroed out rather than added to; a negative value means ablation "
        r"shrinks the gap. Shared-feature ablation shrinks the gap more than "
        r"target-axis ablation in three of the four rows, echoing the "
        r"substantial shared component already documented in PCA denoising "
        r"rather than a clean axis-specific routing.}",
        r"\label{tab:gemma_scope_ablation}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table_gemma_scope_ablation] wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Table: cross-scale (12B<->27B) feature matching (appendix)
# ---------------------------------------------------------------------------

def build_table_gemma_scope_feature_matching(table_dir: Path, out_path: Path) -> None:
    null_df = pd.read_csv(table_dir / "gemma_scope_feature_match_null_12b_27b.csv").set_index("vector")
    rows = []
    for axis in ("warmth", "competence"):
        r = null_df.loc[axis]
        rows.append(
            {
                "axis": axis.capitalize(),
                "n_matches": int(r["n_matches"]),
                "observed_mean": f"{r['observed_mean']:.3f}",
                "observed_median": f"{r['observed_median']:.3f}",
                "null_mean": f"{r['null_mean']:.3f}",
                "null_ci": f"[{r['null_mean_ci_low']:.3f}, {r['null_mean_ci_high']:.3f}]",
                "p": f"{r['permutation_p_mean']:.4f}",
            }
        )

    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/gemma_scope_feature_match_null_12b_27b.csv",
        "% (summary only; the full ranked feature-pair list is in",
        "% gemma_scope_feature_matches_12b_27b.csv).",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrrrl@{}}",
        r"\toprule",
        r"Axis & $n$ matches & Obs. mean & Obs. median & Null mean [95\% CI] & $p$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['axis']} & {row['n_matches']} & {row['observed_mean']} & "
            f"{row['observed_median']} & {row['null_mean']} {row['null_ci']} & {row['p']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Cross-scale SAE feature matching, Gemma-3-12B "
        r"$\leftrightarrow$ Gemma-3-27B.} For each matched feature pair, "
        r"story-profile correlation is compared against a 500-permutation "
        r"null. Observed mean/median well above the null mean, with a small "
        r"permutation $p$-value, indicates the two scales share features with "
        r"correlated story-level responses beyond chance.}",
        r"\label{tab:gemma_scope_feature_matching}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table_gemma_scope_feature_matching] wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Table: hiring callback steering, broad grid, all nine models (main text)
# ---------------------------------------------------------------------------

def _ols_slope_r2(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Plain-Python least-squares fit; no numpy dependency for this module."""
    n = len(xs)
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    ss_xy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    ss_xx = sum((x - x_bar) ** 2 for x in xs)
    slope = ss_xy / ss_xx
    intercept = y_bar - slope * x_bar
    ss_tot = sum((y - y_bar) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


def build_table_hiring_steering_slopes(table_dir: Path, out_path: Path) -> None:
    rows = []
    for label in MODEL_ORDER:
        df = pd.read_csv(table_dir / HIRING_STEERING_BROAD_CSV[label])
        for axis in ("warmth", "competence"):
            sub = df[df["axis"] == axis]
            by_strength = sub.groupby("strength")["delta"].mean().sort_index()
            xs = list(by_strength.index.astype(float))
            ys = list(by_strength.values)
            slope, _, r2 = _ols_slope_r2(xs, ys)
            endpoint = float(by_strength.loc[0.5])
            rows.append(
                {
                    "model": DISPLAY_NAME[label],
                    "axis": axis.capitalize(),
                    "slope": f"{slope:+.2f}",
                    "r2": f"{r2:.2f}",
                    "r2_hi": r2 >= 0.8,
                    "endpoint": f"{endpoint:+.2f}",
                }
            )

    n_hi = sum(1 for r in rows if r["r2_hi"])
    lines = [
        "% Generated by src/build_paper_probe_tables.py.",
        "% Source: results/tables/hiring_steering_raw_<label>*.csv, broad grid",
        "% (alpha in {-0.5,-0.25,0,0.25,0.5}), all nine models, via",
        "% HIRING_STEERING_BROAD_CSV. Slope/R^2 from a plain OLS fit of the",
        "% per-strength mean delta (60 names each); endpoint is the raw mean",
        "% delta at alpha=+0.50, not the fitted value.",
        "% table* + resizebox: same twocolumn-body overflow reason as the",
        "% other 5-column main-text tables this session.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{0.75\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Model & Axis & Slope ($\Delta$ per unit $\alpha$) & $R^2$ & $\Delta$ at $\alpha=+0.50$ \\",
        r"\midrule",
    ]
    for row in rows:
        r2_cell = r"\textbf{" + row["r2"] + "}" if row["r2_hi"] else row["r2"]
        lines.append(
            f"{row['model']} & {row['axis']} & {row['slope']} & {r2_cell} & {row['endpoint']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Hiring callback steering, broad grid, all nine "
        r"models.} Mean change in the callback margin "
        r"$\Delta_{\text{callback}}$ across the 60-name subset, at "
        r"$\alpha\in\{\pm0.25,\pm0.50\}$. Slope and $R^2$ come from an "
        r"ordinary least-squares fit of the five per-strength means; bold "
        rf"marks $R^2\ge0.8$. The fit is close to linear in only {n_hi} of "
        r"the 18 model-axis rows; several others, most visibly Gemma-3-27B "
        r"on both axes, show weak or non-monotonic trends instead.}",
        r"\label{tab:hiring_steering_slopes}",
        r"\end{table*}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table_hiring_steering_slopes] wrote {out_path} ({len(rows)} rows, {n_hi}/18 R2>=0.8)")


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
    build_table_probe_validation(log_dir, table_dir / "probe_validation_9model.tex")
    build_table_cross_model_agreement(
        table_dir, table_dir / "cross_model_agreement_9model.tex"
    )
    build_table_pca_denoising(cfg, table_dir / "pca_denoising_9model.tex")
    build_table_concept_saturation(table_dir, table_dir / "concept_saturation.tex")
    build_table_concept_direction_specificity(
        table_dir, table_dir / "concept_direction_specificity.tex"
    )
    build_table_concept_signal_vs_control(
        table_dir, table_dir / "concept_signal_vs_control_9model.tex"
    )
    build_table_gemma_scope_sae_quality(table_dir, table_dir / "gemma_scope_sae_quality.tex")
    build_table_gemma_scope_ablation(table_dir, table_dir / "gemma_scope_ablation.tex")
    build_table_gemma_scope_feature_matching(
        table_dir, table_dir / "gemma_scope_feature_matching.tex"
    )
    build_table_hiring_steering_slopes(
        table_dir, table_dir / "hiring_steering_slopes_9model.tex"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main()
