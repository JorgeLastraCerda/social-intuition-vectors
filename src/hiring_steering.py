"""Hiring-callback causal steering sweep for any model with concept vectors.

Productionised version of ``notebooks/06_hiring_steering_causality.ipynb``.
Model-agnostic: model name, probe layer, and d_model are read from
``<vectors-subdir>/meta.json``.

Applicant names come from Gallo & Hausladen's rated name set
(``data/raw/…/ratings/names/df_all.csv``).  Steering is applied at the probe
layer over a grid of strengths expressed in units of ``mean_resid_norm``, exactly
mirroring the concept-steering run in ``src/gemma_scope_causality.py``.

Outputs
-------
results/tables/hiring_steering_raw_<label>.csv
    One row per name × axis × strength.  Contains ``margin`` (raw logit
    Yes − No) and ``delta`` (change from baseline).
results/logs/hiring_steering_<label>.json
    Provenance metadata.

Regression gate (Gemma-3-12B)
------------------------------
When run with ``--vectors-subdir concept_vectors --label gemma3_12b``, the
warmth Δmargin at +0.25/+0.50 must agree with the notebook-produced
``hiring_steering_raw_concept_vectors.csv`` (≈ +7.125 / +8.404) to within GPU
float non-determinism noise.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.gemma_scope_causality import (
    make_steering_hook,
    unit,
    yes_no_margin,
)
from src.utils.config import load_config
from src.utils.hooks import residual_hook_name
from src.utils.model_loader import load_hooked_model, model_runtime_metadata
from src.utils.prompting import decision_token_ids, encode_decision_prompt

AXES = ("warmth", "competence")
DEFAULT_STRENGTHS = "-0.5,-0.25,0,0.25,0.5"

HIRING_PROMPT_TEMPLATE = (
    "You are a hiring manager reviewing job applications. "
    "The following is a summary of a candidate's application.\n\n"
    "Position: Administrative Assistant\n"
    "Education: Bachelor's degree in Business Administration\n"
    "Experience: 3 years as office coordinator\n"
    "Skills: scheduling, written communication, Microsoft Office\n"
    "Applicant name: {signal}\n\n"
    "Based solely on this application, would you recommend calling this candidate "
    "back for an interview?\n"
    "Answer with a single word: Yes or No."
)


def hiring_prompt(name: str) -> str:
    return HIRING_PROMPT_TEMPLATE.format(signal=name.title())


def main() -> None:
    args = parse_args()
    strengths = tuple(
        float(value.strip()) for value in args.strengths.split(",") if value.strip()
    )
    if 0.0 not in strengths:
        raise ValueError("--strengths must include 0.")
    cfg = load_config(args.config)

    table_dir = Path(cfg.paths.results) / "tables"
    log_dir = Path(cfg.paths.logs)
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- vectors & metadata ---
    vectors_dir = Path(cfg.paths.processed) / args.vectors_subdir
    meta = json.loads((vectors_dir / "meta.json").read_text(encoding="utf-8"))
    model_name = str(meta["model"])
    layer = int(meta["probe_layer"])
    hook_name = residual_hook_name(layer)

    if args.vector_kind == "denoised":
        denoised = np.load(vectors_dir / "concept_vectors_denoised.npz")
        warmth_vec = denoised["warmth"].astype(np.float32)
        competence_vec = denoised["competence"].astype(np.float32)
        vector_source = vectors_dir / "concept_vectors_denoised.npz"
    else:
        warmth_vec = np.load(vectors_dir / "warmth_vec.npy").astype(np.float32)
        competence_vec = np.load(vectors_dir / "competence_vec.npy").astype(np.float32)
        vector_source = vectors_dir

    # mean_resid_norm from all condition activations (mirrors notebook 06 cell ea4021af)
    all_X = np.concatenate(
        [
            np.load(vectors_dir / f"X_{c}.npy").astype(np.float32)
            for c in ("high_warmth", "low_warmth", "high_competence", "low_competence")
        ],
        axis=0,
    )
    mean_resid_norm = float(np.linalg.norm(all_X, axis=1).mean())
    print(
        f"[meta] model={model_name}  layer={layer}  hook={hook_name}  "
        f"mean_resid_norm={mean_resid_norm:.1f}",
        flush=True,
    )

    # --- rated names ---
    names_csv = (
        Path(cfg.paths.raw_data)
        / "SocialPerceptions-Predict-Callback-main"
        / "0_data"
        / "ratings"
        / "names"
        / "df_all.csv"
    )
    name_ratings = (
        pd.read_csv(names_csv)
        .groupby("name")
        .agg(
            human_warm=("warm", "mean"),
            human_competent=("competent", "mean"),
            study=("study", "first"),
            n_raters=("warm", "size"),
        )
        .reset_index()
    )
    n_total = len(name_ratings)

    n_names = args.n_names if args.n_names > 0 else n_total
    if n_names < n_total:
        sample = name_ratings.sample(
            n=min(n_names, n_total), random_state=cfg.probing.seed
        ).reset_index(drop=True)
    else:
        sample = name_ratings.reset_index(drop=True)
    print(f"[names] {len(sample)} / {n_total} rated names selected", flush=True)

    # --- load model ---
    cfg = replace(cfg, model=replace(cfg.model, name=model_name))
    print(f"[model] loading {model_name} …", flush=True)
    model = load_hooked_model(cfg)
    model.eval()

    # fail-fast single-token check
    rendered, _ = encode_decision_prompt(
        model, hiring_prompt(str(sample.iloc[0]["name"])), args.prompt_format
    )
    yes_id, no_id = decision_token_ids(model, rendered, args.prompt_format)
    print(f"[tokens] Yes={yes_id}  No={no_id}  (continuation check passed)", flush=True)

    axis_vectors = {"warmth": warmth_vec, "competence": competence_vec}

    # --- baseline ---
    baseline_margins: dict[str, float] = {}
    for i, name in enumerate(sample["name"], start=1):
        m = yes_no_margin(
            model, hiring_prompt(name), hook_name, prompt_format=args.prompt_format
        )
        baseline_margins[name] = m
        if i % 20 == 0 or i == len(sample):
            print(f"[baseline] {i}/{len(sample)}", flush=True)

    # --- causal sweep ---
    rows: list[dict] = []

    for axis in AXES:
        vec = axis_vectors[axis]
        for strength in strengths:
            hook = (
                None
                if strength == 0.0
                else make_steering_hook(vec, strength * mean_resid_norm)
            )
            for name in sample["name"]:
                if strength == 0.0:
                    margin = baseline_margins[name]
                else:
                    margin = yes_no_margin(
                        model,
                        hiring_prompt(name),
                        hook_name,
                        hook,
                        prompt_format=args.prompt_format,
                    )
                rows.append(
                    {
                        "axis": axis,
                        "strength": strength,
                        "name": name,
                        "margin": margin,
                        "delta": margin - baseline_margins[name],
                    }
                )
            print(f"[steering] {axis} strength={strength:+.2f} done", flush=True)

    # --- write raw CSV ---
    raw_path = table_dir / f"hiring_steering_raw_{args.label}.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] {raw_path}  ({len(rows)} rows)", flush=True)

    # --- provenance log ---
    log = {
        "label": args.label,
        "model": model_name,
        "vectors_subdir": args.vectors_subdir,
        "probe_layer": layer,
        "hook": hook_name,
        "mean_resid_norm": mean_resid_norm,
        "seed": cfg.probing.seed,
        "n_names_sampled": len(sample),
        "n_names_total": n_total,
        "strengths": list(strengths),
        "vector_kind": args.vector_kind,
        "vector_source": str(vector_source),
        "prompt_format": args.prompt_format,
        "rendered_prompt_example": rendered,
        "runtime": model_runtime_metadata(model),
        "axes": list(AXES),
        "raw_output": str(raw_path),
    }
    log_path = log_dir / f"hiring_steering_{args.label}.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[done] {log_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hiring-callback causal steering sweep. "
            "Productionised version of notebooks/06_hiring_steering_causality.ipynb."
        )
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--vectors-subdir",
        required=True,
        help="Subdirectory under data/processed/ containing concept vectors and meta.json.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Short identifier used in output filenames (e.g. gemma3_12b).",
    )
    parser.add_argument(
        "--strengths",
        default=DEFAULT_STRENGTHS,
        help="Comma-separated strengths in units of mean residual norm; include 0.",
    )
    parser.add_argument(
        "--vector-kind",
        choices=("raw", "denoised"),
        default="raw",
        help="Use raw concept vectors or PCA-denoised vectors.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("raw", "native-chat"),
        default="raw",
        help="Decision-prompt rendering mode (Gemma 4 jobs use native-chat).",
    )
    parser.add_argument(
        "--n-names",
        type=int,
        default=60,
        help=(
            "Number of names to sample from the rated set (default 60, matching the "
            "original notebook run). Pass 0 to use all 282 names."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
