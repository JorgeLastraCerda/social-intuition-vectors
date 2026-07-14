"""Dense (SAE-free) concept steering for any model with concept vectors on disk.

This script replicates the ``raw_dense`` path from
``src/gemma_scope_causality.py`` without any Gemma Scope / SAE dependency.
It is model-agnostic: model name, probe layer, and d_model are all read from
``<vectors-subdir>/meta.json``.

Outputs
-------
results/tables/steering_dense_raw_<label>.csv  — one row per story × direction × strength
results/tables/steering_dense_<label>.csv      — bootstrapped summary (effect ± CI)
results/logs/steering_dense_<label>.json       — provenance metadata

Regression check
----------------
When run with ``--vectors-subdir concept_vectors --label gemma3_12b``,
the ``raw_dense`` rows in the summary CSV must match
``results/tables/gemma_scope_causality_gemma3_12b_local.csv`` exactly
(warmth +0.1 → 3.88125, competence +0.1 → 2.00625, etc.).
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

# -- reuse validated helpers from the Gemma Scope causality script (no SAE) --
from src.gemma_scope_causality import (
    judgement_prompt,
    make_steering_hook,
    rows_for_topics,
    summarize_baseline,
    summarize_steering,
    train_test_topics,
    unit,
    yes_no_margin,
)
from src.gemma_scope_utils import (
    CONDITIONS,
    check_file_size,
    condition_slices,
    load_story_records,
)
from src.utils.config import load_config
from src.utils.hooks import residual_hook_name
from src.utils.model_loader import load_hooked_model, model_runtime_metadata
from src.utils.prompting import decision_token_ids, encode_decision_prompt

AXES = ("warmth", "competence")
DEFAULT_STRENGTHS = "-0.1,-0.05,0,0.05,0.1"


def main() -> None:
    args = parse_args()
    strengths = tuple(
        float(v.strip()) for v in args.strengths.split(",") if v.strip()
    )
    if 0.0 not in strengths:
        raise ValueError("--strengths must include 0.")

    cfg = load_config(args.config)
    vectors_dir = Path(cfg.paths.processed) / args.vectors_subdir
    table_dir = Path(cfg.paths.results) / "tables"
    log_dir = Path(cfg.paths.logs)
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- metadata ---
    meta = json.loads((vectors_dir / "meta.json").read_text(encoding="utf-8"))
    model_name = str(meta["model"])
    layer = int(meta["probe_layer"])
    cfg = replace(cfg, model=replace(cfg.model, name=model_name))

    # --- stimuli ---
    records_by_condition = load_story_records(
        Path(cfg.paths.stimuli) / "concept_stories.jsonl"
    )
    counts = {c: len(records_by_condition[c]) for c in CONDITIONS}
    slices = condition_slices(counts)
    train_topics, test_topics = train_test_topics(
        records_by_condition, cfg.probing.seed, args.n_test_topics
    )

    # --- activations → mean_resid_norm + raw concept directions (train only) ---
    activations_by_condition = {
        c: np.load(vectors_dir / f"X_{c}.npy").astype(np.float32)
        for c in CONDITIONS
    }
    all_activations = np.concatenate(
        [activations_by_condition[c] for c in CONDITIONS], axis=0
    )
    mean_resid_norm = float(np.linalg.norm(all_activations, axis=1).mean())

    raw_train_means: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        local_train = rows_for_topics(records_by_condition[condition], train_topics)
        raw_train_means[condition] = activations_by_condition[condition][
            local_train
        ].mean(axis=0)

    raw_vectors = {
        "warmth": raw_train_means["high_warmth"] - raw_train_means["low_warmth"],
        "competence": (
            raw_train_means["high_competence"] - raw_train_means["low_competence"]
        ),
    }

    # --- random control (orthogonalized to each axis direction) ---
    rng = np.random.default_rng(cfg.probing.seed)

    # --- load model ---
    print(f"[model] loading {model_name}", flush=True)
    model = load_hooked_model(cfg)
    model.eval()
    hook_name = residual_hook_name(layer)

    # fail-fast tokenization check — cheap, catches Llama/Qwen surprises
    check_prompt = judgement_prompt("A person completed a routine task.", "warmth")
    rendered, _ = encode_decision_prompt(model, check_prompt, args.prompt_format)
    yes_id, no_id = decision_token_ids(model, rendered, args.prompt_format)
    print(
        f"[tokens] ' Yes'={yes_id}  ' No'={no_id}  "
        f"(single-token check passed, hook={hook_name})",
        flush=True,
    )

    # --- steering loop ---
    raw_rows: list[dict] = []

    for axis in AXES:
        random_vec = rng.normal(size=int(meta["d_model"])).astype(np.float32)
        random_vec -= unit(raw_vectors[axis]) * float(
            random_vec @ unit(raw_vectors[axis])
        )
        directions = {
            "raw_dense": raw_vectors[axis],
            "random": random_vec,
        }

        test_conditions = (
            (f"high_{axis}", 1),
            (f"low_{axis}", 0),
        )
        test_records: list[tuple[dict, int]] = []
        for condition, label in test_conditions:
            for record in records_by_condition[condition]:
                if int(record["topic_idx"]) in set(test_topics.tolist()):
                    test_records.append((record, label))

        # baseline
        baseline_margins: dict[str, float] = {}
        for idx, (record, label) in enumerate(test_records, start=1):
            prompt = judgement_prompt(record["text"], axis)
            margin = yes_no_margin(
                model, prompt, hook_name, prompt_format=args.prompt_format
            )
            baseline_margins[record["id"]] = margin
            raw_rows.append(
                {
                    "mode": "baseline",
                    "axis": axis,
                    "story_id": record["id"],
                    "topic_idx": int(record["topic_idx"]),
                    "condition": record["condition"],
                    "label": label,
                    "direction": "baseline",
                    "strength": 0.0,
                    "margin": margin,
                    "delta_margin": 0.0,
                }
            )
            if idx % 10 == 0:
                print(f"[baseline] {axis} {idx}/{len(test_records)}", flush=True)

        # steering
        for direction_name, vector in directions.items():
            for strength in strengths:
                if strength == 0.0:
                    for record, label in test_records:
                        raw_rows.append(
                            {
                                "mode": "steering",
                                "axis": axis,
                                "story_id": record["id"],
                                "topic_idx": int(record["topic_idx"]),
                                "condition": record["condition"],
                                "label": label,
                                "direction": direction_name,
                                "strength": strength,
                                "margin": baseline_margins[record["id"]],
                                "delta_margin": 0.0,
                            }
                        )
                    continue

                hook = make_steering_hook(vector, strength * mean_resid_norm)
                for record, label in test_records:
                    prompt = judgement_prompt(record["text"], axis)
                    margin = yes_no_margin(
                        model,
                        prompt,
                        hook_name,
                        hook,
                        prompt_format=args.prompt_format,
                    )
                    raw_rows.append(
                        {
                            "mode": "steering",
                            "axis": axis,
                            "story_id": record["id"],
                            "topic_idx": int(record["topic_idx"]),
                            "condition": record["condition"],
                            "label": label,
                            "direction": direction_name,
                            "strength": strength,
                            "margin": margin,
                            "delta_margin": margin - baseline_margins[record["id"]],
                        }
                    )
                print(
                    f"[steering] {axis} {direction_name} strength={strength:+.2f}",
                    flush=True,
                )

    # --- summaries ---
    summary_rows = summarize_baseline(raw_rows, cfg.probing.seed)
    summary_rows.extend(summarize_steering(raw_rows, cfg.probing.seed))

    # --- write outputs ---
    raw_path = table_dir / f"steering_dense_raw_{args.label}.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    check_file_size(raw_path)

    summary_path = table_dir / f"steering_dense_{args.label}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    log = {
        "label": args.label,
        "model": model_name,
        "probe_layer": layer,
        "hook": hook_name,
        "seed": cfg.probing.seed,
        "train_topics": train_topics.tolist(),
        "test_topics": test_topics.tolist(),
        "mean_resid_norm": mean_resid_norm,
        "strengths": list(strengths),
        "n_test_topics": args.n_test_topics,
        "prompt_format": args.prompt_format,
        "rendered_prompt_example": rendered,
        "runtime": model_runtime_metadata(model),
        "raw_output": str(raw_path),
        "summary_output": str(summary_path),
    }
    log_path = log_dir / f"steering_dense_{args.label}.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"[done] {raw_path}")
    print(f"[done] {summary_path}")
    print(f"[done] {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dense (SAE-free) concept steering for any model with concept vectors. "
            "Replicates the raw_dense path of src/gemma_scope_causality.py."
        )
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--vectors-subdir",
        required=True,
        help="Subdirectory under data/processed/ containing concept vectors.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Short identifier used in output filenames.",
    )
    parser.add_argument(
        "--strengths",
        default=DEFAULT_STRENGTHS,
        help="Comma-separated steering strengths in units of mean_resid_norm; must include 0.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("raw", "native-chat"),
        default="raw",
        help="Decision-prompt rendering mode (Gemma 4 jobs use native-chat).",
    )
    parser.add_argument(
        "--n-test-topics",
        type=int,
        default=10,
        help="Number of topics held out for testing (same split as gemma_scope_causality).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
