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
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

# -- reuse validated helpers from the Gemma Scope causality script (no SAE) --
from src.gemma_scope_causality import (
    judgement_prompt,
    rows_for_topics,
    summarize_baseline,
    summarize_steering,
    train_test_topics,
    unit,
    yes_no_margin,
)
from src.gemma_scope_utils import (
    CONDITIONS,
    bootstrap_mean_ci,
    check_file_size,
    load_story_records,
)
from src.steering_calibration import (
    calibrated_alpha,
    descriptive_null_metrics,
    directional_sd,
    make_torch_hook,
    paired_topic_difference_ci,
    standardized_shift,
)
from src.steering_checkpoint import CheckpointStore, atomic_json_write, sha256_file
from src.utils.config import load_config
from src.utils.hooks import residual_hook_name
from src.utils.model_loader import load_hooked_model, model_runtime_metadata
from src.utils.prompting import decision_token_ids, encode_decision_prompt

AXES = ("warmth", "competence")
DEFAULT_STRENGTHS = "-0.1,-0.05,0,0.05,0.1"


def git_commit() -> str:
    """Return the checked-out commit for provenance without requiring Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def atomic_csv_write(path: Path, rows: list[dict]) -> None:
    """Write a complete CSV and atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def orthogonal_random_directions(
    warmth: np.ndarray,
    competence: np.ndarray,
    *,
    n_directions: int,
    seed: int,
) -> list[np.ndarray]:
    """Return deterministic unit vectors orthogonal to the two-axis span."""
    if n_directions < 1:
        raise ValueError("n_directions must be at least 1")
    basis, _ = np.linalg.qr(
        np.column_stack([unit(warmth), unit(competence)]).astype(np.float64)
    )
    rng = np.random.default_rng(seed)
    directions: list[np.ndarray] = []
    for _ in range(n_directions):
        candidate = rng.normal(size=warmth.shape[0]).astype(np.float64)
        candidate -= basis @ (basis.T @ candidate)
        directions.append(unit(candidate).astype(np.float32))
    return directions


def direction_metadata(judgment_axis: str, direction: str) -> tuple[str, str, str]:
    if direction in AXES:
        kind = "target" if direction == judgment_axis else "cross_axis"
        return direction, kind, ""
    if direction.startswith("random_"):
        return "", "random", direction.removeprefix("random_")
    if direction == "raw_dense":
        return judgment_axis, "target", ""
    if direction == "random":
        return "", "random", "000"
    return "", direction, ""


def empirical_null_rows(summary_rows: list[dict]) -> list[dict]:
    """Compare target/cross-axis endpoint effects and slopes with the random null."""
    output: list[dict] = []
    for judgment_axis in AXES:
        steering = [
            row
            for row in summary_rows
            if row.get("mode") == "steering" and row.get("axis") == judgment_axis
        ]
        strengths = sorted(
            {
                float(row["strength"])
                for row in steering
                if float(row["strength"]) != 0.0
            }
        )
        endpoint = max(strengths)
        random_endpoint = np.array(
            [
                float(row["effect"])
                for row in steering
                if str(row["direction"]).startswith("random_")
                and float(row["strength"]) == endpoint
            ]
        )
        for steering_axis in AXES:
            selected = [row for row in steering if row["direction"] == steering_axis]
            if not selected:
                continue
            x = np.array([float(row["strength"]) for row in selected])
            y = np.array([float(row["effect"]) for row in selected])
            slope = float(np.polyfit(x, y, 1)[0])
            endpoint_effect = float(
                next(
                    row["effect"]
                    for row in selected
                    if float(row["strength"]) == endpoint
                )
            )
            empirical_p = (
                float(
                    (1 + np.sum(np.abs(random_endpoint) >= abs(endpoint_effect)))
                    / (1 + len(random_endpoint))
                )
                if len(random_endpoint)
                else None
            )
            output.append(
                {
                    "judgment_axis": judgment_axis,
                    "steering_axis": steering_axis,
                    "direction_type": "target"
                    if judgment_axis == steering_axis
                    else "cross_axis",
                    "endpoint_strength": endpoint,
                    "endpoint_effect": endpoint_effect,
                    "slope": slope,
                    "n_random_directions": int(len(random_endpoint)),
                    "empirical_two_sided_p": empirical_p,
                }
            )
    return output


def summarize_calibrated_steering(rows: list[dict], seed: int) -> list[dict]:
    """Summarize steering separately for every intervention condition."""
    output: list[dict] = []
    keys = sorted(
        {
            (
                str(row["axis"]),
                str(row["direction"]),
                float(row["strength"]),
                str(row["intervention"]),
            )
            for row in rows
            if row["mode"] == "steering"
        }
    )
    for axis, direction, strength, intervention in keys:
        selected = [
            row
            for row in rows
            if row["mode"] == "steering"
            and row["axis"] == axis
            and row["direction"] == direction
            and float(row["strength"]) == strength
            and row["intervention"] == intervention
        ]
        values = np.asarray([float(row["delta_margin"]) for row in selected])
        groups = np.asarray([int(row["topic_idx"]) for row in selected])
        estimate, low, high = bootstrap_mean_ci(values, groups, seed)
        first = selected[0]
        output.append(
            {
                "mode": "steering",
                "axis": axis,
                "direction": direction,
                "strength": strength,
                "effect": estimate,
                "ci_low": low,
                "ci_high": high,
                "n": len(selected),
                "intervention": intervention,
                "control_scale": first["control_scale"],
                "direction_sd": first["direction_sd"],
                "alpha_absolute": first["alpha_absolute"],
                "standardized_shift": first["standardized_shift"],
                "max_relative_norm_drift": max(
                    float(row["max_relative_norm_drift"]) for row in selected
                ),
            }
        )
    return output


def calibrated_null_rows(
    summary_rows: list[dict], raw_rows: list[dict], seed: int
) -> list[dict]:
    """Descriptive endpoint and slope ranks against the calibrated random null."""
    output: list[dict] = []
    for intervention in sorted(
        {
            str(row.get("intervention", ""))
            for row in summary_rows
            if row.get("intervention")
        }
    ):
        for judgment_axis in AXES:
            steering = [
                row
                for row in summary_rows
                if row.get("mode") == "steering"
                and row.get("axis") == judgment_axis
                and row.get("intervention") == intervention
            ]
            positive = sorted(
                {
                    float(row["strength"])
                    for row in steering
                    if float(row["strength"]) > 0
                }
            )
            if not positive:
                continue
            endpoint = positive[-1]
            random_names = sorted(
                {
                    str(row["direction"])
                    for row in steering
                    if str(row["direction"]).startswith("random_")
                }
            )
            random_endpoints: list[float] = []
            random_slopes: list[float] = []
            for name in random_names:
                selected = [row for row in steering if row["direction"] == name]
                random_endpoints.append(
                    float(
                        next(
                            row["effect"]
                            for row in selected
                            if float(row["strength"]) == endpoint
                        )
                    )
                )
                random_slopes.append(
                    float(
                        np.polyfit(
                            [float(row["strength"]) for row in selected],
                            [float(row["effect"]) for row in selected],
                            1,
                        )[0]
                    )
                )
            for steering_axis in AXES:
                selected = [
                    row for row in steering if row["direction"] == steering_axis
                ]
                if not selected or not random_names:
                    continue
                endpoint_effect = float(
                    next(
                        row["effect"]
                        for row in selected
                        if float(row["strength"]) == endpoint
                    )
                )
                slope = float(
                    np.polyfit(
                        [float(row["strength"]) for row in selected],
                        [float(row["effect"]) for row in selected],
                        1,
                    )[0]
                )
                endpoint_metrics = descriptive_null_metrics(
                    np.asarray(random_endpoints), endpoint_effect
                )
                slope_metrics = descriptive_null_metrics(
                    np.asarray(random_slopes), slope
                )
                paired, paired_low, paired_high = paired_topic_difference_ci(
                    raw_rows,
                    judgment_axis=judgment_axis,
                    steering_axis=steering_axis,
                    intervention=intervention,
                    endpoint_strength=endpoint,
                    seed=seed,
                )
                output.append(
                    {
                        "judgment_axis": judgment_axis,
                        "steering_axis": steering_axis,
                        "direction_type": "target"
                        if judgment_axis == steering_axis
                        else "cross_axis",
                        "intervention": intervention,
                        "control_scale": selected[0]["control_scale"],
                        "endpoint_strength": endpoint,
                        "endpoint_effect": endpoint_effect,
                        "slope": slope,
                        "n_random_directions": len(random_names),
                        **{
                            f"endpoint_{key}": value
                            for key, value in endpoint_metrics.items()
                        },
                        **{
                            f"slope_{key}": value
                            for key, value in slope_metrics.items()
                        },
                        "paired_topic_difference": paired,
                        "paired_topic_ci_low": paired_low,
                        "paired_topic_ci_high": paired_high,
                    }
                )
    return output


def main() -> None:
    args = parse_args()
    strengths = tuple(float(v.strip()) for v in args.strengths.split(",") if v.strip())
    interventions = tuple(
        value.strip() for value in args.interventions.split(",") if value.strip()
    )
    if not interventions or any(
        value not in ("additive", "norm_preserving") for value in interventions
    ):
        raise ValueError(
            "--interventions must contain additive and/or norm_preserving."
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
    train_topics, test_topics = train_test_topics(
        records_by_condition, cfg.probing.seed, args.n_test_topics
    )

    # --- activations → mean_resid_norm + raw concept directions (train only) ---
    activations_by_condition = {
        c: np.load(vectors_dir / f"X_{c}.npy").astype(np.float32) for c in CONDITIONS
    }
    all_activations = np.concatenate(
        [activations_by_condition[c] for c in CONDITIONS], axis=0
    )
    mean_resid_norm = float(np.linalg.norm(all_activations, axis=1).mean())

    train_activations = np.concatenate(
        [
            activations_by_condition[condition][
                rows_for_topics(records_by_condition[condition], train_topics)
            ]
            for condition in CONDITIONS
        ],
        axis=0,
    )

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

    if args.vector_kind == "denoised":
        denoised_path = vectors_dir / "concept_vectors_denoised.npz"
        if not denoised_path.exists():
            raise FileNotFoundError(
                f"{denoised_path} not found; run neutral extraction and PCA denoising first."
            )
        denoised = np.load(denoised_path)
        components = np.asarray(denoised["neutral_pca_components"], dtype=np.float64)
        raw_vectors = {
            axis: (
                np.asarray(vector, dtype=np.float64)
                - components.T @ (components @ np.asarray(vector, dtype=np.float64))
            ).astype(np.float32)
            for axis, vector in raw_vectors.items()
        }

    # --- random control (orthogonalized to each axis direction) ---
    enhanced = args.include_cross_axis or args.n_random_directions != 1
    random_directions = orthogonal_random_directions(
        raw_vectors["warmth"],
        raw_vectors["competence"],
        n_directions=args.n_random_directions,
        seed=cfg.probing.seed,
    )
    direction_sds = {
        "warmth": directional_sd(train_activations, raw_vectors["warmth"]),
        "competence": directional_sd(train_activations, raw_vectors["competence"]),
        **{
            f"random_{index:03d}": directional_sd(train_activations, vector)
            for index, vector in enumerate(random_directions)
        },
    }

    checkpoint: CheckpointStore | None = None
    if args.resume and not args.checkpoint_dir:
        raise ValueError("--resume requires --checkpoint-dir.")
    if args.checkpoint_dir:
        input_paths = {
            "config": Path(args.config),
            "stimuli": Path(cfg.paths.stimuli) / "concept_stories.jsonl",
            "meta": vectors_dir / "meta.json",
            **{
                f"activation_{condition}": vectors_dir / f"X_{condition}.npy"
                for condition in CONDITIONS
            },
        }
        if args.vector_kind == "denoised":
            input_paths["denoised_vectors"] = (
                vectors_dir / "concept_vectors_denoised.npz"
            )
        fingerprint = {
            "git_commit": git_commit(),
            "model": model_name,
            "model_revision": cfg.model.revision,
            "probe_layer": layer,
            "seed": cfg.probing.seed,
            "train_topics": train_topics.tolist(),
            "test_topics": test_topics.tolist(),
            "arguments": {
                "label": args.label,
                "vectors_subdir": args.vectors_subdir,
                "vector_kind": args.vector_kind,
                "include_cross_axis": args.include_cross_axis,
                "n_random_directions": args.n_random_directions,
                "strengths": list(strengths),
                "prompt_format": args.prompt_format,
                "control_scale": args.control_scale,
                "interventions": list(interventions),
                "n_test_topics": args.n_test_topics,
            },
            "input_sha256": {
                name: sha256_file(path) for name, path in sorted(input_paths.items())
            },
        }
        checkpoint = CheckpointStore(
            Path(args.checkpoint_dir), fingerprint, resume=args.resume
        )

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
    work_sequence = 0

    for axis in AXES:
        if enhanced:
            directions = {axis: raw_vectors[axis]}
            if args.include_cross_axis:
                other_axis = "competence" if axis == "warmth" else "warmth"
                directions[other_axis] = raw_vectors[other_axis]
            directions.update(
                {
                    f"random_{index:03d}": vector
                    for index, vector in enumerate(random_directions)
                }
            )
        else:
            directions = {
                "raw_dense": raw_vectors[axis],
                "random": random_directions[0],
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
        baseline_key = {"mode": "baseline", "axis": axis}
        baseline_rows = (
            checkpoint.read(work_sequence, baseline_key) if checkpoint else None
        )
        if baseline_rows is None:
            baseline_rows = []
            for idx, (record, label) in enumerate(test_records, start=1):
                prompt = judgement_prompt(record["text"], axis)
                margin = yes_no_margin(
                    model, prompt, hook_name, prompt_format=args.prompt_format
                )
                baseline_rows.append(
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
                        "judgment_axis": axis,
                        "steering_axis": "",
                        "direction_type": "baseline",
                        "random_id": "",
                        "vector_kind": args.vector_kind,
                        "intervention": "baseline",
                        "control_scale": args.control_scale,
                        "direction_sd": "",
                        "alpha_absolute": 0.0,
                        "standardized_shift": 0.0,
                        "max_relative_norm_drift": 0.0,
                        "decision_flipped": False,
                    }
                )
                if idx % 10 == 0:
                    print(f"[baseline] {axis} {idx}/{len(test_records)}", flush=True)
            if checkpoint:
                checkpoint.write(work_sequence, baseline_key, baseline_rows)
        else:
            print(f"[resume] baseline {axis}", flush=True)
        raw_rows.extend(baseline_rows)
        baseline_margins = {
            str(row["story_id"]): float(row["margin"]) for row in baseline_rows
        }
        work_sequence += 1

        # steering
        for intervention in interventions:
            for direction_name, vector in directions.items():
                direction_sd_value = direction_sds.get(direction_name)
                if direction_sd_value is None:
                    # Legacy direction labels map to their judgment axis.
                    direction_sd_value = direction_sds[axis]
                target_sd = direction_sds[axis]
                for strength in strengths:
                    work_key = {
                        "mode": "steering",
                        "axis": axis,
                        "intervention": intervention,
                        "direction": direction_name,
                        "strength": strength,
                    }
                    completed_rows = (
                        checkpoint.read(work_sequence, work_key) if checkpoint else None
                    )
                    if completed_rows is not None:
                        raw_rows.extend(completed_rows)
                        print(
                            f"[resume] {axis} {direction_name} {intervention} "
                            f"strength={strength:+.2f}",
                            flush=True,
                        )
                        work_sequence += 1
                        continue
                    alpha = calibrated_alpha(
                        strength=strength,
                        mean_residual_norm=mean_resid_norm,
                        target_direction_sd=target_sd,
                        direction_sd=direction_sd_value,
                        control_scale=args.control_scale,
                    )
                    shift_sd = standardized_shift(alpha, direction_sd_value)
                    if strength == 0.0:
                        hook = None
                        diagnostics = None
                    else:
                        hook, diagnostics = make_torch_hook(vector, alpha, intervention)
                    work_rows: list[dict] = []
                    for record, label in test_records:
                        prompt = judgement_prompt(record["text"], axis)
                        margin = (
                            baseline_margins[record["id"]]
                            if hook is None
                            else yes_no_margin(
                                model,
                                prompt,
                                hook_name,
                                hook,
                                prompt_format=args.prompt_format,
                            )
                        )
                        steering_axis, direction_type, random_id = direction_metadata(
                            axis, direction_name
                        )
                        work_rows.append(
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
                                "judgment_axis": axis,
                                "steering_axis": steering_axis,
                                "direction_type": direction_type,
                                "random_id": random_id,
                                "vector_kind": args.vector_kind,
                                "intervention": intervention,
                                "control_scale": args.control_scale,
                                "direction_sd": direction_sd_value,
                                "alpha_absolute": alpha,
                                "standardized_shift": shift_sd,
                                "max_relative_norm_drift": (
                                    0.0
                                    if diagnostics is None
                                    else diagnostics.max_relative_norm_drift
                                ),
                                "decision_flipped": bool(
                                    np.signbit(margin)
                                    != np.signbit(baseline_margins[record["id"]])
                                ),
                            }
                        )
                    if checkpoint:
                        checkpoint.write(work_sequence, work_key, work_rows)
                    raw_rows.extend(work_rows)
                    print(
                        f"[steering] {axis} {direction_name} {intervention} "
                        f"strength={strength:+.2f} alpha={alpha:+.4f}",
                        flush=True,
                    )
                    work_sequence += 1

    if checkpoint:
        raw_rows = checkpoint.consolidate(work_sequence)

    # --- summaries ---
    calibrated_run = args.control_scale == "sd_matched" or interventions != (
        "additive",
    )
    if calibrated_run:
        summary_rows = summarize_calibrated_steering(raw_rows, cfg.probing.seed)
    else:
        summary_rows = summarize_baseline(raw_rows, cfg.probing.seed)
        summary_rows.extend(summarize_steering(raw_rows, cfg.probing.seed))
    for row in summary_rows:
        steering_axis, direction_type, random_id = direction_metadata(
            str(row["axis"]), str(row["direction"])
        )
        row.update(
            {
                "judgment_axis": row["axis"],
                "steering_axis": steering_axis,
                "direction_type": direction_type,
                "random_id": random_id,
                "vector_kind": args.vector_kind,
                "intervention": row.get("intervention", "baseline"),
                "control_scale": row.get("control_scale", args.control_scale),
            }
        )
    if enhanced and calibrated_run:
        null_rows = calibrated_null_rows(summary_rows, raw_rows, cfg.probing.seed)
    else:
        null_rows = empirical_null_rows(summary_rows) if enhanced else []

    # --- write outputs ---
    raw_path = table_dir / f"steering_dense_raw_{args.label}.csv"
    atomic_csv_write(raw_path, raw_rows)
    check_file_size(raw_path)

    summary_path = table_dir / f"steering_dense_{args.label}.csv"
    atomic_csv_write(summary_path, summary_rows)
    null_path = table_dir / f"steering_dense_null_{args.label}.csv"
    if null_rows:
        atomic_csv_write(null_path, null_rows)

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
        "vector_kind": args.vector_kind,
        "include_cross_axis": args.include_cross_axis,
        "n_random_directions": args.n_random_directions,
        "prompt_format": args.prompt_format,
        "control_scale": args.control_scale,
        "interventions": list(interventions),
        "direction_sds_train_topics": direction_sds,
        "calibration_activation_rows": int(train_activations.shape[0]),
        "scientific_gate": "descriptive-only",
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "resumed": bool(args.resume),
        "rendered_prompt_example": rendered,
        "runtime": model_runtime_metadata(model),
        "raw_output": str(raw_path),
        "summary_output": str(summary_path),
        "null_output": str(null_path) if null_rows else None,
    }
    log_path = log_dir / f"steering_dense_{args.label}.json"
    atomic_json_write(log_path, log)

    print(f"[done] {raw_path}")
    print(f"[done] {summary_path}")
    if null_rows:
        print(f"[done] {null_path}")
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
        "--vector-kind",
        choices=("raw", "denoised"),
        default="raw",
        help="Steer with train-topic raw directions or their neutral-PCA-denoised forms.",
    )
    parser.add_argument(
        "--include-cross-axis",
        action="store_true",
        help="Also steer each judgment with the other concept axis.",
    )
    parser.add_argument(
        "--n-random-directions",
        type=int,
        default=1,
        help="Number of seeded controls orthogonal to the warmth/competence span.",
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
        "--control-scale",
        choices=("legacy_l2", "sd_matched"),
        default="legacy_l2",
        help="Scale controls by equal L2 alpha or equal projection-SD shift.",
    )
    parser.add_argument(
        "--interventions",
        default="additive",
        help="Comma-separated additive and/or norm_preserving interventions.",
    )
    parser.add_argument(
        "--n-test-topics",
        type=int,
        default=10,
        help="Number of topics held out for testing (same split as gemma_scope_causality).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Optional directory for atomic per-work-unit checkpoint shards.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching --checkpoint-dir; fingerprints must match exactly.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
