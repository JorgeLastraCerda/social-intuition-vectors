from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy import sparse

from src.gemma_scope_analysis import decode_direction, load_sae
from src.gemma_scope_utils import (
    CONDITIONS,
    bootstrap_mean_ci,
    check_file_size,
    condition_slices,
    decompose_feature_axes,
    load_story_records,
    smallest_energy_feature_set,
    sparse_mean,
)
from src.utils.config import load_config
from src.utils.hooks import residual_hook_name
from src.utils.model_loader import load_hooked_model


AXES = ("warmth", "competence")
STRENGTHS = (-0.5, -0.25, 0.0, 0.25, 0.5)


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / (np.linalg.norm(vector) + 1e-12)


def judgement_prompt(text: str, axis: str) -> str:
    definition = (
        "warm, caring, considerate, and well-intentioned"
        if axis == "warmth"
        else "competent, capable, effective, and skilled"
    )
    return (
        f"Read the story below. Is the protagonist high in {axis}, meaning "
        f"{definition}? Answer only Yes or No.\n\nStory:\n{text}\n\nAnswer:"
    )


def candidate_token_id(
    model,
    candidate: str,
) -> int:
    candidate_tokens = model.to_tokens(candidate, prepend_bos=False)
    if candidate_tokens.numel() != 1:
        raise ValueError(
            f"Candidate {candidate!r} must tokenize to exactly one token; "
            f"got shape {tuple(candidate_tokens.shape)}."
        )
    return int(candidate_tokens.item())


def yes_no_margin(
    model,
    prompt: str,
    hook_name: str,
    hook_fn: Callable | None = None,
) -> float:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    kwargs = {}
    if hook_fn is not None:
        kwargs["fwd_hooks"] = [(hook_name, hook_fn)]
    with torch.no_grad():
        logits = model.run_with_hooks(prompt_tokens, **kwargs)
    yes_id = candidate_token_id(model, " Yes")
    no_id = candidate_token_id(model, " No")
    next_token_logits = logits[0, -1]
    return float((next_token_logits[yes_id] - next_token_logits[no_id]).item())


def make_steering_hook(vector: np.ndarray, alpha: float):
    vector_tensor = torch.from_numpy(unit(vector))

    def hook_fn(residual: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        direction = vector_tensor.to(device=residual.device, dtype=residual.dtype)
        return residual + alpha * direction

    return hook_fn


def make_error_preserving_ablation_hook(sae, feature_indices: np.ndarray):
    feature_indices_tensor = torch.from_numpy(feature_indices.astype(np.int64))

    def hook_fn(residual: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        if feature_indices_tensor.numel() == 0:
            return residual
        original_shape = residual.shape
        flat = residual.reshape(-1, original_shape[-1]).to(
            device=sae.W_dec.device,
            dtype=sae.W_dec.dtype,
        )
        features = sae.encode(flat)
        original_reconstruction = sae.decode(features)
        modified = features.clone()
        indices = feature_indices_tensor.to(device=features.device)
        modified[:, indices] = 0
        modified_reconstruction = sae.decode(modified)
        result = flat + modified_reconstruction - original_reconstruction
        return result.reshape(original_shape).to(dtype=residual.dtype)

    return hook_fn


def train_test_topics(
    records_by_condition: dict[str, list[dict]],
    seed: int,
    n_test_topics: int,
) -> tuple[np.ndarray, np.ndarray]:
    topics = np.array(
        sorted(
            {
                int(record["topic_idx"])
                for records in records_by_condition.values()
                for record in records
            }
        )
    )
    if n_test_topics >= len(topics):
        raise ValueError("n_test_topics must be smaller than the number of topics.")
    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(topics, size=n_test_topics, replace=False))
    train = np.array([topic for topic in topics if topic not in set(test)])
    return train, test


def rows_for_topics(
    records: list[dict],
    topics: np.ndarray,
) -> np.ndarray:
    topic_set = set(int(topic) for topic in topics)
    return np.array(
        [
            i
            for i, record in enumerate(records)
            if int(record["topic_idx"]) in topic_set
        ],
        dtype=np.int64,
    )


def summarize_steering(
    rows: list[dict],
    seed: int,
) -> list[dict]:
    summary: list[dict] = []
    keys = sorted(
        {
            (row["axis"], row["direction"], float(row["strength"]))
            for row in rows
            if row["mode"] == "steering"
        }
    )
    for axis, direction, strength in keys:
        selected = [
            row
            for row in rows
            if row["mode"] == "steering"
            and row["axis"] == axis
            and row["direction"] == direction
            and float(row["strength"]) == strength
        ]
        values = np.array([float(row["delta_margin"]) for row in selected])
        groups = np.array([int(row["topic_idx"]) for row in selected])
        estimate, low, high = bootstrap_mean_ci(values, groups, seed)
        summary.append(
            {
                "mode": "steering",
                "axis": axis,
                "direction": direction,
                "strength": strength,
                "effect": estimate,
                "ci_low": low,
                "ci_high": high,
                "n_stories": len(selected),
                "n_topics": len(np.unique(groups)),
            }
        )
    return summary


def summarize_baseline(
    rows: list[dict],
    seed: int,
) -> list[dict]:
    summary: list[dict] = []
    for axis in AXES:
        selected = [
            row
            for row in rows
            if row["mode"] == "baseline" and row["axis"] == axis
        ]
        accuracy = np.array(
            [
                (float(row["margin"]) > 0.0) == bool(int(row["label"]))
                for row in selected
            ],
            dtype=np.float64,
        )
        accuracy_groups = np.array([int(row["topic_idx"]) for row in selected])
        estimate, low, high = bootstrap_mean_ci(
            accuracy,
            accuracy_groups,
            seed,
        )
        summary.append(
            {
                "mode": "baseline",
                "axis": axis,
                "direction": "accuracy",
                "strength": "",
                "effect": estimate,
                "ci_low": low,
                "ci_high": high,
                "n_stories": len(selected),
                "n_topics": len(np.unique(accuracy_groups)),
            }
        )

        topic_gaps: list[float] = []
        topic_ids: list[int] = []
        for topic in sorted({int(row["topic_idx"]) for row in selected}):
            topic_rows = [row for row in selected if int(row["topic_idx"]) == topic]
            high = next(row for row in topic_rows if row["label"] == 1)
            low_row = next(row for row in topic_rows if row["label"] == 0)
            topic_gaps.append(float(high["margin"]) - float(low_row["margin"]))
            topic_ids.append(topic)
        estimate, low, high = bootstrap_mean_ci(
            np.array(topic_gaps),
            np.array(topic_ids),
            seed,
        )
        summary.append(
            {
                "mode": "baseline",
                "axis": axis,
                "direction": "high_low_margin_gap",
                "strength": "",
                "effect": estimate,
                "ci_low": low,
                "ci_high": high,
                "n_stories": len(selected),
                "n_topics": len(topic_ids),
            }
        )
    return summary


def summarize_ablation(
    rows: list[dict],
    seed: int,
) -> list[dict]:
    summary: list[dict] = []
    for axis in AXES:
        baseline = {
            row["story_id"]: row
            for row in rows
            if row["mode"] == "baseline" and row["axis"] == axis
        }
        interventions = sorted(
            {
                row["direction"]
                for row in rows
                if row["mode"] == "ablation" and row["axis"] == axis
            }
        )
        for intervention in interventions:
            selected = [
                row
                for row in rows
                if row["mode"] == "ablation"
                and row["axis"] == axis
                and row["direction"] == intervention
            ]
            topic_effects: list[float] = []
            topic_ids: list[int] = []
            for topic in sorted({int(row["topic_idx"]) for row in selected}):
                topic_rows = [row for row in selected if int(row["topic_idx"]) == topic]
                high = next(row for row in topic_rows if row["label"] == 1)
                low = next(row for row in topic_rows if row["label"] == 0)
                baseline_gap = (
                    float(baseline[high["story_id"]]["margin"])
                    - float(baseline[low["story_id"]]["margin"])
                )
                ablated_gap = float(high["margin"]) - float(low["margin"])
                topic_effects.append(ablated_gap - baseline_gap)
                topic_ids.append(topic)
            estimate, low, high = bootstrap_mean_ci(
                np.array(topic_effects),
                np.array(topic_ids),
                seed,
            )
            summary.append(
                {
                    "mode": "ablation",
                    "axis": axis,
                    "direction": intervention,
                    "strength": "",
                    "effect": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_stories": len(selected),
                    "n_topics": len(topic_ids),
                }
            )
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    vectors_dir = Path(cfg.paths.processed) / args.vectors_subdir
    scope_dir = Path(cfg.paths.processed) / args.scope_subdir
    table_dir = Path(cfg.paths.results) / "tables"
    log_dir = Path(cfg.paths.logs)
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((vectors_dir / "meta.json").read_text(encoding="utf-8"))
    model_name = str(meta["model"])
    layer = int(meta["probe_layer"])
    cfg = replace(cfg, model=replace(cfg.model, name=model_name))
    records_by_condition = load_story_records(
        Path(cfg.paths.stimuli) / "concept_stories.jsonl"
    )
    counts = {condition: len(records_by_condition[condition]) for condition in CONDITIONS}
    slices = condition_slices(counts)
    train_topics, test_topics = train_test_topics(
        records_by_condition,
        cfg.probing.seed,
        args.n_test_topics,
    )

    activations_by_condition = {
        condition: np.load(vectors_dir / f"X_{condition}.npy").astype(np.float32)
        for condition in CONDITIONS
    }
    all_activations = np.concatenate(
        [activations_by_condition[condition] for condition in CONDITIONS],
        axis=0,
    )
    mean_resid_norm = float(np.linalg.norm(all_activations, axis=1).mean())
    feature_matrix = sparse.load_npz(scope_dir / "story_features_65k.npz").tocsr()
    sae, _, _ = load_sae(args.sae_release, args.sae_id, cfg.model.device)

    feature_train_means: dict[str, np.ndarray] = {}
    raw_train_means: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        local_train = rows_for_topics(records_by_condition[condition], train_topics)
        global_rows = slices[condition].start + local_train
        feature_train_means[condition] = sparse_mean(feature_matrix[global_rows])
        raw_train_means[condition] = activations_by_condition[condition][
            local_train
        ].mean(axis=0)
    feature_warmth = (
        feature_train_means["high_warmth"]
        - feature_train_means["low_warmth"]
    )
    feature_competence = (
        feature_train_means["high_competence"]
        - feature_train_means["low_competence"]
    )
    feature_vectors = decompose_feature_axes(feature_warmth, feature_competence)
    decoded = {
        name: decode_direction(sae, vector)
        for name, vector in feature_vectors.items()
    }
    raw_vectors = {
        "warmth": raw_train_means["high_warmth"] - raw_train_means["low_warmth"],
        "competence": (
            raw_train_means["high_competence"]
            - raw_train_means["low_competence"]
        ),
    }
    selected_features = {
        name: smallest_energy_feature_set(vector, args.ablation_energy)
        for name, vector in feature_vectors.items()
    }
    rng = np.random.default_rng(cfg.probing.seed)
    excluded_features = np.unique(
        np.concatenate(
            [
                selected_features["warmth"],
                selected_features["competence"],
                selected_features["shared"],
            ]
        )
    )
    random_pool = np.setdiff1d(
        np.arange(int(sae.cfg.d_sae), dtype=np.int64),
        excluded_features,
        assume_unique=True,
    )
    random_feature_sets = {
        axis: np.sort(
            rng.choice(
                random_pool,
                size=len(selected_features[axis]),
                replace=False,
            )
        )
        for axis in AXES
    }

    causal_vectors_path = scope_dir / "causal_vectors_65k.npz"
    np.savez_compressed(
        causal_vectors_path,
        train_topics=train_topics,
        test_topics=test_topics,
        mean_resid_norm=np.array(mean_resid_norm),
        **{f"feature_{name}": value for name, value in feature_vectors.items()},
        **{f"residual_{name}": value for name, value in decoded.items()},
        **{f"raw_{name}": value for name, value in raw_vectors.items()},
        **{f"selected_{name}": value for name, value in selected_features.items()},
        **{f"selected_random_{axis}": value for axis, value in random_feature_sets.items()},
    )
    check_file_size(causal_vectors_path)

    print(f"[model] loading {model_name}", flush=True)
    model = load_hooked_model(cfg)
    model.eval()
    hook_name = residual_hook_name(layer)
    yes_token_id = candidate_token_id(model, " Yes")
    no_token_id = candidate_token_id(model, " No")
    print(
        f"[tokens] Yes={yes_token_id} No={no_token_id}; "
        "single-pass next-token margin enabled",
        flush=True,
    )
    raw_rows: list[dict] = []
    direction_metadata: dict[str, dict] = {}
    for axis in AXES:
        other = "competence" if axis == "warmth" else "warmth"
        random = rng.normal(size=int(meta["d_model"])).astype(np.float32)
        random -= unit(raw_vectors[axis]) * float(random @ unit(raw_vectors[axis]))
        directions = {
            "raw_dense": raw_vectors[axis],
            "sae_reconstructed": decoded[axis],
            "axis_specific": decoded[f"{axis}_specific"],
            "shared": decoded["shared"],
            "other_axis": decoded[other],
            "random": random,
        }
        direction_metadata[axis] = {
            name: {"norm": float(np.linalg.norm(vector))}
            for name, vector in directions.items()
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

        baseline_margins: dict[str, float] = {}
        for index, (record, label) in enumerate(test_records, start=1):
            prompt = judgement_prompt(record["text"], axis)
            margin = yes_no_margin(model, prompt, hook_name)
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
            if index % 10 == 0:
                print(f"[baseline] {axis} {index}/{len(test_records)}", flush=True)

        for direction_name, vector in directions.items():
            for strength in STRENGTHS:
                if strength == 0:
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
                    margin = yes_no_margin(model, prompt, hook_name, hook)
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

        ablation_sets = {
            "target_axis": selected_features[axis],
            "other_axis": selected_features[other],
            "shared": selected_features["shared"],
            "random_features": random_feature_sets[axis],
        }
        for name, indices in ablation_sets.items():
            hook = make_error_preserving_ablation_hook(sae, indices)
            for record, label in test_records:
                prompt = judgement_prompt(record["text"], axis)
                margin = yes_no_margin(model, prompt, hook_name, hook)
                raw_rows.append(
                    {
                        "mode": "ablation",
                        "axis": axis,
                        "story_id": record["id"],
                        "topic_idx": int(record["topic_idx"]),
                        "condition": record["condition"],
                        "label": label,
                        "direction": name,
                        "strength": "",
                        "margin": margin,
                        "delta_margin": margin - baseline_margins[record["id"]],
                    }
                )
            print(f"[ablation] {axis} {name} n={len(indices)}", flush=True)

    summary_rows = summarize_baseline(raw_rows, cfg.probing.seed)
    summary_rows.extend(summarize_steering(raw_rows, cfg.probing.seed))
    summary_rows.extend(summarize_ablation(raw_rows, cfg.probing.seed))
    raw_path = table_dir / f"gemma_scope_causality_raw_{args.label}.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    check_file_size(raw_path)
    summary_path = table_dir / f"gemma_scope_causality_{args.label}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    log = {
        "label": args.label,
        "model": model_name,
        "probe_layer": layer,
        "hook": hook_name,
        "seed": cfg.probing.seed,
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "train_topics": train_topics.tolist(),
        "test_topics": test_topics.tolist(),
        "mean_resid_norm": mean_resid_norm,
        "strengths": list(STRENGTHS),
        "ablation_energy": args.ablation_energy,
        "selected_feature_counts": {
            name: len(indices) for name, indices in selected_features.items()
        },
        "direction_metadata": direction_metadata,
        "raw_output": str(raw_path),
        "summary_output": str(summary_path),
        "causal_vectors": str(causal_vectors_path),
    }
    log_path = log_dir / f"gemma_scope_causality_{args.label}.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[done] {raw_path}")
    print(f"[done] {summary_path}")
    print(f"[done] {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Held-out Gemma Scope concept steering and feature ablation."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--vectors-subdir", required=True)
    parser.add_argument("--scope-subdir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--n-test-topics", type=int, default=10)
    parser.add_argument("--ablation-energy", type=float, default=0.50)
    return parser.parse_args()


if __name__ == "__main__":
    main()
