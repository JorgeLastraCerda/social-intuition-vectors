"""Native-HF calibrated steering for pinned Qwen3.6 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.gemma_scope_utils import bootstrap_mean_ci
from src.qwen36_pipeline import (
    CONDITIONS,
    _finish_runtime,
    _load_model_and_verify,
    _validate_native_config,
    load_full_records,
    stage_paths,
)
from src.steering_calibration import (
    calibrated_alpha,
    descriptive_null_metrics,
    directional_sd,
    intervene_tensor,
    paired_topic_difference_ci,
    standardized_shift,
    unit,
)
from src.utils.config import load_config, require_model_name

AXES = ("warmth", "competence")
STRENGTHS = (-0.1, -0.05, 0.0, 0.05, 0.1)


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


def train_test_topics(
    seed: int, n_test_topics: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    topics = np.arange(50, dtype=np.int64)
    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(topics, size=n_test_topics, replace=False))
    train = np.asarray([topic for topic in topics if topic not in set(test)])
    return train, test


def orthogonal_random_directions(
    warmth: np.ndarray, competence: np.ndarray, *, count: int, seed: int
) -> list[np.ndarray]:
    basis, _ = np.linalg.qr(np.column_stack([unit(warmth), unit(competence)]))
    rng = np.random.default_rng(seed)
    output: list[np.ndarray] = []
    for _ in range(count):
        candidate = rng.normal(size=warmth.size)
        candidate -= basis @ (basis.T @ candidate)
        output.append(unit(candidate).astype(np.float32))
    return output


def render_native_chat(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_prompt(tokenizer: Any, rendered: str) -> dict[str, torch.Tensor]:
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    return {key: value.to("cuda:0") for key, value in encoded.items()}


def continuation_ids(tokenizer: Any, rendered: str) -> tuple[int, int]:
    prefix = tokenizer(rendered, add_special_tokens=False)["input_ids"]

    def one(candidate: str) -> int:
        combined = tokenizer(rendered + candidate, add_special_tokens=False)[
            "input_ids"
        ]
        if combined[: len(prefix)] != prefix or len(combined) != len(prefix) + 1:
            raise ValueError(
                f"Native-chat continuation {candidate!r} is not one stable token."
            )
        return int(combined[-1])

    return one("Yes"), one("No")


def direction_metadata(judgment_axis: str, name: str) -> tuple[str, str, str]:
    if name in AXES:
        return name, "target" if name == judgment_axis else "cross_axis", ""
    return "", "random", name.removeprefix("random_")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted(
        {
            (row["axis"], row["direction"], row["strength"], row["intervention"])
            for row in rows
            if row["mode"] == "steering"
        }
    )
    for axis, direction, strength, intervention in keys:
        selected = [
            row
            for row in rows
            if row["mode"] == "steering"
            and (row["axis"], row["direction"], row["strength"], row["intervention"])
            == (axis, direction, strength, intervention)
        ]
        values = np.asarray([row["delta_margin"] for row in selected], dtype=float)
        groups = np.asarray([row["topic_idx"] for row in selected], dtype=int)
        effect, low, high = bootstrap_mean_ci(values, groups, seed)
        first = selected[0]
        output.append(
            {
                "mode": "steering",
                "axis": axis,
                "direction": direction,
                "strength": strength,
                "effect": effect,
                "ci_low": low,
                "ci_high": high,
                "n": len(selected),
                "intervention": intervention,
                "control_scale": "sd_matched",
                "direction_sd": first["direction_sd"],
                "alpha_absolute": first["alpha_absolute"],
                "standardized_shift": first["standardized_shift"],
                "max_relative_norm_drift": max(
                    row["max_relative_norm_drift"] for row in selected
                ),
            }
        )
    return output


def null_summary(
    summary: list[dict[str, Any]], raw_rows: list[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for intervention in ("additive", "norm_preserving"):
        for axis in AXES:
            rows = [
                row
                for row in summary
                if row["axis"] == axis and row["intervention"] == intervention
            ]
            random_names = sorted(
                {
                    row["direction"]
                    for row in rows
                    if row["direction"].startswith("random_")
                }
            )
            random_endpoint: list[float] = []
            random_slope: list[float] = []
            for name in random_names:
                selected = [row for row in rows if row["direction"] == name]
                random_endpoint.append(
                    next(row["effect"] for row in selected if row["strength"] == 0.1)
                )
                random_slope.append(
                    float(
                        np.polyfit(
                            [row["strength"] for row in selected],
                            [row["effect"] for row in selected],
                            1,
                        )[0]
                    )
                )
            for steering_axis in AXES:
                selected = [row for row in rows if row["direction"] == steering_axis]
                endpoint = next(
                    row["effect"] for row in selected if row["strength"] == 0.1
                )
                slope = float(
                    np.polyfit(
                        [row["strength"] for row in selected],
                        [row["effect"] for row in selected],
                        1,
                    )[0]
                )
                end_metrics = descriptive_null_metrics(
                    np.asarray(random_endpoint), endpoint
                )
                slope_metrics = descriptive_null_metrics(
                    np.asarray(random_slope), slope
                )
                paired, paired_low, paired_high = paired_topic_difference_ci(
                    raw_rows,
                    judgment_axis=axis,
                    steering_axis=steering_axis,
                    intervention=intervention,
                    endpoint_strength=0.1,
                    seed=seed,
                )
                output.append(
                    {
                        "judgment_axis": axis,
                        "steering_axis": steering_axis,
                        "direction_type": "target"
                        if axis == steering_axis
                        else "cross_axis",
                        "intervention": intervention,
                        "control_scale": "sd_matched",
                        "endpoint_strength": 0.1,
                        "endpoint_effect": endpoint,
                        "slope": slope,
                        "n_random_directions": len(random_names),
                        **{
                            f"endpoint_{key}": value
                            for key, value in end_metrics.items()
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
    cfg = load_config(args.config)
    _validate_native_config(cfg, require_cuda=True)
    if "transformer_lens" in sys.modules:
        raise RuntimeError("TransformerLens was imported in the native-HF process.")
    paths = stage_paths(cfg)
    vectors_dir = paths.vectors_dir
    label = f"{cfg.native_hf.label}_calibrated"
    table_dir = Path(cfg.paths.results) / "tables"
    log_dir = Path(cfg.paths.logs)
    raw_path = table_dir / f"steering_dense_raw_{label}.csv"
    summary_path = table_dir / f"steering_dense_{label}.csv"
    null_path = table_dir / f"steering_dense_null_{label}.csv"
    log_path = log_dir / f"steering_dense_{label}.json"
    collisions = [
        path for path in (raw_path, summary_path, null_path, log_path) if path.exists()
    ]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite calibrated outputs: {collisions}")

    buckets, stimuli_hash = load_full_records(
        Path(cfg.paths.stimuli) / "concept_stories.jsonl"
    )
    train_topics, test_topics = train_test_topics(cfg.probing.seed)
    activations = {
        condition: np.load(vectors_dir / f"X_{condition}.npy").astype(np.float32)
        for condition in CONDITIONS
    }
    all_activations = np.concatenate(list(activations.values()))
    mean_residual_norm = float(np.linalg.norm(all_activations, axis=1).mean())
    train_matrix = np.concatenate(
        [activations[condition][train_topics] for condition in CONDITIONS]
    )
    vectors = {
        "warmth": activations["high_warmth"][train_topics].mean(0)
        - activations["low_warmth"][train_topics].mean(0),
        "competence": activations["high_competence"][train_topics].mean(0)
        - activations["low_competence"][train_topics].mean(0),
    }
    randoms = orthogonal_random_directions(
        vectors["warmth"],
        vectors["competence"],
        count=args.n_random_directions,
        seed=cfg.probing.seed,
    )
    named_vectors = {
        **vectors,
        **{f"random_{index:03d}": vector for index, vector in enumerate(randoms)},
    }
    direction_sds = {
        name: directional_sd(train_matrix, vector)
        for name, vector in named_vectors.items()
    }

    first_text = buckets["high_warmth"][0]["text"]
    started = time.time()
    model, base, language, tokenizer, n_layers, d_model, layer, counters = (
        _load_model_and_verify(cfg, first_text)
    )
    rendered_check = render_native_chat(
        tokenizer, judgement_prompt(first_text, "warmth")
    )
    yes_id, no_id = continuation_ids(tokenizer, rendered_check)
    language_layers = language.layers
    raw_rows: list[dict[str, Any]] = []

    def margin(prompt: str, hook=None) -> float:
        rendered = render_native_chat(tokenizer, prompt)
        encoded = encode_prompt(tokenizer, rendered)
        handle = language_layers[layer].register_forward_hook(hook) if hook else None
        try:
            with torch.inference_mode():
                hidden = base(
                    **encoded, use_cache=False, return_dict=True
                ).last_hidden_state
                logits = model.lm_head(hidden[:, -1, :]).float()
        finally:
            if handle is not None:
                handle.remove()
        return float((logits[0, yes_id] - logits[0, no_id]).item())

    for axis in AXES:
        records = [
            (record, label_value)
            for condition, label_value in ((f"high_{axis}", 1), (f"low_{axis}", 0))
            for record in buckets[condition]
            if int(record["topic_idx"]) in set(test_topics.tolist())
        ]
        baselines = {
            record["id"]: margin(judgement_prompt(record["text"], axis))
            for record, _ in records
        }
        for record, label_value in records:
            raw_rows.append(
                {
                    "mode": "baseline",
                    "axis": axis,
                    "story_id": record["id"],
                    "topic_idx": int(record["topic_idx"]),
                    "condition": record["condition"],
                    "label": label_value,
                    "direction": "baseline",
                    "strength": 0.0,
                    "margin": baselines[record["id"]],
                    "delta_margin": 0.0,
                    "judgment_axis": axis,
                    "steering_axis": "",
                    "direction_type": "baseline",
                    "random_id": "",
                    "vector_kind": "raw",
                    "intervention": "baseline",
                    "control_scale": "sd_matched",
                    "direction_sd": "",
                    "alpha_absolute": 0.0,
                    "standardized_shift": 0.0,
                    "max_relative_norm_drift": 0.0,
                    "decision_flipped": False,
                }
            )
        for intervention in ("additive", "norm_preserving"):
            for direction_name, vector in named_vectors.items():
                for strength in STRENGTHS:
                    alpha = calibrated_alpha(
                        strength=strength,
                        mean_residual_norm=mean_residual_norm,
                        target_direction_sd=direction_sds[axis],
                        direction_sd=direction_sds[direction_name],
                        control_scale="sd_matched",
                    )
                    diagnostics = {"max": 0.0}

                    def hook(_module, _inputs, output, *, _alpha=alpha, _vector=vector):
                        residual = output[0] if isinstance(output, tuple) else output
                        original = residual.float().norm(dim=-1, keepdim=True)
                        changed = intervene_tensor(
                            residual,
                            torch.from_numpy(unit(_vector).astype(np.float32)),
                            _alpha,
                            intervention,
                        )
                        final = changed.float().norm(dim=-1, keepdim=True)
                        drift = torch.where(
                            original > 0,
                            (final - original).abs() / original,
                            torch.zeros_like(original),
                        )
                        diagnostics["max"] = max(
                            diagnostics["max"], float(drift.max().item())
                        )
                        if isinstance(output, tuple):
                            return (changed, *output[1:])
                        return changed

                    for record, label_value in records:
                        value = (
                            baselines[record["id"]]
                            if strength == 0
                            else margin(judgement_prompt(record["text"], axis), hook)
                        )
                        steering_axis, direction_type, random_id = direction_metadata(
                            axis, direction_name
                        )
                        raw_rows.append(
                            {
                                "mode": "steering",
                                "axis": axis,
                                "story_id": record["id"],
                                "topic_idx": int(record["topic_idx"]),
                                "condition": record["condition"],
                                "label": label_value,
                                "direction": direction_name,
                                "strength": strength,
                                "margin": value,
                                "delta_margin": value - baselines[record["id"]],
                                "judgment_axis": axis,
                                "steering_axis": steering_axis,
                                "direction_type": direction_type,
                                "random_id": random_id,
                                "vector_kind": "raw",
                                "intervention": intervention,
                                "control_scale": "sd_matched",
                                "direction_sd": direction_sds[direction_name],
                                "alpha_absolute": alpha,
                                "standardized_shift": standardized_shift(
                                    alpha, direction_sds[direction_name]
                                ),
                                "max_relative_norm_drift": diagnostics["max"],
                                "decision_flipped": bool(
                                    np.signbit(value)
                                    != np.signbit(baselines[record["id"]])
                                ),
                            }
                        )
                    print(
                        f"[steering] {axis} {direction_name} {intervention} {strength:+.2f}",
                        flush=True,
                    )

    summary_rows = summarize(raw_rows, cfg.probing.seed)
    null_rows = null_summary(summary_rows, raw_rows, cfg.probing.seed)
    runtime = _finish_runtime(cfg, model, base, language, tokenizer, counters, started)
    _write_csv(raw_path, raw_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(null_path, null_rows)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "label": label,
                "model": require_model_name(cfg),
                "revision": cfg.model.revision,
                "probe_layer": layer,
                "seed": cfg.probing.seed,
                "stimuli_sha256": stimuli_hash,
                "train_topics": train_topics.tolist(),
                "test_topics": test_topics.tolist(),
                "mean_residual_norm": mean_residual_norm,
                "direction_sds_train_topics": direction_sds,
                "control_scale": "sd_matched",
                "interventions": ["additive", "norm_preserving"],
                "n_random_directions": args.n_random_directions,
                "strengths": list(STRENGTHS),
                "yes_token_id": yes_id,
                "no_token_id": no_id,
                "transformer_lens_imported": "transformer_lens" in sys.modules,
                "scientific_gate": "descriptive-only",
                "runtime": runtime,
                "outputs": [str(raw_path), str(summary_path), str(null_path)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-random-directions", type=int, default=99)
    return parser.parse_args()


if __name__ == "__main__":
    main()
