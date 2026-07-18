"""Native-Hugging-Face Stage 1--3 smoke test for Qwen3.6.

The smoke loads one BF16 model, runs a deterministic topic-balanced subset once
through every language layer, and derives fixed-layer vectors, probe metrics,
and an all-layer sweep from the same activation buffer. TransformerLens and
nnsight are deliberately not used.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.layer_sweep import sweep_metrics_at_layer
from src.utils.config import ProjectConfig, load_config, require_model_name
from src.utils.hooks import layer_from_fraction
from src.validate_probes import probe_axis, projected_cv_accuracy


CONDITIONS = (
    "high_warmth",
    "low_warmth",
    "high_competence",
    "low_competence",
)


@dataclass(frozen=True)
class SmokePaths:
    vectors_dir: Path
    probe_table: Path
    probe_log: Path
    sweep_table: Path
    sweep_meta: Path
    technical_log: Path

    def all(self) -> tuple[Path, ...]:
        return (
            self.vectors_dir,
            self.probe_table,
            self.probe_log,
            self.sweep_table,
            self.sweep_meta,
            self.technical_log,
        )


def smoke_paths(cfg: ProjectConfig, label: str | None = None) -> SmokePaths:
    resolved = label or cfg.smoke.label
    tables = Path(cfg.paths.results) / "tables"
    logs = Path(cfg.paths.logs)
    sweep = tables / f"layer_sweep_{resolved}.csv"
    return SmokePaths(
        vectors_dir=Path(cfg.paths.processed) / f"concept_vectors_{resolved}",
        probe_table=tables / f"probe_metrics_{resolved}.csv",
        probe_log=logs / f"validate_probes_{resolved}.json",
        sweep_table=sweep,
        sweep_meta=sweep.with_suffix(".meta.json"),
        technical_log=logs / "smoke_qwen36_27b.json",
    )


def require_outputs_absent(paths: SmokePaths) -> None:
    collisions = [str(path) for path in paths.all() if path.exists()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing Qwen3.6 smoke outputs: "
            + ", ".join(collisions)
        )


def select_topic_records(
    stimuli_path: Path,
    *,
    n_topics: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[int]]:
    """Select complete topics reproducibly and keep rows aligned by topic."""

    by_topic: dict[int, dict[str, dict[str, Any]]] = {}
    with stimuli_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            condition = record.get("condition")
            if condition not in CONDITIONS:
                raise ValueError(f"Unknown condition {condition!r} in {stimuli_path}.")
            topic = int(record["topic_idx"])
            topic_bucket = by_topic.setdefault(topic, {})
            if condition in topic_bucket:
                raise ValueError(
                    f"Duplicate topic/condition pair: topic={topic}, condition={condition}."
                )
            topic_bucket[condition] = record

    complete = sorted(
        topic for topic, rows in by_topic.items() if set(rows) == set(CONDITIONS)
    )
    if len(complete) < n_topics:
        raise ValueError(
            f"Requested {n_topics} complete topics, but only {len(complete)} exist."
        )
    selected = sorted(random.Random(seed).sample(complete, n_topics))
    buckets = {
        condition: [by_topic[topic][condition] for topic in selected]
        for condition in CONDITIONS
    }
    return buckets, selected


def encode_raw_passage(tokenizer, text: str, *, bos_token_id: int) -> dict[str, torch.Tensor]:
    """Encode raw text with exactly one explicit BOS, matching prepend_bos=True."""

    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Unexpected tokenizer shape: {tuple(input_ids.shape)}")
    bos = torch.tensor([[bos_token_id]], dtype=input_ids.dtype)
    input_ids = torch.cat((bos, input_ids), dim=1)
    attention_mask = torch.ones_like(input_ids)
    if input_ids.shape[1] > 1 and int(input_ids[0, 1]) == bos_token_id:
        raise AssertionError("Tokenizer produced a BOS despite add_special_tokens=False.")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def mean_pool_after_token(hidden: torch.Tensor, start_token: int) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(f"Expected [1, seq, d_model], got {tuple(hidden.shape)}")
    if hidden.shape[1] <= start_token:
        raise ValueError(
            f"Sequence length {hidden.shape[1]} does not exceed start_token={start_token}."
        )
    return hidden[:, start_token:, :].mean(dim=1).squeeze(0)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run_smoke(cfg: ProjectConfig, paths: SmokePaths) -> None:
    if cfg.model.backend != "huggingface-native":
        raise ValueError(
            "Qwen3.6 smoke requires model.backend='huggingface-native'; "
            f"got {cfg.model.backend!r}."
        )
    if not cfg.model.revision:
        raise ValueError("Qwen3.6 smoke requires a pinned model.revision.")
    if cfg.model.dtype != "bfloat16" or cfg.model.device != "cuda":
        raise ValueError("Qwen3.6 smoke requires bfloat16 on cuda.")
    if "transformer_lens" in sys.modules:
        raise RuntimeError("TransformerLens was imported in the native-HF smoke process.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Qwen3.6 smoke requires exactly one visible CUDA GPU.")

    require_outputs_absent(paths)
    stimuli_path = Path(cfg.paths.stimuli) / "concept_stories.jsonl"
    buckets, selected_topics = select_topic_records(
        stimuli_path,
        n_topics=cfg.smoke.n_topics,
        seed=cfg.probing.seed,
    )

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    dtype = getattr(torch, cfg.model.dtype)
    model_name = require_model_name(cfg)
    torch.manual_seed(cfg.probing.seed)
    np.random.seed(cfg.probing.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    processor = AutoProcessor.from_pretrained(
        model_name,
        revision=cfg.model.revision,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        revision=cfg.model.revision,
        dtype=dtype,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("AutoProcessor did not expose a tokenizer.")

    text_cfg = model.config.text_config
    n_layers = int(text_cfg.num_hidden_layers)
    d_model = int(text_cfg.hidden_size)
    if n_layers != cfg.smoke.expected_layers or d_model != cfg.smoke.expected_d_model:
        raise AssertionError(
            "Unexpected Qwen3.6 dimensions: "
            f"layers={n_layers}, d_model={d_model}; expected "
            f"{cfg.smoke.expected_layers}/{cfg.smoke.expected_d_model}."
        )
    probe_layer = layer_from_fraction(n_layers, cfg.probing.probe_layer_frac)
    expected_probe_layer = layer_from_fraction(
        cfg.smoke.expected_layers, cfg.probing.probe_layer_frac
    )
    if probe_layer != expected_probe_layer:
        raise AssertionError((probe_layer, expected_probe_layer))

    base_model = model.model
    language_model = base_model.language_model
    layers = language_model.layers
    if len(layers) != n_layers:
        raise AssertionError(f"Language layer count mismatch: {len(layers)} != {n_layers}")
    hook_module_path = f"model.language_model.layers.{probe_layer}"

    vision_calls = 0

    def count_vision_calls(_module, _inputs):
        nonlocal vision_calls
        vision_calls += 1

    vision_handle = base_model.visual.register_forward_pre_hook(count_vision_calls)

    first_text = buckets[CONDITIONS[0]][0]["text"]
    first_inputs = encode_raw_passage(
        tokenizer, first_text, bos_token_id=int(text_cfg.bos_token_id)
    )
    first_inputs = {key: value.to("cuda:0") for key, value in first_inputs.items()}
    captured: dict[str, torch.Tensor] = {}

    with torch.inference_mode():
        baseline = base_model(
            **first_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        baseline_last_logits = model.lm_head(baseline.last_hidden_state[:, -1:, :])

        def capture_probe(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured["activation"] = tensor.detach()

        hook_handle = layers[probe_layer].register_forward_hook(capture_probe)
        hooked = base_model(
            **first_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hook_handle.remove()
        hooked_last_logits = model.lm_head(hooked.last_hidden_state[:, -1:, :])

    if len(hooked.hidden_states) != n_layers + 1:
        raise AssertionError(
            f"Expected {n_layers + 1} hidden states, got {len(hooked.hidden_states)}."
        )
    hook_activation = captured.get("activation")
    if hook_activation is None:
        raise AssertionError("Probe-layer forward hook did not fire.")
    hidden_activation = hooked.hidden_states[probe_layer + 1]
    hook_hidden_max_diff = float(
        (hook_activation.float() - hidden_activation.float()).abs().max()
    )
    passive_hook_max_logit_diff = float(
        (baseline_last_logits.float() - hooked_last_logits.float()).abs().max()
    )
    if hook_hidden_max_diff > 1e-5:
        raise AssertionError(
            f"Hook/hidden-state mismatch: max diff {hook_hidden_max_diff:.8g}."
        )
    if passive_hook_max_logit_diff > 1e-5:
        raise AssertionError(
            f"Passive hook changed logits: max diff {passive_hook_max_logit_diff:.8g}."
        )
    del baseline, hooked, baseline_last_logits, hooked_last_logits
    captured.clear()

    n_stories = cfg.smoke.n_topics * len(CONDITIONS)
    acts = np.empty((n_layers, n_stories, d_model), dtype=np.float32)
    story_indices: dict[str, list[int]] = {}
    topic_groups: dict[str, np.ndarray] = {}
    token_lengths: list[int] = []
    offset = 0

    with torch.inference_mode():
        for condition in CONDITIONS:
            condition_rows = buckets[condition]
            story_indices[condition] = list(range(offset, offset + len(condition_rows)))
            topic_groups[condition] = np.asarray(
                [int(record["topic_idx"]) for record in condition_rows], dtype=np.int64
            )
            for local_index, record in enumerate(condition_rows):
                encoded = encode_raw_passage(
                    tokenizer,
                    record["text"],
                    bos_token_id=int(text_cfg.bos_token_id),
                )
                seq_len = int(encoded["input_ids"].shape[1])
                if seq_len <= cfg.probing.start_token:
                    raise ValueError(
                        f"topic={record['topic_idx']} condition={condition} "
                        f"has seq_len={seq_len} <= start_token={cfg.probing.start_token}."
                    )
                token_lengths.append(seq_len)
                encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
                outputs = base_model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                if len(outputs.hidden_states) != n_layers + 1:
                    raise AssertionError(
                        f"Expected {n_layers + 1} hidden states, "
                        f"got {len(outputs.hidden_states)}."
                    )
                global_index = offset + local_index
                for layer_index in range(n_layers):
                    pooled = mean_pool_after_token(
                        outputs.hidden_states[layer_index + 1],
                        cfg.probing.start_token,
                    )
                    acts[layer_index, global_index] = pooled.float().cpu().numpy()
                if not np.isfinite(acts[:, global_index]).all():
                    raise FloatingPointError(
                        f"Non-finite activation at story index {global_index}."
                    )
            offset += len(condition_rows)
            print(f"[extract] {condition}: {len(condition_rows)} stories", flush=True)

    vision_handle.remove()
    if vision_calls != 0:
        raise AssertionError(f"Vision encoder was called {vision_calls} times for text-only input.")

    X = {
        condition: acts[probe_layer, story_indices[condition], :]
        for condition in CONDITIONS
    }
    warmth_vec = X["high_warmth"].mean(axis=0) - X["low_warmth"].mean(axis=0)
    competence_vec = (
        X["high_competence"].mean(axis=0) - X["low_competence"].mean(axis=0)
    )
    if not np.isfinite(warmth_vec).all() or not np.isfinite(competence_vec).all():
        raise FloatingPointError("Non-finite Stage 1 concept vector.")
    if np.linalg.norm(warmth_vec) <= 0 or np.linalg.norm(competence_vec) <= 0:
        raise AssertionError("Stage 1 concept-vector norms must be positive.")

    warmth_metrics = probe_axis(
        X["high_warmth"],
        X["low_warmth"],
        warmth_vec,
        "warmth",
        cfg.probing.seed,
        topic_groups["high_warmth"],
        topic_groups["low_warmth"],
    )
    competence_metrics = probe_axis(
        X["high_competence"],
        X["low_competence"],
        competence_vec,
        "competence",
        cfg.probing.seed,
        topic_groups["high_competence"],
        topic_groups["low_competence"],
    )
    w_unit = warmth_vec / (np.linalg.norm(warmth_vec) + 1e-12)
    c_unit = competence_vec / (np.linalg.norm(competence_vec) + 1e-12)
    axis_cosine = float(np.dot(w_unit, c_unit))
    cross_w_on_c = projected_cv_accuracy(
        X["high_competence"], X["low_competence"], warmth_vec, cfg.probing.seed
    )
    cross_c_on_w = projected_cv_accuracy(
        X["high_warmth"], X["low_warmth"], competence_vec, cfg.probing.seed
    )

    sweep_rows = [
        sweep_metrics_at_layer(
            layer_index,
            acts,
            story_indices,
            topic_groups,
            n_layers,
            probe_layer,
        )
        for layer_index in range(n_layers)
    ]
    probe_row = sweep_rows[probe_layer]
    if abs(float(probe_row["warmth_cohens_d"]) - float(warmth_metrics["cohens_d"])) > 1e-6:
        raise AssertionError("Stage 2/3 warmth Cohen's d mismatch at the probe layer.")
    if abs(float(probe_row["comp_cohens_d"]) - float(competence_metrics["cohens_d"])) > 1e-6:
        raise AssertionError("Stage 2/3 competence Cohen's d mismatch at the probe layer.")
    if abs(float(probe_row["cos_wc"]) - round(axis_cosine, 6)) > 1e-6:
        raise AssertionError("Stage 2/3 axis-cosine mismatch at the probe layer.")

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    total_vram = torch.cuda.get_device_properties(0).total_memory
    peak_fraction = peak_reserved / total_vram
    if peak_fraction > cfg.smoke.max_vram_fraction:
        raise MemoryError(
            f"Peak reserved VRAM fraction {peak_fraction:.4f} exceeds "
            f"limit {cfg.smoke.max_vram_fraction:.4f}."
        )

    resolved_revision = getattr(model.config, "_commit_hash", None)
    runtime = {
        "backend": "huggingface-native",
        "hook_backend": "torch-forward-hook+transformers-hidden-states",
        "transformers_version": _version("transformers"),
        "torch_version": torch.__version__,
        "torchvision_version": _version("torchvision"),
        "accelerate_version": _version("accelerate"),
        "transformer_lens_version": _version("transformer-lens"),
        "transformer_lens_imported": "transformer_lens" in sys.modules,
        "model_class": model.__class__.__name__,
        "base_model_class": base_model.__class__.__name__,
        "language_model_class": language_model.__class__.__name__,
        "processor_class": processor.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_revision_requested": cfg.model.revision,
        "model_revision_resolved": resolved_revision,
        "dtype": str(next(model.parameters()).dtype),
        "parameter_devices": sorted({str(parameter.device) for parameter in model.parameters()}),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_total_vram_gib": round(total_vram / 1024**3, 6),
        "peak_allocated_vram_gib": round(peak_allocated / 1024**3, 6),
        "peak_reserved_vram_gib": round(peak_reserved / 1024**3, 6),
        "peak_reserved_vram_fraction": round(peak_fraction, 8),
        "elapsed_seconds": round(time.time() - started, 3),
    }

    paths.vectors_dir.mkdir(parents=True)
    for condition in CONDITIONS:
        np.save(paths.vectors_dir / f"X_{condition}.npy", X[condition])
    np.save(paths.vectors_dir / "warmth_vec.npy", warmth_vec)
    np.save(paths.vectors_dir / "competence_vec.npy", competence_vec)
    stage1_meta = {
        "model": model_name,
        "revision": cfg.model.revision,
        "probe_layer": probe_layer,
        "probe_layer_frac": cfg.probing.probe_layer_frac,
        "n_layers": n_layers,
        "d_model": d_model,
        "start_token": cfg.probing.start_token,
        "seed": cfg.probing.seed,
        "selected_topics": selected_topics,
        "n_per_condition": {condition: len(X[condition]) for condition in CONDITIONS},
        "warmth_vec_norm": round(float(np.linalg.norm(warmth_vec)), 6),
        "competence_vec_norm": round(float(np.linalg.norm(competence_vec)), 6),
        "input_format": "raw-passage-explicit-bos",
        "smoke": True,
        "runtime": runtime,
    }
    _json_write(paths.vectors_dir / "meta.json", stage1_meta)

    _write_csv(paths.probe_table, [warmth_metrics, competence_metrics])
    probe_log = {
        "meta": stage1_meta,
        "warmth": warmth_metrics,
        "competence": competence_metrics,
        "axis_cosine": round(axis_cosine, 6),
        "cross_warmth_on_competence_cv": round(cross_w_on_c, 6),
        "cross_competence_on_warmth_cv": round(cross_c_on_w, 6),
        "pass_warmth_cv": warmth_metrics["cv_mean"] > 0.8,
        "pass_competence_cv": competence_metrics["cv_mean"] > 0.8,
        "pass_orthogonality": abs(axis_cosine) < 0.3,
        "pass_warmth_topic_cv": warmth_metrics["topic_cv_mean"] > 0.8,
        "pass_competence_topic_cv": competence_metrics["topic_cv_mean"] > 0.8,
        "scientific_flags_are_non_gating": True,
    }
    _json_write(paths.probe_log, probe_log)

    _write_csv(paths.sweep_table, sweep_rows)
    _json_write(
        paths.sweep_meta,
        {
            "model": model_name,
            "revision": cfg.model.revision,
            "n_layers": n_layers,
            "d_model": d_model,
            "probe_layer": probe_layer,
            "probe_layer_frac": cfg.probing.probe_layer_frac,
            "start_token": cfg.probing.start_token,
            "seed": cfg.probing.seed,
            "selected_topics": selected_topics,
            "n_stories": n_stories,
            "label": cfg.smoke.label,
            "input_format": "raw-passage-explicit-bos",
            "smoke": True,
            "runtime": runtime,
        },
    )
    _json_write(
        paths.technical_log,
        {
            "status": "pass",
            "model": model_name,
            "revision": cfg.model.revision,
            "seed": cfg.probing.seed,
            "n_layers": n_layers,
            "d_model": d_model,
            "probe_layer": probe_layer,
            "hook_module_path": hook_module_path,
            "hook_hidden_max_diff": hook_hidden_max_diff,
            "passive_hook_max_logit_diff": passive_hook_max_logit_diff,
            "vision_forward_calls": vision_calls,
            "selected_topics": selected_topics,
            "n_stories": n_stories,
            "token_length_min": min(token_lengths),
            "token_length_max": max(token_lengths),
            "token_lengths": token_lengths,
            "stage2_probe_layer_match_tolerance": 1e-6,
            "runtime": runtime,
            "outputs": [str(path) for path in paths.all()[:-1]],
        },
    )
    print(json.dumps({"status": "pass", "runtime": runtime}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/qwen36_smoke.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = smoke_paths(cfg)
    if args.dry_run:
        buckets, topics = select_topic_records(
            Path(cfg.paths.stimuli) / "concept_stories.jsonl",
            n_topics=cfg.smoke.n_topics,
            seed=cfg.probing.seed,
        )
        print(
            json.dumps(
                {
                    "model": cfg.model.name,
                    "revision": cfg.model.revision,
                    "selected_topics": topics,
                    "n_per_condition": {key: len(value) for key, value in buckets.items()},
                    "outputs": [str(path) for path in paths.all()],
                },
                indent=2,
            )
        )
        return
    run_smoke(cfg, paths)


if __name__ == "__main__":
    main()
