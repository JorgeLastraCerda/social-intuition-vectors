"""Native-Hugging-Face production stages for Qwen3.6 concept probing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.layer_sweep import sweep_metrics_at_layer
from src.qwen36_smoke import CONDITIONS, encode_raw_passage, mean_pool_after_token
from src.utils.config import ProjectConfig, load_config, require_model_name
from src.utils.hooks import layer_from_fraction
from src.validate_probes import probe_axis, projected_cv_accuracy


@dataclass(frozen=True)
class StagePaths:
    vectors_dir: Path
    probe_table: Path
    probe_log: Path
    sweep_table: Path
    sweep_meta: Path
    technical_logs: dict[int, Path]

    def outputs(self, stage: int) -> tuple[Path, ...]:
        if stage == 1:
            return self.vectors_dir, self.technical_logs[1]
        if stage == 2:
            return self.probe_table, self.probe_log, self.technical_logs[2]
        if stage == 3:
            return self.sweep_table, self.sweep_meta, self.technical_logs[3]
        raise ValueError(f"stage must be 1, 2, or 3; got {stage}")


def stage_paths(cfg: ProjectConfig) -> StagePaths:
    label = cfg.native_hf.label
    tables = Path(cfg.paths.results) / "tables"
    logs = Path(cfg.paths.logs)
    sweep = tables / f"layer_sweep_{label}.csv"
    return StagePaths(
        vectors_dir=Path(cfg.paths.processed) / f"concept_vectors_{label}",
        probe_table=tables / f"probe_metrics_{label}.csv",
        probe_log=logs / f"validate_probes_{label}.json",
        sweep_table=sweep,
        sweep_meta=sweep.with_suffix(".meta.json"),
        technical_logs={stage: logs / f"{label}_stage{stage}.json" for stage in (1, 2, 3)},
    )


def require_outputs_absent(paths: StagePaths, stage: int) -> None:
    collisions = [str(path) for path in paths.outputs(stage) if path.exists()]
    if collisions:
        raise FileExistsError(
            f"Refusing to overwrite Qwen3.6 Stage {stage} outputs: "
            + ", ".join(collisions)
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_full_records(
    stimuli_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    buckets: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    with stimuli_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            condition = record.get("condition")
            if condition not in buckets:
                raise ValueError(f"Unknown condition {condition!r} in {stimuli_path}.")
            buckets[condition].append(record)
    counts = {condition: len(records) for condition, records in buckets.items()}
    if counts != {condition: 50 for condition in CONDITIONS}:
        raise ValueError(f"Expected 50 stories per condition; got {counts}.")
    for condition, records in buckets.items():
        topics = [int(record["topic_idx"]) for record in records]
        if len(set(topics)) != len(topics):
            raise ValueError(f"Duplicate topic within condition {condition!r}.")
    return buckets, _sha256(stimuli_path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _validate_native_config(cfg: ProjectConfig, *, require_cuda: bool) -> None:
    if cfg.model.backend != "huggingface-native":
        raise ValueError("Qwen3.6 production requires model.backend='huggingface-native'.")
    if not cfg.model.revision:
        raise ValueError("Qwen3.6 production requires a pinned model.revision.")
    if cfg.model.dtype != "bfloat16" or cfg.model.device != "cuda":
        raise ValueError("Qwen3.6 production requires bfloat16 and model.device='cuda'.")
    if not cfg.native_hf.label.startswith("qwen36_"):
        raise ValueError("native_hf.label must start with 'qwen36_'.")
    if "transformer_lens" in sys.modules:
        raise RuntimeError("TransformerLens was imported in a native-HF process.")
    if require_cuda and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("A Qwen3.6 GPU stage requires exactly one visible CUDA GPU.")


def _load_model_and_verify(
    cfg: ProjectConfig,
    first_text: str,
) -> tuple[Any, Any, Any, Any, int, int, int, dict[str, float | int]]:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    torch.manual_seed(cfg.probing.seed)
    np.random.seed(cfg.probing.seed)
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(
        require_model_name(cfg), revision=cfg.model.revision
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        require_model_name(cfg),
        revision=cfg.model.revision,
        dtype=getattr(torch, cfg.model.dtype),
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
    if (n_layers, d_model) != (
        cfg.native_hf.expected_layers,
        cfg.native_hf.expected_d_model,
    ):
        raise AssertionError(
            f"Unexpected architecture {(n_layers, d_model)}; expected "
            f"{(cfg.native_hf.expected_layers, cfg.native_hf.expected_d_model)}."
        )
    probe_layer = layer_from_fraction(n_layers, cfg.probing.probe_layer_frac)
    base_model = model.model
    language_model = base_model.language_model
    layers = language_model.layers
    if len(layers) != n_layers:
        raise AssertionError(f"Language layer count mismatch: {len(layers)} != {n_layers}.")

    counters: dict[str, float | int] = {"vision_forward_calls": 0}

    def count_vision_calls(_module, _inputs):
        counters["vision_forward_calls"] = int(counters["vision_forward_calls"]) + 1

    vision_handle = base_model.visual.register_forward_pre_hook(count_vision_calls)
    encoded = encode_raw_passage(
        tokenizer, first_text, bos_token_id=int(text_cfg.bos_token_id)
    )
    encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
    captured: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        baseline = base_model(
            **encoded, output_hidden_states=True, use_cache=False, return_dict=True
        )
        baseline_logits = model.lm_head(baseline.last_hidden_state[:, -1:, :])

        def capture(_module, _inputs, output):
            captured["activation"] = (
                output[0] if isinstance(output, tuple) else output
            ).detach()

        hook_handle = layers[probe_layer].register_forward_hook(capture)
        hooked = base_model(
            **encoded, output_hidden_states=True, use_cache=False, return_dict=True
        )
        hook_handle.remove()
        hooked_logits = model.lm_head(hooked.last_hidden_state[:, -1:, :])
    activation = captured.get("activation")
    if activation is None:
        raise AssertionError("Probe-layer hook did not fire.")
    counters["hook_hidden_max_diff"] = float(
        (activation.float() - hooked.hidden_states[probe_layer + 1].float()).abs().max()
    )
    counters["passive_hook_max_logit_diff"] = float(
        (baseline_logits.float() - hooked_logits.float()).abs().max()
    )
    if float(counters["hook_hidden_max_diff"]) > 1e-5:
        raise AssertionError(f"Hook/hidden mismatch: {counters['hook_hidden_max_diff']}.")
    if float(counters["passive_hook_max_logit_diff"]) > 1e-5:
        raise AssertionError(f"Passive hook changed logits: {counters['passive_hook_max_logit_diff']}.")
    del baseline, hooked, baseline_logits, hooked_logits
    captured.clear()
    counters["_vision_handle"] = vision_handle  # type: ignore[assignment]
    return model, base_model, language_model, tokenizer, n_layers, d_model, probe_layer, counters


def _finish_runtime(
    cfg: ProjectConfig,
    model: Any,
    base_model: Any,
    language_model: Any,
    processor_or_tokenizer: Any,
    counters: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    handle = counters.pop("_vision_handle")
    handle.remove()
    if int(counters["vision_forward_calls"]) != 0:
        raise AssertionError(
            f"Vision encoder was called {counters['vision_forward_calls']} times."
        )
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    total_vram = torch.cuda.get_device_properties(0).total_memory
    peak_fraction = peak_reserved / total_vram
    if peak_fraction > cfg.native_hf.max_vram_fraction:
        raise MemoryError(
            f"Peak reserved VRAM fraction {peak_fraction:.4f} exceeds "
            f"{cfg.native_hf.max_vram_fraction:.4f}."
        )
    return {
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
        "tokenizer_class": processor_or_tokenizer.__class__.__name__,
        "model_revision_requested": cfg.model.revision,
        "model_revision_resolved": getattr(model.config, "_commit_hash", None),
        "dtype": str(next(model.parameters()).dtype),
        "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_total_vram_gib": round(total_vram / 1024**3, 6),
        "peak_allocated_vram_gib": round(peak_allocated / 1024**3, 6),
        "peak_reserved_vram_gib": round(peak_reserved / 1024**3, 6),
        "peak_reserved_vram_fraction": round(peak_fraction, 8),
        "elapsed_seconds": round(time.time() - started, 3),
        **{key: value for key, value in counters.items() if not key.startswith("_")},
    }


def _base_meta(
    cfg: ProjectConfig,
    *,
    n_layers: int,
    d_model: int,
    probe_layer: int,
    stimuli_sha256: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": require_model_name(cfg),
        "revision": cfg.model.revision,
        "probe_layer": probe_layer,
        "probe_layer_frac": cfg.probing.probe_layer_frac,
        "n_layers": n_layers,
        "d_model": d_model,
        "start_token": cfg.probing.start_token,
        "seed": cfg.probing.seed,
        "n_stories": 200,
        "input_format": "raw-passage-explicit-bos",
        "stimuli_sha256": stimuli_sha256,
        "runtime": runtime,
    }


def run_stage1(cfg: ProjectConfig, paths: StagePaths) -> None:
    _validate_native_config(cfg, require_cuda=True)
    require_outputs_absent(paths, 1)
    stimuli = Path(cfg.paths.stimuli) / "concept_stories.jsonl"
    buckets, stimuli_sha256 = load_full_records(stimuli)
    started = time.time()
    model, base, language, tokenizer, n_layers, d_model, probe_layer, counters = (
        _load_model_and_verify(cfg, buckets[CONDITIONS[0]][0]["text"])
    )
    text_cfg = model.config.text_config
    layers = language.layers
    token_lengths: list[int] = []
    X: dict[str, np.ndarray] = {}
    captured: dict[str, torch.Tensor] = {}

    def capture_probe(_module, _inputs, output):
        captured["activation"] = output[0] if isinstance(output, tuple) else output

    hook_handle = layers[probe_layer].register_forward_hook(capture_probe)
    with torch.inference_mode():
        for condition in CONDITIONS:
            rows = buckets[condition]
            matrix = np.empty((len(rows), d_model), dtype=np.float32)
            for index, record in enumerate(rows):
                encoded = encode_raw_passage(
                    tokenizer, record["text"], bos_token_id=int(text_cfg.bos_token_id)
                )
                seq_len = int(encoded["input_ids"].shape[1])
                if seq_len <= cfg.probing.start_token:
                    raise ValueError(
                        f"topic={record['topic_idx']} condition={condition} "
                        f"has seq_len={seq_len} <= start_token={cfg.probing.start_token}."
                    )
                token_lengths.append(seq_len)
                encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
                captured.clear()
                base(**encoded, output_hidden_states=False, use_cache=False, return_dict=True)
                activation = captured.get("activation")
                if activation is None:
                    raise AssertionError("Stage 1 hook did not fire.")
                matrix[index] = mean_pool_after_token(
                    activation, cfg.probing.start_token
                ).float().cpu().numpy()
            if not np.isfinite(matrix).all():
                raise FloatingPointError(f"Non-finite Stage 1 matrix for {condition}.")
            X[condition] = matrix
            print(f"[stage1] {condition}: {len(rows)} stories", flush=True)
    hook_handle.remove()
    runtime = _finish_runtime(cfg, model, base, language, tokenizer, counters, started)
    warmth = X["high_warmth"].mean(0) - X["low_warmth"].mean(0)
    competence = X["high_competence"].mean(0) - X["low_competence"].mean(0)
    if not np.isfinite(warmth).all() or not np.isfinite(competence).all():
        raise FloatingPointError("Non-finite concept vector.")
    if min(np.linalg.norm(warmth), np.linalg.norm(competence)) <= 0:
        raise AssertionError("Concept-vector norms must be positive.")
    meta = _base_meta(
        cfg,
        n_layers=n_layers,
        d_model=d_model,
        probe_layer=probe_layer,
        stimuli_sha256=stimuli_sha256,
        runtime=runtime,
    )
    meta.update(
        {
            "n_per_condition": {condition: len(X[condition]) for condition in CONDITIONS},
            "warmth_vec_norm": round(float(np.linalg.norm(warmth)), 6),
            "competence_vec_norm": round(float(np.linalg.norm(competence)), 6),
            "token_length_min": min(token_lengths),
            "token_length_max": max(token_lengths),
        }
    )
    paths.vectors_dir.mkdir(parents=True)
    for condition, matrix in X.items():
        np.save(paths.vectors_dir / f"X_{condition}.npy", matrix)
    np.save(paths.vectors_dir / "warmth_vec.npy", warmth)
    np.save(paths.vectors_dir / "competence_vec.npy", competence)
    _write_json(paths.vectors_dir / "meta.json", meta)
    _write_json(
        paths.technical_logs[1],
        {"status": "pass", "stage": 1, "label": cfg.native_hf.label, **meta},
    )


def run_stage2(cfg: ProjectConfig, paths: StagePaths) -> None:
    _validate_native_config(cfg, require_cuda=False)
    require_outputs_absent(paths, 2)
    stimuli = Path(cfg.paths.stimuli) / "concept_stories.jsonl"
    buckets, stimuli_sha256 = load_full_records(stimuli)
    meta = json.loads((paths.vectors_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("stimuli_sha256") != stimuli_sha256:
        raise AssertionError("Stage 1 stimulus hash does not match the current corpus.")
    X = {
        condition: np.load(paths.vectors_dir / f"X_{condition}.npy")
        for condition in CONDITIONS
    }
    warmth = np.load(paths.vectors_dir / "warmth_vec.npy")
    competence = np.load(paths.vectors_dir / "competence_vec.npy")
    groups = {
        condition: np.asarray(
            [int(record["topic_idx"]) for record in buckets[condition]], dtype=np.int64
        )
        for condition in CONDITIONS
    }
    started = time.time()
    warmth_metrics = probe_axis(
        X["high_warmth"], X["low_warmth"], warmth, "warmth", cfg.probing.seed,
        groups["high_warmth"], groups["low_warmth"],
    )
    competence_metrics = probe_axis(
        X["high_competence"], X["low_competence"], competence, "competence",
        cfg.probing.seed, groups["high_competence"], groups["low_competence"],
    )
    w_unit = warmth / (np.linalg.norm(warmth) + 1e-12)
    c_unit = competence / (np.linalg.norm(competence) + 1e-12)
    cosine = float(np.dot(w_unit, c_unit))
    cross_w_on_c = projected_cv_accuracy(
        X["high_competence"], X["low_competence"], warmth, cfg.probing.seed
    )
    cross_c_on_w = projected_cv_accuracy(
        X["high_warmth"], X["low_warmth"], competence, cfg.probing.seed
    )
    _write_csv(paths.probe_table, [warmth_metrics, competence_metrics])
    log = {
        "meta": meta,
        "warmth": warmth_metrics,
        "competence": competence_metrics,
        "axis_cosine": round(cosine, 6),
        "cross_warmth_on_competence_cv": round(cross_w_on_c, 6),
        "cross_competence_on_warmth_cv": round(cross_c_on_w, 6),
        "pass_warmth_cv": warmth_metrics["cv_mean"] > 0.8,
        "pass_competence_cv": competence_metrics["cv_mean"] > 0.8,
        "pass_orthogonality": abs(cosine) < 0.3,
        "pass_warmth_topic_cv": warmth_metrics["topic_cv_mean"] > 0.8,
        "pass_competence_topic_cv": competence_metrics["topic_cv_mean"] > 0.8,
        "scientific_flags_are_non_gating": True,
    }
    _write_json(paths.probe_log, log)
    _write_json(
        paths.technical_logs[2],
        {
            "status": "pass",
            "stage": 2,
            "label": cfg.native_hf.label,
            "model": cfg.model.name,
            "revision": cfg.model.revision,
            "seed": cfg.probing.seed,
            "stimuli_sha256": stimuli_sha256,
            "backend": "numpy-scikit-learn-cpu",
            "elapsed_seconds": round(time.time() - started, 3),
        },
    )


def run_stage3(cfg: ProjectConfig, paths: StagePaths) -> None:
    _validate_native_config(cfg, require_cuda=True)
    require_outputs_absent(paths, 3)
    stimuli = Path(cfg.paths.stimuli) / "concept_stories.jsonl"
    buckets, stimuli_sha256 = load_full_records(stimuli)
    started = time.time()
    model, base, language, tokenizer, n_layers, d_model, probe_layer, counters = (
        _load_model_and_verify(cfg, buckets[CONDITIONS[0]][0]["text"])
    )
    text_cfg = model.config.text_config
    all_records = [record for condition in CONDITIONS for record in buckets[condition]]
    story_indices: dict[str, list[int]] = {}
    topic_groups: dict[str, np.ndarray] = {}
    offset = 0
    for condition in CONDITIONS:
        rows = buckets[condition]
        story_indices[condition] = list(range(offset, offset + len(rows)))
        topic_groups[condition] = np.asarray(
            [int(record["topic_idx"]) for record in rows], dtype=np.int64
        )
        offset += len(rows)
    acts = np.empty((n_layers, len(all_records), d_model), dtype=np.float32)
    token_lengths: list[int] = []
    with torch.inference_mode():
        for story_index, record in enumerate(all_records):
            encoded = encode_raw_passage(
                tokenizer, record["text"], bos_token_id=int(text_cfg.bos_token_id)
            )
            seq_len = int(encoded["input_ids"].shape[1])
            if seq_len <= cfg.probing.start_token:
                raise ValueError(
                    f"story={story_index} has seq_len={seq_len} <= "
                    f"start_token={cfg.probing.start_token}."
                )
            token_lengths.append(seq_len)
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            outputs = base(
                **encoded, output_hidden_states=True, use_cache=False, return_dict=True
            )
            if len(outputs.hidden_states) != n_layers + 1:
                raise AssertionError(
                    f"Expected {n_layers + 1} hidden states; got {len(outputs.hidden_states)}."
                )
            for layer_index in range(n_layers):
                acts[layer_index, story_index] = mean_pool_after_token(
                    outputs.hidden_states[layer_index + 1], cfg.probing.start_token
                ).float().cpu().numpy()
            if (story_index + 1) % 20 == 0:
                print(f"[stage3] {story_index + 1}/200 stories", flush=True)
    if not np.isfinite(acts).all():
        raise FloatingPointError("Stage 3 activation buffer contains NaN or Inf.")
    runtime = _finish_runtime(cfg, model, base, language, tokenizer, counters, started)
    rows = [
        sweep_metrics_at_layer(
            layer, acts, story_indices, topic_groups, n_layers, probe_layer
        )
        for layer in range(n_layers)
    ]
    _write_csv(paths.sweep_table, rows)
    meta = _base_meta(
        cfg,
        n_layers=n_layers,
        d_model=d_model,
        probe_layer=probe_layer,
        stimuli_sha256=stimuli_sha256,
        runtime=runtime,
    )
    meta.update(
        {
            "label": cfg.native_hf.label,
            "token_length_min": min(token_lengths),
            "token_length_max": max(token_lengths),
        }
    )
    _write_json(paths.sweep_meta, meta)
    _write_json(
        paths.technical_logs[3],
        {"status": "pass", "stage": 3, "label": cfg.native_hf.label, **meta},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = stage_paths(cfg)
    if args.dry_run:
        buckets, stimuli_sha256 = load_full_records(
            Path(cfg.paths.stimuli) / "concept_stories.jsonl"
        )
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "model": cfg.model.name,
                    "revision": cfg.model.revision,
                    "label": cfg.native_hf.label,
                    "n_per_condition": {key: len(value) for key, value in buckets.items()},
                    "stimuli_sha256": stimuli_sha256,
                    "outputs": [str(path) for path in paths.outputs(args.stage)],
                },
                indent=2,
            )
        )
        return
    {1: run_stage1, 2: run_stage2, 3: run_stage3}[args.stage](cfg, paths)


if __name__ == "__main__":
    main()
