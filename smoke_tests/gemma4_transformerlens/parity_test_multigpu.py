"""Audit Gemma 4 single-GPU versus two-GPU TransformerBridge parity.

The ``run`` command coordinates three fresh Python processes (single A, single B,
and two-device). Each capture writes temporary numeric snapshots outside the Git
repository. The coordinator compares them and writes one compact, tracked JSON
decision artifact even when a capture fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.gemma_scope_causality import make_steering_hook
from src.hiring_steering import hiring_prompt
from src.utils.config import load_config
from src.utils.hooks import mean_activation_after_token, residual_hook_name
from src.utils.model_loader import load_hooked_model, model_runtime_metadata
from src.utils.prompting import decision_token_ids, encode_decision_prompt, encode_passage


CONDITIONS = ("high_warmth", "low_warmth", "high_competence", "low_competence")
AXES = ("warmth", "competence")
STRENGTHS = (-0.05, 0.0, 0.05)
SMOKE_PASSAGE = "A person carefully completed a routine task."
HIRING_NAME = "Jordan Lee"
LOGIT_MAX_ABS_FLOOR = 0.02
NRMSE_FLOOR = 1e-4
COSINE_ERROR_FLOOR = 1e-6
REPEAT_MULTIPLIER = 5.0


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalise_device(value: Any) -> str:
    if isinstance(value, int):
        return f"cuda:{value}"
    text = str(value)
    if text.isdigit():
        return f"cuda:{text}"
    if text == "cuda":
        return "cuda:0"
    return text


def _first_story_per_condition(path: Path) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            condition = str(record["condition"])
            if condition in CONDITIONS and condition not in selected:
                selected[condition] = {
                    "id": str(record["id"]),
                    "text": str(record["text"]),
                }
            if len(selected) == len(CONDITIONS):
                break
    missing = sorted(set(CONDITIONS) - set(selected))
    if missing:
        raise ValueError(f"Missing parity stories for conditions: {missing}")
    return selected


def _language_layers(original_model: Any) -> Any:
    try:
        return original_model.model.language_model.layers
    except AttributeError as exc:
        raise RuntimeError(
            "Gemma 4 language layers not found at model.language_model.layers."
        ) from exc


def inspect_topology(original_model: Any, n_devices: int) -> dict[str, Any]:
    """Return and validate the actual parameter placement of all language layers."""

    layers = _language_layers(original_model)
    layer_devices: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, layer in enumerate(layers):
        devices = {
            _normalise_device(tensor.device)
            for tensor in (*tuple(layer.parameters()), *tuple(layer.buffers()))
            if tensor.numel() > 0
        }
        if len(devices) != 1:
            failures.append(
                f"language layer {index} spans {sorted(devices) or ['no-device']}"
            )
        layer_devices.append({"layer": index, "devices": sorted(devices)})

    all_parameter_devices = {
        _normalise_device(parameter.device)
        for parameter in original_model.parameters()
    }
    forbidden = sorted(all_parameter_devices & {"cpu", "disk", "meta"})
    if forbidden:
        failures.append(f"forbidden parameter targets present: {forbidden}")

    used_layer_devices = sorted(
        {device for row in layer_devices for device in row["devices"]}
    )
    expected = [f"cuda:{index}" for index in range(n_devices)]
    if used_layer_devices != expected:
        failures.append(
            f"language layers use {used_layer_devices}, expected exactly {expected}"
        )

    raw_map = getattr(original_model, "hf_device_map", None)
    device_map = None
    if raw_map:
        device_map = {
            str(key): _normalise_device(value) for key, value in raw_map.items()
        }
    if n_devices > 1 and not device_map:
        failures.append("multi-device capture has no hf_device_map")

    return {
        "requested_n_devices": n_devices,
        "language_layer_count": len(layers),
        "layer_devices": layer_devices,
        "layer_assignment_counts": dict(Counter(
            device for row in layer_devices for device in row["devices"]
        )),
        "parameter_devices": sorted(all_parameter_devices),
        "hf_device_map": device_map,
        "pass": not failures,
        "failures": failures,
    }


def _tensor_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def capture(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    vectors_dir = Path(cfg.paths.processed) / args.vectors_subdir
    vector_meta = json.loads((vectors_dir / "meta.json").read_text(encoding="utf-8"))
    model_name = str(vector_meta["model"])
    cfg = replace(cfg, model=replace(cfg.model, name=model_name))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)

    model = load_hooked_model(cfg, n_devices=args.n_devices)
    model.eval()
    if model.cfg.n_layers != 48 or model.cfg.d_model != 3840:
        raise AssertionError(
            f"Expected Gemma 4 12B geometry (48, 3840), got "
            f"({model.cfg.n_layers}, {model.cfg.d_model})."
        )

    topology = inspect_topology(model.original_model, args.n_devices)
    if not topology["pass"]:
        raise RuntimeError("; ".join(topology["failures"]))

    hook_names = [residual_hook_name(index) for index in range(model.cfg.n_layers)]
    arrays: dict[str, np.ndarray] = {}
    input_records: list[dict[str, Any]] = []

    smoke_tokens = encode_passage(model, SMOKE_PASSAGE)
    with torch.no_grad():
        bridge_logits, smoke_cache = model.run_with_cache(
            smoke_tokens,
            names_filter=hook_names,
        )
        hf_logits = model.original_model(input_ids=smoke_tokens).logits
    arrays["tokens__smoke"] = _tensor_array(smoke_tokens).astype(np.int64)
    arrays["logits__smoke"] = _tensor_array(bridge_logits)
    arrays["hf_logits__smoke"] = _tensor_array(hf_logits)
    for layer, hook_name in enumerate(hook_names):
        arrays[f"activation__smoke__layer_{layer:02d}"] = _tensor_array(
            smoke_cache[hook_name]
        )
    bridge_hf_max_logit_diff = float(
        (bridge_logits.float() - hf_logits.float()).abs().max().item()
    )
    input_records.append(
        {
            "kind": "smoke",
            "id": "smoke-passage",
            "sha256": _sha256_text(SMOKE_PASSAGE),
            "token_count": int(smoke_tokens.shape[1]),
        }
    )
    del smoke_cache, bridge_logits, hf_logits

    stories = _first_story_per_condition(
        Path(cfg.paths.stimuli) / "concept_stories.jsonl"
    )
    for condition in CONDITIONS:
        record = stories[condition]
        tokens = encode_passage(model, record["text"])
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=hook_names,
                return_type=None,
            )
        arrays[f"tokens__story__{condition}"] = _tensor_array(tokens).astype(np.int64)
        for layer, hook_name in enumerate(hook_names):
            pooled = mean_activation_after_token(
                cache[hook_name], cfg.probing.start_token
            ).squeeze(0)
            arrays[f"activation__story__{condition}__layer_{layer:02d}"] = (
                _tensor_array(pooled)
            )
        input_records.append(
            {
                "kind": "story",
                "condition": condition,
                "id": record["id"],
                "sha256": _sha256_text(record["text"]),
                "token_count": int(tokens.shape[1]),
            }
        )
        del cache

    warmth = np.load(vectors_dir / "warmth_vec.npy").astype(np.float32)
    competence = np.load(vectors_dir / "competence_vec.npy").astype(np.float32)
    arrays["vector__warmth"] = warmth
    arrays["vector__competence"] = competence
    all_activations = np.concatenate(
        [
            np.load(vectors_dir / f"X_{condition}.npy").astype(np.float32)
            for condition in CONDITIONS
        ],
        axis=0,
    )
    mean_resid_norm = float(np.linalg.norm(all_activations, axis=1).mean())

    prompt = hiring_prompt(HIRING_NAME)
    rendered_prompt, hiring_tokens = encode_decision_prompt(model, prompt, "native-chat")
    yes_id, no_id = decision_token_ids(model, rendered_prompt, "native-chat")
    hook_name = residual_hook_name(int(vector_meta["probe_layer"]))
    margins: dict[str, float] = {}
    baseline_logits = None
    for axis, vector in (("warmth", warmth), ("competence", competence)):
        for strength in STRENGTHS:
            hooks = []
            if strength != 0.0:
                hooks = [(
                    hook_name,
                    make_steering_hook(vector, strength * mean_resid_norm),
                )]
            with torch.no_grad():
                logits = model.run_with_hooks(hiring_tokens, fwd_hooks=hooks)
            next_logits = logits[0, -1].float()
            margins[f"{axis}:{strength:+.2f}"] = float(
                (next_logits[yes_id] - next_logits[no_id]).item()
            )
            if strength == 0.0 and baseline_logits is None:
                baseline_logits = next_logits
    if baseline_logits is None:
        raise AssertionError("Hiring baseline logits were not captured.")
    arrays["tokens__hiring"] = _tensor_array(hiring_tokens).astype(np.int64)
    arrays["logits__hiring_next"] = _tensor_array(baseline_logits)
    input_records.append(
        {
            "kind": "hiring",
            "id": HIRING_NAME,
            "sha256": _sha256_text(rendered_prompt),
            "token_count": int(hiring_tokens.shape[1]),
            "yes_token_id": yes_id,
            "no_token_id": no_id,
        }
    )

    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
    peak_vram = {
        f"cuda:{index}": torch.cuda.max_memory_allocated(index) / 1024**3
        for index in range(torch.cuda.device_count())
    }

    snapshot_path = Path(args.snapshot)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(snapshot_path, **arrays)
    metadata = {
        "arm": args.arm,
        "model": model_name,
        "seed": args.seed,
        "n_devices": args.n_devices,
        "probe_layer": int(vector_meta["probe_layer"]),
        "input_records": input_records,
        "mean_resid_norm": mean_resid_norm,
        "bridge_hf_max_logit_diff": bridge_hf_max_logit_diff,
        "steering_margins": margins,
        "peak_allocated_vram_gib": peak_vram,
        "topology": topology,
        "runtime": model_runtime_metadata(model),
        "snapshot": str(snapshot_path),
    }
    _json_dump(Path(args.metadata), metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def numeric_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference64 = np.asarray(reference, dtype=np.float64).ravel()
    candidate64 = np.asarray(candidate, dtype=np.float64).ravel()
    delta = candidate64 - reference64
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    rmse = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    reference_rms = (
        float(np.sqrt(np.mean(reference64 * reference64))) if reference64.size else 0.0
    )
    nrmse = rmse / max(reference_rms, 1e-12)
    denominator = float(np.linalg.norm(reference64) * np.linalg.norm(candidate64))
    cosine = 1.0 if denominator == 0.0 and max_abs == 0.0 else 0.0
    if denominator > 0.0:
        cosine = float(np.dot(reference64, candidate64) / denominator)
    cosine_error = max(0.0, 1.0 - min(1.0, cosine))
    return {
        "max_abs": max_abs,
        "rmse": rmse,
        "nrmse": nrmse,
        "cosine": cosine,
        "cosine_error": cosine_error,
    }


def compare_captures(
    capture_paths: dict[str, tuple[Path, Path]],
    *,
    run_id: str,
    git_commit: str,
) -> dict[str, Any]:
    metadata = {
        arm: json.loads(meta_path.read_text(encoding="utf-8"))
        for arm, (_, meta_path) in capture_paths.items()
    }
    snapshots = {
        arm: np.load(snapshot_path)
        for arm, (snapshot_path, _) in capture_paths.items()
    }
    failures: list[str] = []
    comparisons: dict[str, Any] = {}

    for field in ("model", "seed", "probe_layer", "input_records", "mean_resid_norm"):
        values = [metadata[arm][field] for arm in ("single_a", "single_b", "multi")]
        if not (values[0] == values[1] == values[2]):
            failures.append(f"capture metadata differs for {field}")
    runtime_fields = (
        "transformer_lens_version",
        "transformers_version",
        "torch_version",
        "dtype",
    )
    for field in runtime_fields:
        values = [
            metadata[arm]["runtime"][field]
            for arm in ("single_a", "single_b", "multi")
        ]
        if not (values[0] == values[1] == values[2]):
            failures.append(f"runtime metadata differs for {field}")

    key_sets = {arm: set(snapshot.files) for arm, snapshot in snapshots.items()}
    if not (key_sets["single_a"] == key_sets["single_b"] == key_sets["multi"]):
        failures.append("snapshot key sets differ between capture arms")

    common_keys = sorted(set.intersection(*key_sets.values()))
    for key in common_keys:
        single_a = snapshots["single_a"][key]
        single_b = snapshots["single_b"][key]
        multi = snapshots["multi"][key]
        row: dict[str, Any] = {"shape": list(single_a.shape)}
        if single_a.shape != single_b.shape or single_a.shape != multi.shape:
            row["pass"] = False
            row["failure"] = "shape mismatch"
            failures.append(f"{key}: shape mismatch")
            comparisons[key] = row
            continue
        if key.startswith("tokens__"):
            ab_equal = bool(np.array_equal(single_a, single_b))
            am_equal = bool(np.array_equal(single_a, multi))
            row.update({"single_repeat_exact": ab_equal, "multi_exact": am_equal})
            row["pass"] = ab_equal and am_equal
            if not row["pass"]:
                failures.append(f"{key}: token IDs differ")
            comparisons[key] = row
            continue
        if not all(np.isfinite(array).all() for array in (single_a, single_b, multi)):
            row["pass"] = False
            row["failure"] = "non-finite values"
            failures.append(f"{key}: non-finite values")
            comparisons[key] = row
            continue

        repeat = numeric_metrics(single_a, single_b)
        distributed = numeric_metrics(single_a, multi)
        row["single_repeat"] = repeat
        row["multi"] = distributed
        if key.startswith(("logits__", "hf_logits__")):
            limit = max(LOGIT_MAX_ABS_FLOOR, REPEAT_MULTIPLIER * repeat["max_abs"])
            passed = distributed["max_abs"] <= limit
            row["gate"] = {"metric": "max_abs", "limit": limit}
        else:
            nrmse_limit = max(NRMSE_FLOOR, REPEAT_MULTIPLIER * repeat["nrmse"])
            cosine_limit = max(
                COSINE_ERROR_FLOOR,
                REPEAT_MULTIPLIER * repeat["cosine_error"],
            )
            passed = (
                distributed["nrmse"] <= nrmse_limit
                and distributed["cosine_error"] <= cosine_limit
            )
            row["gate"] = {
                "nrmse_limit": nrmse_limit,
                "cosine_error_limit": cosine_limit,
            }
        row["pass"] = bool(passed)
        if not passed:
            failures.append(f"{key}: numeric parity gate failed")
        comparisons[key] = row

    margin_comparisons: dict[str, Any] = {}
    margin_keys = sorted(metadata["single_a"]["steering_margins"])
    for key in margin_keys:
        single_a = float(metadata["single_a"]["steering_margins"][key])
        single_b = float(metadata["single_b"]["steering_margins"][key])
        multi = float(metadata["multi"]["steering_margins"][key])
        repeat_difference = abs(single_a - single_b)
        multi_difference = abs(single_a - multi)
        limit = max(LOGIT_MAX_ABS_FLOOR, REPEAT_MULTIPLIER * repeat_difference)
        passed = np.isfinite([single_a, single_b, multi]).all() and multi_difference <= limit
        margin_comparisons[key] = {
            "single_a": single_a,
            "single_b": single_b,
            "multi": multi,
            "single_repeat_abs_diff": repeat_difference,
            "multi_abs_diff": multi_difference,
            "limit": limit,
            "pass": bool(passed),
        }
        if not passed:
            failures.append(f"steering margin {key}: parity gate failed")

    for arm, arm_metadata in metadata.items():
        if not arm_metadata["topology"]["pass"]:
            failures.append(f"{arm}: topology gate failed")
        if arm_metadata["bridge_hf_max_logit_diff"] > LOGIT_MAX_ABS_FLOOR:
            failures.append(f"{arm}: Bridge/HF logit parity exceeds 0.02")
    multi_peaks = metadata["multi"]["peak_allocated_vram_gib"]
    for device in ("cuda:0", "cuda:1"):
        if float(multi_peaks.get(device, 0.0)) <= 0.0:
            failures.append(f"multi: {device} reports no allocated VRAM")

    return {
        "schema_version": 1,
        "run_id": run_id,
        "git_commit": git_commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not failures else "fail",
        "model": metadata["single_a"]["model"],
        "thresholds": {
            "logit_or_margin_max_abs_floor": LOGIT_MAX_ABS_FLOOR,
            "nrmse_floor": NRMSE_FLOOR,
            "cosine_error_floor": COSINE_ERROR_FLOOR,
            "single_repeat_multiplier": REPEAT_MULTIPLIER,
        },
        "arms": metadata,
        "tensor_comparisons": comparisons,
        "steering_margin_comparisons": margin_comparisons,
        "failures": failures,
    }


def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    capture_specs = (
        ("single_a", 1),
        ("single_b", 1),
        ("multi", 2),
    )
    capture_paths: dict[str, tuple[Path, Path]] = {}
    capture_failures: list[dict[str, Any]] = []

    for arm, n_devices in capture_specs:
        snapshot = work_dir / f"{arm}.npz"
        metadata = work_dir / f"{arm}.json"
        stdout_path = work_dir / f"{arm}.stdout.log"
        stderr_path = work_dir / f"{arm}.stderr.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "capture",
            "--config",
            args.config,
            "--vectors-subdir",
            args.vectors_subdir,
            "--arm",
            arm,
            "--n-devices",
            str(n_devices),
            "--seed",
            str(args.seed),
            "--snapshot",
            str(snapshot),
            "--metadata",
            str(metadata),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not snapshot.exists() or not metadata.exists():
            capture_failures.append(
                {
                    "arm": arm,
                    "n_devices": n_devices,
                    "return_code": completed.returncode,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "stderr_tail": completed.stderr[-12000:],
                }
            )
            break
        capture_paths[arm] = (snapshot, metadata)

    if capture_failures:
        result = {
            "schema_version": 1,
            "run_id": args.run_id,
            "git_commit": args.git_commit,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "phase": "capture",
            "capture_failures": capture_failures,
            "work_dir": str(work_dir),
        }
    else:
        try:
            result = compare_captures(
                capture_paths,
                run_id=args.run_id,
                git_commit=args.git_commit,
            )
            result["work_dir"] = str(work_dir)
        except Exception as exc:
            result = {
                "schema_version": 1,
                "run_id": args.run_id,
                "git_commit": args.git_commit,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "fail",
                "phase": "comparison",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "work_dir": str(work_dir),
            }

    _json_dump(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--config", default="config/config.yaml")
    capture_parser.add_argument("--vectors-subdir", required=True)
    capture_parser.add_argument("--arm", required=True)
    capture_parser.add_argument("--n-devices", required=True, type=int)
    capture_parser.add_argument("--seed", required=True, type=int)
    capture_parser.add_argument("--snapshot", required=True)
    capture_parser.add_argument("--metadata", required=True)
    capture_parser.set_defaults(function=capture)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", default="config/config.yaml")
    run_parser.add_argument("--vectors-subdir", required=True)
    run_parser.add_argument("--seed", default=20260527, type=int)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--git-commit", required=True)
    run_parser.add_argument("--work-dir", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.set_defaults(function=run)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
