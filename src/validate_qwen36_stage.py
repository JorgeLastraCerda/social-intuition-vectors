"""Technical validators for independent Qwen3.6 production stages."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from src.qwen36_pipeline import CONDITIONS, require_outputs_absent, stage_paths
from src.utils.config import ProjectConfig, load_config
from src.utils.hooks import layer_from_fraction


def _finite(values: np.ndarray, label: str) -> None:
    if values.size == 0 or not np.isfinite(values).all():
        raise AssertionError(f"{label}: missing, NaN, or Inf values.")


def _expected(cfg: ProjectConfig) -> dict[str, Any]:
    return {
        "model": cfg.model.name,
        "revision": cfg.model.revision,
        "n_layers": cfg.native_hf.expected_layers,
        "d_model": cfg.native_hf.expected_d_model,
        "probe_layer": layer_from_fraction(
            cfg.native_hf.expected_layers, cfg.probing.probe_layer_frac
        ),
        "probe_layer_frac": cfg.probing.probe_layer_frac,
        "start_token": cfg.probing.start_token,
        "seed": cfg.probing.seed,
        "n_stories": 200,
        "input_format": "raw-passage-explicit-bos",
    }


def _match(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise AssertionError(
                f"{label}: {key} expected {value!r}, got {payload.get(key)!r}."
            )


def _validate_gpu_runtime(cfg: ProjectConfig, runtime: dict[str, Any]) -> None:
    expected = {
        "backend": "huggingface-native",
        "transformer_lens_version": "not-installed",
        "transformer_lens_imported": False,
        "model_revision_requested": cfg.model.revision,
        "model_revision_resolved": cfg.model.revision,
        "parameter_devices": ["cuda:0"],
        "vision_forward_calls": 0,
    }
    _match(runtime, expected, "runtime")
    if float(runtime["hook_hidden_max_diff"]) > 1e-5:
        raise AssertionError("Hook activation does not match the hidden-state output.")
    if float(runtime["passive_hook_max_logit_diff"]) > 1e-5:
        raise AssertionError("Passive hook changed final-token logits.")
    if "RTX PRO 6000" not in str(runtime.get("cuda_device_name")):
        raise AssertionError(f"Unexpected GPU: {runtime.get('cuda_device_name')!r}.")
    if float(runtime["peak_reserved_vram_fraction"]) > cfg.native_hf.max_vram_fraction:
        raise AssertionError("Peak reserved VRAM exceeds the configured capacity gate.")


def validate_stage1(cfg: ProjectConfig) -> None:
    paths = stage_paths(cfg)
    meta_path = paths.vectors_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _match(meta, _expected(cfg), str(meta_path))
    if len(str(meta.get("stimuli_sha256", ""))) != 64:
        raise AssertionError("Stage 1 metadata is missing a SHA-256 stimulus hash.")
    for condition in CONDITIONS:
        path = paths.vectors_dir / f"X_{condition}.npy"
        matrix = np.load(path, mmap_mode="r")
        shape = (50, cfg.native_hf.expected_d_model)
        if matrix.shape != shape:
            raise AssertionError(f"{path}: expected {shape}, got {matrix.shape}.")
        _finite(matrix, str(path))
    for filename in ("warmth_vec.npy", "competence_vec.npy"):
        path = paths.vectors_dir / filename
        vector = np.load(path, mmap_mode="r")
        if vector.shape != (cfg.native_hf.expected_d_model,):
            raise AssertionError(f"{path}: unexpected shape {vector.shape}.")
        _finite(vector, str(path))
        if not np.any(np.asarray(vector) != 0):
            raise AssertionError(f"{path}: vector must be nonzero.")
    technical = json.loads(paths.technical_logs[1].read_text(encoding="utf-8"))
    _match(
        technical,
        {"status": "pass", "stage": 1, "label": cfg.native_hf.label, **_expected(cfg)},
        str(paths.technical_logs[1]),
    )
    _validate_gpu_runtime(cfg, technical["runtime"])


def validate_stage2(cfg: ProjectConfig) -> None:
    validate_stage1(cfg)
    paths = stage_paths(cfg)
    table = pd.read_csv(paths.probe_table)
    if len(table) != 2 or set(table["axis"]) != {"warmth", "competence"}:
        raise AssertionError("Stage 2 must contain exactly warmth and competence rows.")
    required_columns = {
        "direction_topic_cv_mean",
        "direction_topic_cv_std",
        "direction_topic_cv_folds",
    }
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise AssertionError(
            f"Stage 2 is missing strict direction columns: {sorted(missing_columns)}."
        )
    numeric = table.drop(columns=["axis"]).select_dtypes(include=[np.number])
    _finite(numeric.to_numpy(float), str(paths.probe_table))
    log = json.loads(paths.probe_log.read_text(encoding="utf-8"))
    _match(log["meta"], _expected(cfg), f"{paths.probe_log}:meta")
    if log.get("scientific_flags_are_non_gating") is not True:
        raise AssertionError("Stage 2 scientific flags must be explicitly non-gating.")
    for axis in ("warmth", "competence"):
        payload = log.get(axis, {})
        for key in ("direction_topic_cv_mean", "direction_topic_cv_std"):
            _finite(np.asarray([payload.get(key)], dtype=float), f"{axis}:{key}")
        folds = np.asarray(payload.get("direction_topic_cv_folds", []), dtype=float)
        if folds.shape != (5,):
            raise AssertionError(f"{axis}:direction_topic_cv_folds must have five folds.")
        _finite(folds, f"{axis}:direction_topic_cv_folds")
    strict_prefixes = (
        "cross_warmth_to_competence_topic_transfer",
        "cross_competence_to_warmth_topic_transfer",
    )
    for prefix in strict_prefixes:
        _finite(
            np.asarray([log.get(f"{prefix}_mean"), log.get(f"{prefix}_std")], dtype=float),
            prefix,
        )
        folds = np.asarray(log.get(f"{prefix}_folds", []), dtype=float)
        if folds.shape != (5,):
            raise AssertionError(f"{prefix}_folds must have five folds.")
        _finite(folds, f"{prefix}_folds")
    aliases = (
        ("cross_warmth_on_competence_cv", "cross_warmth_on_competence_calibrated_cv"),
        ("cross_competence_on_warmth_cv", "cross_competence_on_warmth_calibrated_cv"),
    )
    for legacy, calibrated in aliases:
        if log.get(legacy) != log.get(calibrated):
            raise AssertionError(f"{calibrated} must equal its compatibility field {legacy}.")
    for key in (
        "pass_warmth_cv",
        "pass_competence_cv",
        "pass_orthogonality",
        "pass_warmth_topic_cv",
        "pass_competence_topic_cv",
    ):
        if not isinstance(log.get(key), bool):
            raise AssertionError(f"{paths.probe_log}: {key} must be boolean.")
    technical = json.loads(paths.technical_logs[2].read_text(encoding="utf-8"))
    _match(
        technical,
        {
            "status": "pass",
            "stage": 2,
            "label": cfg.native_hf.label,
            "model": cfg.model.name,
            "revision": cfg.model.revision,
            "seed": cfg.probing.seed,
            "stimuli_sha256": log["meta"]["stimuli_sha256"],
            "backend": "numpy-scikit-learn-cpu",
        },
        str(paths.technical_logs[2]),
    )


def validate_stage3(cfg: ProjectConfig) -> None:
    paths = stage_paths(cfg)
    table = pd.read_csv(paths.sweep_table)
    n_layers = cfg.native_hf.expected_layers
    expected_layer = layer_from_fraction(n_layers, cfg.probing.probe_layer_frac)
    if len(table) != n_layers or table["layer"].tolist() != list(range(n_layers)):
        raise AssertionError("Stage 3 layer rows are incomplete or unordered.")
    probe_rows = table[table["is_probe_layer"].astype(str).str.lower() == "true"]
    if probe_rows["layer"].tolist() != [expected_layer]:
        raise AssertionError("Stage 3 probe-layer marker is incorrect.")
    for column in (
        "frac",
        "warmth_topic_cv",
        "warmth_topic_cv_std",
        "comp_topic_cv",
        "comp_topic_cv_std",
        "warmth_cohens_d",
        "comp_cohens_d",
        "cos_wc",
        "mean_resid_norm",
    ):
        _finite(table[column].to_numpy(float), f"{paths.sweep_table}:{column}")
    if not (table["mean_resid_norm"].to_numpy(float) > 0).all():
        raise AssertionError("Stage 3 residual norms must be positive.")
    meta = json.loads(paths.sweep_meta.read_text(encoding="utf-8"))
    _match(meta, _expected(cfg), str(paths.sweep_meta))
    technical = json.loads(paths.technical_logs[3].read_text(encoding="utf-8"))
    _match(
        technical,
        {"status": "pass", "stage": 3, "label": cfg.native_hf.label, **_expected(cfg)},
        str(paths.technical_logs[3]),
    )
    _validate_gpu_runtime(cfg, technical["runtime"])


def cross_stage_audit(cfg: ProjectConfig) -> dict[str, Any]:
    validate_stage2(cfg)
    validate_stage3(cfg)
    paths = stage_paths(cfg)
    table = pd.read_csv(paths.probe_table).set_index("axis")
    sweep = pd.read_csv(paths.sweep_table)
    probe = sweep[sweep["is_probe_layer"].astype(str).str.lower() == "true"].iloc[0]
    log = json.loads(paths.probe_log.read_text(encoding="utf-8"))
    differences = {
        "warmth_cohens_d": abs(
            float(table.loc["warmth", "cohens_d"]) - float(probe["warmth_cohens_d"])
        ),
        "competence_cohens_d": abs(
            float(table.loc["competence", "cohens_d"]) - float(probe["comp_cohens_d"])
        ),
        "axis_cosine": abs(float(log["axis_cosine"]) - float(probe["cos_wc"])),
    }
    return {
        "tolerance": 1e-6,
        "differences": differences,
        "pass": all(value <= 1e-6 for value in differences.values()),
        "non_gating_reproducibility_audit": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3))
    parser.add_argument("--require-absent", action="store_true")
    parser.add_argument("--cross-stage", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.cross_stage:
        print(json.dumps(cross_stage_audit(cfg), indent=2))
        return
    if args.stage is None:
        raise SystemExit("--stage is required unless --cross-stage is used.")
    paths = stage_paths(cfg)
    if args.require_absent:
        if args.stage == 2:
            validate_stage1(cfg)
        require_outputs_absent(paths, args.stage)
        print(f"[preflight] {cfg.native_hf.label} Stage {args.stage} targets are absent")
        return
    {1: validate_stage1, 2: validate_stage2, 3: validate_stage3}[args.stage](cfg)
    print(f"[validated] {cfg.native_hf.label} Stage {args.stage} technical gates passed")


if __name__ == "__main__":
    main()
