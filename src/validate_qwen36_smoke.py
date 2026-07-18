"""Structural and numerical acceptance gates for the Qwen3.6 Stage 1--3 smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.qwen36_smoke import CONDITIONS, require_outputs_absent, smoke_paths
from src.utils.config import load_config
from src.utils.hooks import layer_from_fraction


def require_finite(values: np.ndarray, label: str) -> None:
    if values.size == 0 or not np.isfinite(values).all():
        raise AssertionError(f"{label}: missing, NaN, or Inf values")


def validate(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    paths = smoke_paths(cfg)
    expected_layer = layer_from_fraction(
        cfg.smoke.expected_layers, cfg.probing.probe_layer_frac
    )

    technical = json.loads(paths.technical_log.read_text(encoding="utf-8"))
    expected_technical = {
        "status": "pass",
        "model": cfg.model.name,
        "revision": cfg.model.revision,
        "seed": cfg.probing.seed,
        "n_layers": cfg.smoke.expected_layers,
        "d_model": cfg.smoke.expected_d_model,
        "probe_layer": expected_layer,
        "vision_forward_calls": 0,
        "n_stories": cfg.smoke.n_topics * len(CONDITIONS),
    }
    for key, value in expected_technical.items():
        if technical.get(key) != value:
            raise AssertionError(
                f"{paths.technical_log}: {key} expected {value!r}, "
                f"got {technical.get(key)!r}"
            )
    if float(technical["hook_hidden_max_diff"]) > 1e-5:
        raise AssertionError("Hook activation does not match hidden_states[probe_layer+1].")
    if float(technical["passive_hook_max_logit_diff"]) > 1e-5:
        raise AssertionError("Passive hook changed the final-token logits.")
    if int(technical["token_length_min"]) <= cfg.probing.start_token:
        raise AssertionError("At least one smoke passage does not exceed start_token.")
    runtime = technical.get("runtime", {})
    if runtime.get("backend") != "huggingface-native":
        raise AssertionError("Smoke did not use the native Hugging Face backend.")
    if runtime.get("transformer_lens_imported") is not False:
        raise AssertionError("TransformerLens was imported by the smoke process.")
    if runtime.get("transformer_lens_version") != "not-installed":
        raise AssertionError("TransformerLens is installed in the isolated smoke environment.")
    if runtime.get("parameter_devices") != ["cuda:0"]:
        raise AssertionError(
            f"Expected all parameters on cuda:0, got {runtime.get('parameter_devices')!r}."
        )
    if float(runtime["peak_reserved_vram_fraction"]) > cfg.smoke.max_vram_fraction:
        raise AssertionError("Peak reserved VRAM exceeds the configured capacity gate.")

    meta_path = paths.vectors_dir / "meta.json"
    stage1_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_stage1 = {
        "model": cfg.model.name,
        "revision": cfg.model.revision,
        "probe_layer": expected_layer,
        "n_layers": cfg.smoke.expected_layers,
        "d_model": cfg.smoke.expected_d_model,
        "seed": cfg.probing.seed,
        "input_format": "raw-passage-explicit-bos",
        "smoke": True,
    }
    for key, value in expected_stage1.items():
        if stage1_meta.get(key) != value:
            raise AssertionError(
                f"{meta_path}: {key} expected {value!r}, got {stage1_meta.get(key)!r}"
            )
    selected_topics = stage1_meta.get("selected_topics", [])
    if len(selected_topics) != cfg.smoke.n_topics or len(set(selected_topics)) != len(
        selected_topics
    ):
        raise AssertionError("Stage 1 selected_topics is incomplete or duplicated.")

    for condition in CONDITIONS:
        path = paths.vectors_dir / f"X_{condition}.npy"
        array = np.load(path, mmap_mode="r")
        expected_shape = (cfg.smoke.n_topics, cfg.smoke.expected_d_model)
        if array.shape != expected_shape:
            raise AssertionError(f"{path}: expected {expected_shape}, got {array.shape}")
        require_finite(array, str(path))
    for filename in ("warmth_vec.npy", "competence_vec.npy"):
        path = paths.vectors_dir / filename
        vector = np.load(path, mmap_mode="r")
        if vector.shape != (cfg.smoke.expected_d_model,):
            raise AssertionError(f"{path}: unexpected shape {vector.shape}")
        require_finite(vector, str(path))
        if not np.any(np.asarray(vector) != 0):
            raise AssertionError(f"{path}: vector norm must be positive")

    probe_table = pd.read_csv(paths.probe_table)
    if len(probe_table) != 2 or set(probe_table["axis"]) != {"warmth", "competence"}:
        raise AssertionError("Stage 2 must contain exactly warmth and competence rows.")
    probe_numeric = probe_table.drop(columns=["axis"]).select_dtypes(include=[np.number])
    require_finite(probe_numeric.to_numpy(float), str(paths.probe_table))
    probe_log = json.loads(paths.probe_log.read_text(encoding="utf-8"))
    if probe_log.get("scientific_flags_are_non_gating") is not True:
        raise AssertionError("Smoke scientific flags must be explicitly non-gating.")
    for key in (
        "pass_warmth_cv",
        "pass_competence_cv",
        "pass_orthogonality",
        "pass_warmth_topic_cv",
        "pass_competence_topic_cv",
    ):
        if not isinstance(probe_log.get(key), bool):
            raise AssertionError(f"{paths.probe_log}: {key} must be boolean")

    sweep = pd.read_csv(paths.sweep_table)
    expected_layers = cfg.smoke.expected_layers
    if len(sweep) != expected_layers:
        raise AssertionError(f"Stage 3 expected {expected_layers} rows, got {len(sweep)}")
    if sweep["layer"].tolist() != list(range(expected_layers)):
        raise AssertionError("Stage 3 layer column is incomplete or unordered.")
    probe_rows = sweep[sweep["is_probe_layer"].astype(str).str.lower() == "true"]
    if probe_rows["layer"].tolist() != [expected_layer]:
        raise AssertionError("Stage 3 has an incorrect probe-layer marker.")
    columns = (
        "frac",
        "warmth_topic_cv",
        "warmth_topic_cv_std",
        "comp_topic_cv",
        "comp_topic_cv_std",
        "warmth_cohens_d",
        "comp_cohens_d",
        "cos_wc",
        "mean_resid_norm",
    )
    for column in columns:
        require_finite(sweep[column].to_numpy(float), f"{paths.sweep_table}:{column}")
    if not (sweep["mean_resid_norm"].to_numpy(float) > 0).all():
        raise AssertionError("Every Stage 3 mean residual norm must be positive.")

    sweep_meta = json.loads(paths.sweep_meta.read_text(encoding="utf-8"))
    expected_sweep_meta = {
        "model": cfg.model.name,
        "revision": cfg.model.revision,
        "n_layers": expected_layers,
        "d_model": cfg.smoke.expected_d_model,
        "probe_layer": expected_layer,
        "seed": cfg.probing.seed,
        "n_stories": cfg.smoke.n_topics * len(CONDITIONS),
        "smoke": True,
    }
    for key, value in expected_sweep_meta.items():
        if sweep_meta.get(key) != value:
            raise AssertionError(
                f"{paths.sweep_meta}: {key} expected {value!r}, got {sweep_meta.get(key)!r}"
            )

    by_axis = probe_table.set_index("axis")
    probe_row = probe_rows.iloc[0]
    comparisons = (
        (float(by_axis.loc["warmth", "cohens_d"]), float(probe_row["warmth_cohens_d"])),
        (
            float(by_axis.loc["competence", "cohens_d"]),
            float(probe_row["comp_cohens_d"]),
        ),
        (float(probe_log["axis_cosine"]), float(probe_row["cos_wc"])),
    )
    for stage2_value, stage3_value in comparisons:
        if abs(stage2_value - stage3_value) > 1e-6:
            raise AssertionError(
                f"Stage 2/3 probe-layer mismatch: {stage2_value} vs {stage3_value}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/qwen36_smoke.yaml")
    parser.add_argument("--require-absent", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.require_absent:
        require_outputs_absent(smoke_paths(cfg))
        print("[preflight] Qwen3.6 smoke outputs are absent")
        return
    validate(args.config)
    print("[validated] Qwen3.6 Stage 1--3 smoke technical gates passed")


if __name__ == "__main__":
    main()
