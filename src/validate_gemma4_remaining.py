"""Write-once structural and provenance gates for Gemma 4 remaining experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ProjectConfig, load_config


def model_label(cfg: ProjectConfig) -> str:
    label = cfg.smoke.label
    return label.removesuffix("_pinned")


def vectors_subdir(cfg: ProjectConfig) -> str:
    return f"concept_vectors_{model_label(cfg)}"


def require_revision(payload: dict, cfg: ProjectConfig, *, context: str) -> None:
    runtime = payload.get("runtime", payload)
    requested = runtime.get("model_revision_requested", payload.get("revision"))
    resolved = runtime.get("model_revision_resolved", payload.get("revision"))
    if requested != cfg.model.revision or resolved != cfg.model.revision:
        raise AssertionError(
            f"{context}: revision mismatch requested={requested!r} resolved={resolved!r} "
            f"expected={cfg.model.revision!r}"
        )


def paths_for_run(cfg: ProjectConfig, run: str, *, full282: bool = False) -> list[Path]:
    label = model_label(cfg)
    tables = Path(cfg.paths.results) / "tables"
    logs = Path(cfg.paths.logs)
    vectors = Path(cfg.paths.processed) / vectors_subdir(cfg)
    if run == "smoke":
        return [logs / f"smoke_{cfg.smoke.label}.json"]
    if run == "neutral":
        return [vectors / "X_neutral.npy", vectors / "neutral_meta.json"]
    if run == "pca":
        return [
            vectors / "concept_vectors_denoised.npz",
            vectors / "denoise_summary.json",
        ]
    if run in {"dense_raw", "dense_denoised"}:
        kind = run.removeprefix("dense_")
        output_label = f"{label}_{kind}"
        return [
            tables / f"steering_dense_raw_{output_label}.csv",
            tables / f"steering_dense_{output_label}.csv",
            tables / f"steering_dense_null_{output_label}.csv",
            logs / f"steering_dense_{output_label}.json",
        ]
    if run == "audit":
        return [
            tables / f"hiring_audit_{label}.csv",
            logs / f"hiring_probe_vs_human_{label}.json",
        ]
    if run in {"hiring_local", "hiring_broad", "hiring_denoised"}:
        suffix = {
            "hiring_local": "local",
            "hiring_broad": "broad",
            "hiring_denoised": "denoised_local",
        }[run]
        if full282:
            suffix += "_full282"
        output_label = f"{label}_{suffix}"
        return [
            tables / f"hiring_steering_raw_{output_label}.csv",
            tables / f"hiring_steering_{output_label}.csv",
            logs / f"hiring_steering_{output_label}.json",
            logs / f"hiring_steering_summary_{output_label}.json",
        ]
    if run == "posthoc":
        return [
            tables / f"hiring_disparity_{label}.csv",
            logs / f"hiring_mediation_{label}.json",
            tables / f"hiring_group_r4_{label}.csv",
            tables / f"hiring_name_level_{label}.csv",
            logs / f"hiring_r4_{label}.json",
        ]
    if run == "full282_gate":
        return [logs / f"hiring_full282_gate_{label}.json"]
    raise ValueError(f"Unknown run {run!r}")


def validate(cfg: ProjectConfig, run: str, *, full282: bool = False) -> None:
    paths = paths_for_run(cfg, run, full282=full282)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{run}: missing outputs: {missing}")
    label = model_label(cfg)
    if run == "smoke":
        payload = json.loads(paths[0].read_text(encoding="utf-8"))
        require_revision(payload, cfg, context=run)
        if payload["model"] != cfg.model.name:
            raise AssertionError(f"smoke model mismatch: {payload['model']!r}")
        if (payload["n_layers"], payload["d_model"]) != (
            cfg.smoke.expected_layers,
            cfg.smoke.expected_d_model,
        ):
            raise AssertionError("smoke architecture mismatch")
        if not np.isfinite(
            [
                payload["bridge_hf_max_logit_diff"],
                payload["baseline_margin"],
                payload["steered_margin"],
            ]
        ).all():
            raise AssertionError("smoke contains non-finite metrics")
        if payload["baseline_margin"] == payload["steered_margin"]:
            raise AssertionError("smoke steering produced no change")
        return
    if run == "neutral":
        array = np.load(paths[0], mmap_mode="r")
        if array.shape != (cfg.neutral.n_texts, cfg.smoke.expected_d_model):
            raise AssertionError(f"neutral shape mismatch: {array.shape}")
        payload = json.loads(paths[1].read_text(encoding="utf-8"))
        require_revision(payload, cfg, context=run)
        if payload["seed"] != cfg.probing.seed:
            raise AssertionError("neutral seed mismatch")
        return
    if run == "pca":
        archive = np.load(paths[0])
        required = {
            "warmth",
            "competence",
            "neutral_pca_components",
            "k",
            "variance_threshold",
        }
        if not required.issubset(archive.files):
            raise AssertionError(
                f"PCA archive missing {sorted(required - set(archive.files))}"
            )
        if (
            int(archive["k"]) < 1
            or float(archive["variance_threshold"]) != cfg.neutral.variance_threshold
        ):
            raise AssertionError("PCA selection metadata mismatch")
        if (
            not np.isfinite(archive["warmth"]).all()
            or not np.isfinite(archive["competence"]).all()
        ):
            raise AssertionError("PCA vectors are non-finite")
        return
    if run in {"dense_raw", "dense_denoised"}:
        raw = pd.read_csv(paths[0])
        if len(raw) != 10440:
            raise AssertionError(f"{run}: expected 10440 raw rows, got {len(raw)}")
        required = {
            "judgment_axis",
            "steering_axis",
            "direction_type",
            "random_id",
            "vector_kind",
            "strength",
            "margin",
            "delta_margin",
        }
        if not required.issubset(raw.columns):
            raise AssertionError(f"{run}: missing enhanced columns")
        zero = raw[(raw["mode"] == "steering") & (raw["strength"] == 0.0)][
            "delta_margin"
        ]
        if not np.array_equal(zero.to_numpy(float), np.zeros(len(zero))):
            raise AssertionError(f"{run}: zero-strength deltas are not exact zero")
        if (
            raw["direction_type"].eq("random").groupby(raw["judgment_axis"]).sum().min()
            != 5000
        ):
            raise AssertionError(
                f"{run}: expected 50 random directions per judgment axis"
            )
        null = pd.read_csv(paths[2])
        if len(null) != 4 or set(null["n_random_directions"]) != {50}:
            raise AssertionError(f"{run}: null summary contract failed")
        payload = json.loads(paths[3].read_text(encoding="utf-8"))
        require_revision(payload, cfg, context=run)
        return
    if run == "audit":
        audit = pd.read_csv(paths[0])
        if len(audit) != 282:
            raise AssertionError(f"audit expected 282 names, got {len(audit)}")
        for column in ("model_warmth", "model_competence", "callback_margin"):
            if not np.isfinite(audit[column].to_numpy(float)).all():
                raise AssertionError(f"audit non-finite {column}")
        require_revision(
            json.loads(paths[1].read_text(encoding="utf-8")), cfg, context=run
        )
        return
    if run in {"hiring_local", "hiring_broad", "hiring_denoised"}:
        raw = pd.read_csv(paths[0])
        expected = 2820 if full282 else 600
        if len(raw) != expected:
            raise AssertionError(f"{run}: expected {expected} rows, got {len(raw)}")
        zero = raw[raw["strength"] == 0.0]["delta"].to_numpy(float)
        if not np.array_equal(zero, np.zeros_like(zero)):
            raise AssertionError(f"{run}: zero-strength deltas are not exact zero")
        summary = pd.read_csv(paths[1])
        if len(summary) != 10 or set(summary["n_boot"]) != {5000}:
            raise AssertionError(f"{run}: summary contract failed")
        require_revision(
            json.loads(paths[2].read_text(encoding="utf-8")), cfg, context=run
        )
        return
    if run == "posthoc":
        r4 = json.loads(paths[-1].read_text(encoding="utf-8"))
        if r4["n_matched"] != 149:
            raise AssertionError(
                f"{label}: expected 149 R4 matches, got {r4['n_matched']}"
            )
        mediation = json.loads(paths[1].read_text(encoding="utf-8"))
        if mediation["n_boot"] != 5000 or mediation["seed"] != cfg.probing.seed:
            raise AssertionError("posthoc bootstrap contract failed")
        return
    if run == "full282_gate":
        payload = json.loads(paths[0].read_text(encoding="utf-8"))
        if not isinstance(payload.get("run_full_282"), bool):
            raise AssertionError("full282 gate lacks boolean decision")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    targets = paths_for_run(cfg, args.run, full282=args.full282)
    if args.require_absent:
        collisions = [str(path) for path in targets if path.exists()]
        if collisions:
            raise FileExistsError(f"Refusing existing outputs: {collisions}")
        print(f"[validated] {args.run}: {len(targets)} outputs absent")
        return
    validate(cfg, args.run, full282=args.full282)
    print(f"[validated] {args.run}: {len(targets)} outputs complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run",
        required=True,
        choices=(
            "smoke",
            "neutral",
            "pca",
            "dense_raw",
            "dense_denoised",
            "audit",
            "hiring_local",
            "hiring_broad",
            "hiring_denoised",
            "posthoc",
            "full282_gate",
        ),
    )
    parser.add_argument("--full282", action="store_true")
    parser.add_argument("--require-absent", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
