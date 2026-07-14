"""Fail-fast structural validation for a completed Gemma 4 replication run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config


def require_rows(path: Path, expected: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != expected:
        raise AssertionError(f"{path}: expected {expected} rows, got {len(frame)}")
    return frame


def require_finite(frame: pd.DataFrame, path: Path, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column not in frame or not np.isfinite(frame[column].to_numpy(float)).all():
            raise AssertionError(f"{path}: missing or non-finite {column}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    vectors = Path(cfg.paths.processed) / args.vectors_subdir
    tables = Path(cfg.paths.results) / "tables"
    logs = Path(cfg.paths.logs)
    meta = json.loads((vectors / "meta.json").read_text(encoding="utf-8"))
    expected_layer = round((args.expected_layers - 1) * cfg.probing.probe_layer_frac)
    expected = {
        "model": args.model,
        "n_layers": args.expected_layers,
        "d_model": args.expected_d_model,
        "probe_layer": expected_layer,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise AssertionError(f"meta.{key}: expected {value!r}, got {meta.get(key)!r}")
    for condition in (
        "high_warmth",
        "low_warmth",
        "high_competence",
        "low_competence",
    ):
        array = np.load(vectors / f"X_{condition}.npy", mmap_mode="r")
        if array.shape != (50, args.expected_d_model):
            raise AssertionError(f"X_{condition}: unexpected shape {array.shape}")
    neutral = np.load(vectors / "X_neutral.npy", mmap_mode="r")
    if neutral.shape != (cfg.neutral.n_texts, args.expected_d_model):
        raise AssertionError(f"X_neutral: unexpected shape {neutral.shape}")
    if not (vectors / "concept_vectors_denoised.npz").exists():
        raise FileNotFoundError(vectors / "concept_vectors_denoised.npz")

    layer = require_rows(tables / f"layer_sweep_{args.label}.csv", args.expected_layers)
    require_finite(layer, tables / f"layer_sweep_{args.label}.csv", ("warmth_cohens_d", "comp_cohens_d", "cos_wc"))
    audit = require_rows(tables / f"hiring_audit_{args.label}.csv", 282)
    require_finite(audit, tables / f"hiring_audit_{args.label}.csv", ("callback_margin", "model_warmth", "model_competence"))
    baseline_by_suffix = {}
    for suffix in ("broad", "local", "denoised_local"):
        path = tables / f"hiring_steering_raw_{args.label}_{suffix}.csv"
        frame = require_rows(path, 600)
        require_finite(frame, path, ("margin", "delta"))
        zero = frame[frame["strength"] == 0.0]["delta"].to_numpy(float)
        if not np.array_equal(zero, np.zeros_like(zero)):
            raise AssertionError(f"{path}: strength-zero delta is not exactly zero")
        baseline_by_suffix[suffix] = (
            frame[frame["strength"] == 0.0]
            .sort_values(["axis", "name"])["margin"]
            .to_numpy(float)
        )
    if not np.array_equal(baseline_by_suffix["broad"], baseline_by_suffix["local"]):
        raise AssertionError("Broad and local raw baselines differ")
    if not np.array_equal(
        baseline_by_suffix["broad"], baseline_by_suffix["denoised_local"]
    ):
        raise AssertionError("Raw and denoised baselines differ")
    r4_log = json.loads(
        (logs / f"hiring_r4_{args.label}.json").read_text(encoding="utf-8")
    )
    if r4_log["n_matched"] != args.expected_r4_names:
        raise AssertionError(
            f"R4 matched {r4_log['n_matched']} names, expected {args.expected_r4_names}"
        )
    print(f"[validated] {args.label}: all structural acceptance gates passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--vectors-subdir", required=True)
    parser.add_argument("--expected-layers", required=True, type=int)
    parser.add_argument("--expected-d-model", required=True, type=int)
    parser.add_argument("--expected-r4-names", default=149, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
