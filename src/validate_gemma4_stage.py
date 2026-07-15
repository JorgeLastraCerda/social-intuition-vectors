"""Technical acceptance gates for Gemma 4 replication stages 1--3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config

CONDITIONS = (
    "high_warmth",
    "low_warmth",
    "high_competence",
    "low_competence",
)


def stage_targets(
    stage: int,
    *,
    processed_dir: Path,
    tables_dir: Path,
    logs_dir: Path,
    vectors_subdir: str,
    label: str,
) -> tuple[Path, ...]:
    if stage == 1:
        return (processed_dir / vectors_subdir,)
    if stage == 2:
        return (
            tables_dir / f"probe_metrics_{label}.csv",
            logs_dir / f"validate_probes_{label}.json",
        )
    if stage == 3:
        csv_path = tables_dir / f"layer_sweep_{label}.csv"
        return (csv_path, csv_path.with_suffix(".meta.json"))
    raise ValueError(f"stage must be 1, 2, or 3; got {stage}")


def require_targets_absent(paths: tuple[Path, ...]) -> None:
    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(
            "Refusing to overwrite existing stage outputs: " + ", ".join(collisions)
        )


def require_finite_array(array: np.ndarray, path: Path) -> None:
    if not np.isfinite(array).all():
        raise AssertionError(f"{path}: contains NaN or Inf")


def validate_stage1(
    vectors_dir: Path,
    *,
    model: str,
    expected_layers: int,
    expected_d_model: int,
    expected_layer: int,
) -> None:
    meta_path = vectors_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_meta = {
        "model": model,
        "n_layers": expected_layers,
        "d_model": expected_d_model,
        "probe_layer": expected_layer,
        "seed": 20260527,
        "input_format": "raw-passage",
    }
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise AssertionError(
                f"{meta_path}: {key} expected {value!r}, got {meta.get(key)!r}"
            )

    for condition in CONDITIONS:
        path = vectors_dir / f"X_{condition}.npy"
        array = np.load(path, mmap_mode="r")
        if array.shape != (50, expected_d_model):
            raise AssertionError(f"{path}: expected (50, {expected_d_model}), got {array.shape}")
        require_finite_array(array, path)

    for name in ("warmth_vec.npy", "competence_vec.npy"):
        path = vectors_dir / name
        vector = np.load(path, mmap_mode="r")
        if vector.shape != (expected_d_model,):
            raise AssertionError(
                f"{path}: expected ({expected_d_model},), got {vector.shape}"
            )
        require_finite_array(vector, path)
        if float(np.linalg.norm(vector)) <= 0.0:
            raise AssertionError(f"{path}: vector norm must be positive")


def validate_stage2(
    table_path: Path,
    log_path: Path,
    *,
    model: str,
    expected_layers: int,
    expected_d_model: int,
    expected_layer: int,
) -> None:
    table = pd.read_csv(table_path)
    if len(table) != 2 or set(table["axis"]) != {"warmth", "competence"}:
        raise AssertionError(f"{table_path}: expected exactly warmth and competence rows")
    numeric = table.drop(columns=["axis"]).select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy(float)).all():
        raise AssertionError(f"{table_path}: missing, NaN, or Inf numeric metrics")

    log = json.loads(log_path.read_text(encoding="utf-8"))
    meta = log.get("meta", {})
    expected_meta = {
        "model": model,
        "n_layers": expected_layers,
        "d_model": expected_d_model,
        "probe_layer": expected_layer,
    }
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise AssertionError(
                f"{log_path}: meta.{key} expected {value!r}, got {meta.get(key)!r}"
            )
    for key in (
        "pass_warmth_cv",
        "pass_competence_cv",
        "pass_orthogonality",
        "pass_warmth_topic_cv",
        "pass_competence_topic_cv",
    ):
        if not isinstance(log.get(key), bool):
            raise AssertionError(f"{log_path}: {key} must be boolean")


def validate_stage3(
    table_path: Path,
    meta_path: Path,
    *,
    model: str,
    expected_layers: int,
    expected_d_model: int,
    expected_layer: int,
) -> None:
    table = pd.read_csv(table_path)
    if len(table) != expected_layers:
        raise AssertionError(
            f"{table_path}: expected {expected_layers} rows, got {len(table)}"
        )
    if table["layer"].tolist() != list(range(expected_layers)):
        raise AssertionError(f"{table_path}: layer column is incomplete or out of order")
    if "is_probe_layer" not in table:
        raise AssertionError(f"{table_path}: missing is_probe_layer")
    probe_rows = table[table["is_probe_layer"].astype(str).str.lower() == "true"]
    if probe_rows["layer"].tolist() != [expected_layer]:
        raise AssertionError(
            f"{table_path}: expected only layer {expected_layer} as probe layer"
        )
    for column in (
        "warmth_topic_cv",
        "comp_topic_cv",
        "warmth_cohens_d",
        "comp_cohens_d",
        "cos_wc",
        "mean_resid_norm",
        "frac",
    ):
        if column not in table or not np.isfinite(table[column].to_numpy(float)).all():
            raise AssertionError(f"{table_path}: missing or non-finite {column}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_meta = {
        "model": model,
        "n_layers": expected_layers,
        "d_model": expected_d_model,
        "probe_layer": expected_layer,
        "seed": 20260527,
        "n_stories": 200,
        "input_format": "raw-passage",
    }
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise AssertionError(
                f"{meta_path}: {key} expected {value!r}, got {meta.get(key)!r}"
            )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    processed_dir = Path(cfg.paths.processed)
    tables_dir = Path(cfg.paths.results) / "tables"
    logs_dir = Path(cfg.paths.logs)
    targets = stage_targets(
        args.stage,
        processed_dir=processed_dir,
        tables_dir=tables_dir,
        logs_dir=logs_dir,
        vectors_subdir=args.vectors_subdir,
        label=args.label,
    )
    if args.require_absent:
        require_targets_absent(targets)
        print(f"[preflight] stage {args.stage} targets are absent")
        return

    expected_layer = round((args.expected_layers - 1) * cfg.probing.probe_layer_frac)
    common = {
        "model": args.model,
        "expected_layers": args.expected_layers,
        "expected_d_model": args.expected_d_model,
        "expected_layer": expected_layer,
    }
    if args.stage == 1:
        validate_stage1(targets[0], **common)
    elif args.stage == 2:
        validate_stage2(targets[0], targets[1], **common)
    else:
        validate_stage3(targets[0], targets[1], **common)
    print(f"[validated] {args.label} stage {args.stage}: technical gates passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stage", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--vectors-subdir", required=True)
    parser.add_argument("--expected-layers", required=True, type=int)
    parser.add_argument("--expected-d-model", required=True, type=int)
    parser.add_argument("--require-absent", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
