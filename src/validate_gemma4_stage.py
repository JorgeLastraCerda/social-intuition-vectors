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


def validate_stage3b(
    table_path: Path,
    meta_path: Path,
    audit_path: Path,
    *,
    model: str,
    expected_layers: int,
    expected_d_model: int,
    expected_layer: int,
) -> None:
    """Validate enhanced per-layer direction, transfer, and bootstrap artifacts."""
    validate_stage3(
        table_path,
        meta_path,
        model=model,
        expected_layers=expected_layers,
        expected_d_model=expected_d_model,
        expected_layer=expected_layer,
    )
    table = pd.read_csv(table_path)
    bounded_columns = (
        "warmth_direction_topic_cv",
        "comp_direction_topic_cv",
        "warmth_to_comp_topic_transfer",
        "comp_to_warmth_topic_transfer",
    )
    for column in bounded_columns:
        values = table[column].to_numpy(float)
        if not np.isfinite(values).all() or not ((0.0 <= values) & (values <= 1.0)).all():
            raise AssertionError(f"{table_path}: {column} must be finite and within [0, 1]")
    if not ((-1.0 <= table["cos_wc"]) & (table["cos_wc"] <= 1.0)).all():
        raise AssertionError(f"{table_path}: cos_wc must be within [-1, 1]")
    for metric in ("warmth_cohens_d", "comp_cohens_d", "cos_wc"):
        low = table[f"{metric}_ci_low"].to_numpy(float)
        high = table[f"{metric}_ci_high"].to_numpy(float)
        if not np.isfinite(low).all() or not np.isfinite(high).all() or not (low <= high).all():
            raise AssertionError(f"{table_path}: invalid bootstrap interval for {metric}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("analysis_profile") != "stage3b":
        raise AssertionError(f"{meta_path}: analysis_profile must be 'stage3b'")
    if meta.get("seed") != 20260527 or meta.get("n_bootstrap") != 1000:
        raise AssertionError(f"{meta_path}: unexpected Stage 3B seed/bootstrap count")
    for key in ("git_commit", "stimuli_sha256"):
        value = meta.get(key)
        if not isinstance(value, str) or not value:
            raise AssertionError(f"{meta_path}: missing {key}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("analysis_profile") != "stage3b" or audit.get("n_layers") != expected_layers:
        raise AssertionError(f"{audit_path}: Stage 3B profile/layer mismatch")
    folds_by_layer = audit.get("folds_by_layer", {})
    if set(folds_by_layer) != {str(i) for i in range(expected_layers)}:
        raise AssertionError(f"{audit_path}: incomplete folds_by_layer")
    for layer_folds in folds_by_layer.values():
        if set(layer_folds) != set(bounded_columns):
            raise AssertionError(f"{audit_path}: incomplete fold metrics")
        if any(len(scores) != 5 for scores in layer_folds.values()):
            raise AssertionError(f"{audit_path}: every metric must contain five folds")
    bootstrap = audit.get("bootstrap", {})
    if bootstrap.get("n_bootstrap") != 1000 or bootstrap.get("n_topics") != 50:
        raise AssertionError(f"{audit_path}: unexpected bootstrap design")
    for metric in ("warmth_cohens_d", "comp_cohens_d", "cos_wc"):
        summary = bootstrap.get("peaks", {}).get(metric, {})
        probabilities = np.asarray(summary.get("layer_probabilities", []), dtype=float)
        if probabilities.shape != (expected_layers,) or not np.isclose(probabilities.sum(), 1.0):
            raise AssertionError(f"{audit_path}: invalid peak probabilities for {metric}")


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
    audit_path = logs_dir / f"validate_layer_sweep_{args.label}.json"
    absent_targets = targets
    if args.stage == 3 and args.analysis_profile == "stage3b":
        absent_targets = (*targets, audit_path)
    if args.require_absent:
        require_targets_absent(absent_targets)
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
        if args.analysis_profile == "stage3b":
            validate_stage3b(targets[0], targets[1], audit_path, **common)
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
    parser.add_argument(
        "--analysis-profile", choices=("legacy", "stage3b"), default="legacy"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
