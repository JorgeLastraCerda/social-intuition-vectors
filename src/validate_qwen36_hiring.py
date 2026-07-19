"""Validate native-HF Qwen3.6 hiring audit and steering artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config


def artifact_paths(config: str, task: str, label: str) -> tuple[Path, ...]:
    cfg = load_config(config)
    tables = Path(cfg.paths.results) / "tables"
    logs = Path(cfg.paths.logs)
    if task == "audit":
        return (
            tables / f"hiring_audit_{label}.csv",
            logs / f"hiring_probe_vs_human_{label}.json",
        )
    if task == "neutral":
        vectors = Path(cfg.paths.processed) / f"concept_vectors_{cfg.native_hf.label}"
        return vectors / "X_neutral.npy", vectors / "neutral_meta.json"
    return (
        tables / f"hiring_steering_raw_{label}.csv",
        logs / f"hiring_steering_{label}.json",
    )


def validate(
    config: str,
    task: str,
    label: str,
    *,
    require_absent: bool = False,
    n_names: int = 60,
) -> dict:
    cfg = load_config(config)
    paths = artifact_paths(config, task, label)
    if require_absent:
        collisions = [str(path) for path in paths if path.exists()]
        if collisions:
            raise FileExistsError(
                f"Refusing existing Qwen hiring outputs: {collisions}"
            )
        return {"status": "absent", "task": task, "label": label}
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Qwen hiring outputs: {missing}")
    meta = json.loads(paths[1].read_text(encoding="utf-8"))
    if (
        meta.get("model") != cfg.model.name
        or meta.get("revision") != cfg.model.revision
    ):
        raise AssertionError("Qwen hiring model or revision mismatch.")
    runtime = meta.get("runtime", {})
    if runtime.get("model_revision_resolved") != cfg.model.revision:
        raise AssertionError("Qwen hiring resolved revision mismatch.")
    if meta.get("transformer_lens_imported") is not False:
        if task != "neutral":
            raise AssertionError("Native-HF Qwen hiring imported TransformerLens.")
    if task == "neutral":
        matrix = np.load(paths[0], mmap_mode="r")
        expected = (cfg.neutral.n_texts, cfg.native_hf.expected_d_model)
        if matrix.shape != expected or not np.isfinite(matrix).all():
            raise AssertionError(
                f"Qwen neutral matrix failed contract: {matrix.shape}."
            )
        if meta.get("seed") != cfg.probing.seed:
            raise AssertionError("Qwen neutral seed mismatch.")
        return {
            "status": "pass",
            "task": task,
            "label": label,
            "rows": matrix.shape[0],
            "d_model": matrix.shape[1],
        }
    table = pd.read_csv(paths[0])
    if task == "audit":
        required = {
            "name",
            "human_warm",
            "human_competent",
            "model_warmth",
            "model_competence",
            "callback_margin",
        }
        if len(table) != 282 or table["name"].nunique() != 282:
            raise AssertionError("Qwen hiring audit must contain 282 unique names.")
        if not required.issubset(table.columns):
            raise AssertionError(f"Qwen audit missing columns: {required - set(table)}")
        numeric = table[[column for column in required if column != "name"]]
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise FloatingPointError("Qwen hiring audit contains non-finite values.")
        if len(meta.get("correlations", [])) != 6:
            raise AssertionError("Qwen hiring audit must report six correlations.")
        return {"status": "pass", "task": task, "label": label, "rows": 282}
    expected = n_names * 2 * 5
    if len(table) != expected or table["name"].nunique() != n_names:
        raise AssertionError(
            f"Qwen hiring steering expected {expected} rows/{n_names} names; "
            f"got {len(table)}/{table['name'].nunique()}."
        )
    required = {"axis", "strength", "name", "margin", "delta"}
    if not required.issubset(table.columns):
        raise AssertionError(f"Qwen steering missing columns: {required - set(table)}")
    if not np.isfinite(table[["strength", "margin", "delta"]].to_numpy()).all():
        raise FloatingPointError("Qwen hiring steering contains non-finite values.")
    zero = table[table["strength"] == 0.0]["delta"].to_numpy(float)
    if not np.array_equal(zero, np.zeros_like(zero)):
        raise AssertionError("Qwen zero-strength hiring deltas are not exact zero.")
    return {
        "status": "pass",
        "task": task,
        "label": label,
        "rows": expected,
        "names": n_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--task", choices=("audit", "steering", "neutral"), required=True
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--n-names", type=int, default=60)
    parser.add_argument("--require-absent", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                args.config,
                args.task,
                args.label,
                require_absent=args.require_absent,
                n_names=args.n_names,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
