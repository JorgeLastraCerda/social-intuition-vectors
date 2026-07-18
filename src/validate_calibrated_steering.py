"""Validate write-once calibrated-steering artifacts and technical invariants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.utils.config import load_config


def artifact_paths(config: str, label: str) -> tuple[Path, Path, Path, Path]:
    cfg = load_config(config)
    tables = Path(cfg.paths.results) / "tables"
    logs = Path(cfg.paths.logs)
    return (
        tables / f"steering_dense_raw_{label}.csv",
        tables / f"steering_dense_{label}.csv",
        tables / f"steering_dense_null_{label}.csv",
        logs / f"steering_dense_{label}.json",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate(config: str, label: str, *, require_absent: bool = False) -> dict:
    paths = artifact_paths(config, label)
    if require_absent:
        collisions = [str(path) for path in paths if path.exists()]
        if collisions:
            raise FileExistsError(f"Refusing existing calibrated outputs: {collisions}")
        return {"status": "absent", "label": label}
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing calibrated outputs: {missing}")

    raw = read_csv(paths[0])
    summary = read_csv(paths[1])
    null = read_csv(paths[2])
    meta = json.loads(paths[3].read_text(encoding="utf-8"))
    required = {
        "direction_sd",
        "alpha_absolute",
        "standardized_shift",
        "control_scale",
        "intervention",
        "max_relative_norm_drift",
    }
    if not raw or not required.issubset(raw[0]):
        raise AssertionError(
            f"Raw output lacks calibrated fields: {required - set(raw[0])}"
        )
    if len(raw) != 40440:
        raise AssertionError(f"Expected 40,440 raw rows, got {len(raw)}.")
    if len(summary) != 2020:
        raise AssertionError(f"Expected 2,020 summary rows, got {len(summary)}.")
    if len(null) != 8:
        raise AssertionError(f"Expected 8 null rows, got {len(null)}.")
    if meta.get("control_scale") != "sd_matched":
        raise AssertionError("Calibration log must specify sd_matched.")
    if meta.get("n_random_directions") != 99:
        raise AssertionError("Calibration log must specify 99 random directions.")
    if meta.get("scientific_gate") != "descriptive-only":
        raise AssertionError("Scientific effects must remain descriptive-only.")
    if meta.get("transformer_lens_imported") not in (None, False):
        raise AssertionError("Native-HF run imported TransformerLens.")

    steering = [row for row in raw if row["mode"] == "steering"]
    numeric_fields = (
        "strength",
        "margin",
        "delta_margin",
        "direction_sd",
        "alpha_absolute",
        "standardized_shift",
        "max_relative_norm_drift",
    )
    for field in numeric_fields:
        values = np.asarray([float(row[field]) for row in steering])
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Non-finite {field} values.")
    for axis in ("warmth", "competence"):
        for intervention in ("additive", "norm_preserving"):
            for strength in (-0.1, -0.05, 0.0, 0.05, 0.1):
                shifts = np.asarray(
                    [
                        float(row["standardized_shift"])
                        for row in steering
                        if row["axis"] == axis
                        and row["intervention"] == intervention
                        and float(row["strength"]) == strength
                    ]
                )
                if shifts.size == 0 or float(np.ptp(shifts)) > 1e-5:
                    raise AssertionError(
                        f"Standardized shifts are not matched for {axis}/{intervention}/{strength}."
                    )
    norm_preserving_drift = max(
        float(row["max_relative_norm_drift"])
        for row in steering
        if row["intervention"] == "norm_preserving"
    )
    if norm_preserving_drift > 5e-3:
        raise AssertionError(
            f"Norm-preserving drift {norm_preserving_drift:.6g} exceeds 5e-3."
        )
    return {
        "status": "pass",
        "label": label,
        "raw_rows": len(raw),
        "summary_rows": len(summary),
        "null_rows": len(null),
        "max_norm_preserving_drift": norm_preserving_drift,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--require-absent", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.config, args.label, require_absent=args.require_absent),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
