"""Compute tracked same-story agreement metrics for the Qwen3.6 full runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

CONDITIONS = ("high_warmth", "low_warmth", "high_competence", "low_competence")
AXES = ("warmth", "competence")


def load_projections(vec_dir: Path, axis: str) -> list[np.ndarray]:
    """Load condition activations and project them onto a unit concept vector."""
    vector = np.load(vec_dir / f"{axis}_vec.npy").astype(np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Invalid {axis} vector in {vec_dir}")
    vector /= norm

    projections: list[np.ndarray] = []
    for condition in CONDITIONS:
        path = vec_dir / f"X_{condition}.npy"
        activations = np.load(path).astype(np.float64)
        if activations.ndim != 2 or len(activations) == 0:
            raise ValueError(f"Invalid activation matrix: {path}")
        if not np.isfinite(activations).all():
            raise ValueError(f"Non-finite activation matrix: {path}")
        projections.append(activations @ vector)
    if len({len(values) for values in projections}) != 1:
        raise ValueError(f"Condition row counts differ in {vec_dir}")
    return projections


def agreement_records(
    vec_dirs: list[Path], labels: list[str]
) -> list[dict[str, str | int | float]]:
    """Return overall, within-condition, and condition-specific Spearman metrics."""
    if len(vec_dirs) != len(labels) or len(vec_dirs) < 2:
        raise ValueError("vec_dirs and labels must have equal length >= 2")

    projected = {
        axis: [load_projections(vec_dir, axis) for vec_dir in vec_dirs]
        for axis in AXES
    }
    records: list[dict[str, str | int | float]] = []
    for axis in AXES:
        for left in range(len(vec_dirs)):
            for right in range(left + 1, len(vec_dirs)):
                values_left = projected[axis][left]
                values_right = projected[axis][right]
                record: dict[str, str | int | float] = {
                    "axis": axis,
                    "model_a": labels[left],
                    "model_b": labels[right],
                    "n_stories": sum(map(len, values_left)),
                    "n_per_condition": len(values_left[0]),
                    "overall_rho": round(
                        float(
                            spearmanr(
                                np.concatenate(values_left),
                                np.concatenate(values_right),
                            ).statistic
                        ),
                        6,
                    ),
                    "within_condition_rho": round(
                        float(
                            spearmanr(
                                np.concatenate([rankdata(v) for v in values_left]),
                                np.concatenate([rankdata(v) for v in values_right]),
                            ).statistic
                        ),
                        6,
                    ),
                }
                for condition, a, b in zip(CONDITIONS, values_left, values_right):
                    record[f"{condition}_rho"] = round(
                        float(spearmanr(a, b).statistic), 6
                    )
                records.append(record)
    return records


def write_records(records: list[dict[str, str | int | float]], output: Path) -> None:
    """Write comparison records with a stable schema."""
    if not records:
        raise ValueError("No agreement records to write")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vec-dirs", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    vec_dirs = [Path(value.strip()) for value in args.vec_dirs.split(",")]
    labels = [value.strip() for value in args.labels.split(",")]
    records = agreement_records(vec_dirs, labels)
    write_records(records, args.output)
    for record in records:
        print(
            f"{record['axis']}: overall={record['overall_rho']:.3f}, "
            f"within={record['within_condition_rho']:.3f}"
        )


if __name__ == "__main__":
    main()
