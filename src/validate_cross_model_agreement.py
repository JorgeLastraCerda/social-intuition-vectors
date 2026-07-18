from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


CONDITIONS = ("high_warmth", "low_warmth", "high_competence", "low_competence")
AXES = ("warmth", "competence")


def _load_projections(vec_dir: Path, axis: str) -> list[np.ndarray]:
    vector = np.load(vec_dir / f"{axis}_vec.npy").astype(np.float64)
    vector /= np.linalg.norm(vector) + 1e-12
    projections: list[np.ndarray] = []
    for condition in CONDITIONS:
        activations = np.load(vec_dir / f"X_{condition}.npy").astype(np.float64)
        if activations.ndim != 2 or len(activations) == 0:
            raise ValueError(f"Invalid activations in {vec_dir / f'X_{condition}.npy'}")
        if not np.isfinite(activations).all():
            raise ValueError(f"Non-finite activations in {vec_dir / f'X_{condition}.npy'}")
        projections.append(activations @ vector)
    if len({len(values) for values in projections}) != 1:
        raise ValueError(f"Condition row counts differ in {vec_dir}")
    return projections


def compute_agreement_records(
    vec_dirs: list[Path],
    model_labels: list[str],
) -> list[dict[str, str | int | float]]:
    if len(vec_dirs) != len(model_labels) or len(vec_dirs) < 2:
        raise ValueError("vec_dirs and model_labels must have the same length >= 2")

    projections = {
        axis: [_load_projections(vec_dir, axis) for vec_dir in vec_dirs]
        for axis in AXES
    }
    records: list[dict[str, str | int | float]] = []
    for axis in AXES:
        for model_a in range(len(vec_dirs)):
            for model_b in range(model_a + 1, len(vec_dirs)):
                values_a = projections[axis][model_a]
                values_b = projections[axis][model_b]
                condition_rhos = [
                    float(spearmanr(a, b).statistic)
                    for a, b in zip(values_a, values_b)
                ]
                overall_rho = float(
                    spearmanr(np.concatenate(values_a), np.concatenate(values_b)).statistic
                )
                within_rho = float(
                    spearmanr(
                        np.concatenate([rankdata(values) for values in values_a]),
                        np.concatenate([rankdata(values) for values in values_b]),
                    ).statistic
                )
                record: dict[str, str | int | float] = {
                    "axis": axis,
                    "model_a": model_labels[model_a],
                    "model_b": model_labels[model_b],
                    "n_stories": sum(len(values) for values in values_a),
                    "n_per_condition": len(values_a[0]),
                    "overall_rho": round(overall_rho, 6),
                    "within_condition_rho": round(within_rho, 6),
                }
                for condition, rho in zip(CONDITIONS, condition_rhos):
                    record[f"{condition}_rho"] = round(rho, 6)
                records.append(record)
    return records


def write_records(records: list[dict[str, str | int | float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate cross-model story agreement.")
    parser.add_argument("--vec-dirs", required=True, help="Comma-separated vector directories.")
    parser.add_argument("--labels", required=True, help="Comma-separated model labels.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vec_dirs = [Path(value.strip()) for value in args.vec_dirs.split(",")]
    labels = [value.strip() for value in args.labels.split(",")]
    records = compute_agreement_records(vec_dirs, labels)
    write_records(records, Path(args.output))
    for record in records:
        print(
            f"{record['axis']} {record['model_a']} vs {record['model_b']}: "
            f"overall={record['overall_rho']:.3f}, "
            f"within={record['within_condition_rho']:.3f}"
        )


if __name__ == "__main__":
    main()
