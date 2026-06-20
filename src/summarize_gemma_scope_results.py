from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.gemma_scope_utils import check_file_size
from src.utils.config import load_config


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = [Path(value.strip()) for value in args.causality_csvs.split(",")]
    labels = [value.strip() for value in args.labels.split(",")]
    if len(paths) != len(labels):
        raise ValueError("--causality-csvs and --labels must have equal lengths.")

    output_rows: list[dict] = []
    for path, label in zip(paths, labels):
        rows = [
            row
            for row in load_rows(path)
            if row["mode"] == "steering"
        ]
        keys = sorted({(row["axis"], row["direction"]) for row in rows})
        for axis, direction in keys:
            selected = [
                row
                for row in rows
                if row["axis"] == axis and row["direction"] == direction
            ]
            strengths = np.array(
                [float(row["strength"]) for row in selected],
                dtype=np.float64,
            )
            effects = np.array(
                [float(row["effect"]) for row in selected],
                dtype=np.float64,
            )
            slope, intercept = np.polyfit(strengths, effects, 1)
            fitted = slope * strengths + intercept
            residual_ss = float(np.square(effects - fitted).sum())
            total_ss = float(np.square(effects - effects.mean()).sum())
            output_rows.append(
                {
                    "label": label,
                    "axis": axis,
                    "direction": direction,
                    "local_slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": (
                        1.0 - residual_ss / total_ss
                        if total_ss > 0
                        else float("nan")
                    ),
                    "min_strength": float(strengths.min()),
                    "max_strength": float(strengths.max()),
                    "n_strengths": len(strengths),
                }
            )

    output_path = (
        Path(cfg.paths.results)
        / "tables"
        / "gemma_scope_local_steering_slopes.csv"
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    check_file_size(output_path)
    print(f"[done] {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize local-regime Gemma Scope steering slopes."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--causality-csvs", required=True)
    parser.add_argument("--labels", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
