"""Summarize hiring steering runs and pre-register the conditional 282-name gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config

AXES = ("warmth", "competence")


def paired_bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int,
    n_boot: int = 5000,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError(
            "paired_bootstrap_ci requires a non-empty one-dimensional array"
        )
    rng = np.random.default_rng(seed)
    estimates = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(
        axis=1
    )
    return (
        float(values.mean()),
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def margin_diagnostic(values: np.ndarray) -> dict[str, float | int | bool]:
    values = np.asarray(values, dtype=np.float64)
    fraction = float(np.isclose(values * 8.0, np.round(values * 8.0), atol=1e-6).mean())
    sd = float(values.std(ddof=1))
    return {
        "n": int(len(values)),
        "n_unique": int(len(np.unique(values))),
        "sd": sd,
        "fraction_on_0.125_grid": fraction,
        "quantisation_warning": bool(fraction > 0.8 and sd < 0.25),
    }


def summarize_frame(
    frame: pd.DataFrame,
    *,
    seed: int,
    n_boot: int = 5000,
) -> tuple[pd.DataFrame, dict]:
    required = {"axis", "strength", "name", "margin", "delta"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing hiring steering columns: {sorted(missing)}")
    if frame.duplicated(["axis", "strength", "name"]).any():
        raise ValueError("Duplicate axis/strength/name rows in hiring steering input")
    rows: list[dict] = []
    axis_diagnostics: dict[str, dict] = {}
    for axis_index, axis in enumerate(AXES):
        subset = frame[frame["axis"] == axis]
        strengths = sorted(float(value) for value in subset["strength"].unique())
        if 0.0 not in strengths:
            raise ValueError(f"{axis}: strengths must contain zero")
        means: list[float] = []
        for strength_index, strength in enumerate(strengths):
            values = subset[subset["strength"] == strength]["delta"].to_numpy(float)
            estimate, ci_low, ci_high = paired_bootstrap_ci(
                values,
                seed=seed + axis_index * 100 + strength_index,
                n_boot=n_boot,
            )
            means.append(estimate)
            rows.append(
                {
                    "axis": axis,
                    "strength": strength,
                    "mean_delta": estimate,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "n_names": int(len(values)),
                    "n_boot": n_boot,
                }
            )
        x = np.asarray(strengths, dtype=np.float64)
        y = np.asarray(means, dtype=np.float64)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else 1.0 - ss_res / ss_tot
        positive_endpoint = max(strengths)
        endpoint_row = next(
            row
            for row in rows
            if row["axis"] == axis and row["strength"] == positive_endpoint
        )
        monotone = bool(np.all(np.diff(y) >= -1e-12))
        axis_diagnostics[axis] = {
            "slope": float(slope),
            "r_squared": float(r_squared),
            "monotone_non_decreasing": monotone,
            "positive_endpoint": positive_endpoint,
            "endpoint_mean_delta": float(endpoint_row["mean_delta"]),
            "endpoint_ci_95_low": float(endpoint_row["ci_95_low"]),
            "endpoint_ci_95_high": float(endpoint_row["ci_95_high"]),
            "endpoint_sign_matches_slope": bool(
                np.sign(endpoint_row["mean_delta"]) == np.sign(slope)
                or endpoint_row["mean_delta"] == 0.0
                or slope == 0.0
            ),
        }
    zero = frame[frame["strength"] == 0.0]
    if not np.array_equal(zero["delta"].to_numpy(float), np.zeros(len(zero))):
        raise ValueError("Zero-strength deltas must be exactly zero")
    baseline = zero.drop_duplicates("name")["margin"].to_numpy(float)
    return pd.DataFrame(rows), {
        "axes": axis_diagnostics,
        "margin_diagnostic": margin_diagnostic(baseline),
    }


def gate_reasons(summaries: dict[str, dict]) -> list[str]:
    reasons: list[str] = []
    for regime, payload in summaries.items():
        for axis, diagnostic in payload["axes"].items():
            if (
                diagnostic["endpoint_ci_95_low"]
                <= 0.0
                <= diagnostic["endpoint_ci_95_high"]
            ):
                reasons.append(f"{regime}:{axis}:endpoint_ci_includes_zero")
            if diagnostic["r_squared"] < 0.50:
                reasons.append(f"{regime}:{axis}:r_squared_below_0.50")
            if not diagnostic["endpoint_sign_matches_slope"]:
                reasons.append(f"{regime}:{axis}:endpoint_slope_sign_mismatch")
            if not diagnostic["monotone_non_decreasing"]:
                reasons.append(f"{regime}:{axis}:non_monotone")
    if "local" in summaries and "denoised_local" in summaries:
        for axis in AXES:
            raw = summaries["local"]["axes"][axis]["endpoint_mean_delta"]
            denoised = summaries["denoised_local"]["axes"][axis]["endpoint_mean_delta"]
            if raw != 0.0 and denoised != 0.0 and np.sign(raw) != np.sign(denoised):
                reasons.append(f"raw_vs_denoised:{axis}:endpoint_sign_reversal")
    return reasons


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    tables = Path(cfg.paths.results) / "tables"
    logs = Path(cfg.paths.logs)
    tables.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    if args.command == "summarize":
        raw_path = tables / f"hiring_steering_raw_{args.label}.csv"
        frame = pd.read_csv(raw_path)
        summary, diagnostics = summarize_frame(
            frame, seed=cfg.probing.seed, n_boot=args.n_boot
        )
        summary_path = tables / f"hiring_steering_{args.label}.csv"
        log_path = logs / f"hiring_steering_summary_{args.label}.json"
        summary.to_csv(summary_path, index=False)
        log_path.write_text(
            json.dumps(
                {
                    "label": args.label,
                    "seed": cfg.probing.seed,
                    "raw_input": str(raw_path),
                    "summary_output": str(summary_path),
                    **diagnostics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[done] {summary_path}\n[done] {log_path}")
        return

    regimes = {
        regime: json.loads(
            (
                logs / f"hiring_steering_summary_{args.model_label}_{suffix}.json"
            ).read_text(encoding="utf-8")
        )
        for regime, suffix in (
            ("local", "local"),
            ("broad", "broad"),
            ("denoised_local", "denoised_local"),
        )
    }
    reasons = gate_reasons(regimes)
    output = logs / f"hiring_full282_gate_{args.model_label}.json"
    output.write_text(
        json.dumps(
            {
                "model_label": args.model_label,
                "run_full_282": bool(reasons),
                "reasons": reasons,
                "policy": "run all local, broad, and denoised-local regimes if any reason fires",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[gate] run_full_282={bool(reasons)} reasons={len(reasons)} -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--config", default="config/config.yaml")
    summarize.add_argument("--label", required=True)
    summarize.add_argument("--n-boot", type=int, default=5000)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--config", default="config/config.yaml")
    gate.add_argument("--model-label", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
