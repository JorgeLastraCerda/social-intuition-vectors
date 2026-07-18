"""Shared calibration and intervention primitives for concept steering.

Calibration statistics are estimated from training-topic activations only.  The
legacy target-axis intervention remains ``strength * mean_residual_norm``;
controls are rescaled to produce the same standardized projection shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

ControlScale = Literal["legacy_l2", "sd_matched"]
Intervention = Literal["additive", "norm_preserving"]


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Direction must have a finite positive norm.")
    return vector / norm


def directional_sd(activations: np.ndarray, direction: np.ndarray) -> float:
    """Sample SD of training activations projected onto a unit direction."""
    matrix = np.asarray(activations, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("activations must contain at least two rows.")
    if matrix.shape[1] != np.asarray(direction).size:
        raise ValueError("Activation width and direction width do not match.")
    value = float(np.std(matrix @ unit(direction), ddof=1))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Directional projection SD must be finite and positive.")
    return value


def calibrated_alpha(
    *,
    strength: float,
    mean_residual_norm: float,
    target_direction_sd: float,
    direction_sd: float,
    control_scale: ControlScale,
) -> float:
    """Return absolute alpha for legacy or target-SD-matched steering."""
    base = float(strength) * float(mean_residual_norm)
    if control_scale == "legacy_l2":
        return base
    if control_scale != "sd_matched":
        raise ValueError(f"Unknown control scale {control_scale!r}.")
    if min(target_direction_sd, direction_sd) <= 0:
        raise ValueError("Directional SDs must be positive.")
    return base * float(direction_sd) / float(target_direction_sd)


def standardized_shift(alpha: float, direction_sd: float) -> float:
    """Express the intervention displacement in within-direction SD units."""
    if direction_sd <= 0:
        raise ValueError("direction_sd must be positive.")
    return float(alpha) / float(direction_sd)


@dataclass
class NormDiagnostics:
    """Maximum observed token-level relative norm change for one hook."""

    max_relative_norm_drift: float = 0.0


def intervene_tensor(
    residual: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
    intervention: Intervention,
    diagnostics: NormDiagnostics | None = None,
) -> torch.Tensor:
    """Apply an additive or per-token norm-preserving displacement."""
    if intervention not in ("additive", "norm_preserving"):
        raise ValueError(f"Unknown intervention {intervention!r}.")
    direction = direction.to(device=residual.device, dtype=residual.dtype)
    direction = direction / direction.float().norm().clamp_min(1e-12).to(
        direction.dtype
    )
    original_norm = residual.float().norm(dim=-1, keepdim=True)
    result = residual + float(alpha) * direction
    if intervention == "norm_preserving":
        changed_norm = result.float().norm(dim=-1, keepdim=True)
        scale = torch.where(
            original_norm > 0,
            original_norm / changed_norm.clamp_min(1e-12),
            torch.ones_like(original_norm),
        )
        result = result * scale.to(result.dtype)
    if diagnostics is not None:
        final_norm = result.float().norm(dim=-1, keepdim=True)
        relative = torch.where(
            original_norm > 0,
            (final_norm - original_norm).abs() / original_norm,
            torch.zeros_like(original_norm),
        )
        diagnostics.max_relative_norm_drift = max(
            diagnostics.max_relative_norm_drift, float(relative.max().item())
        )
    return result


def make_torch_hook(
    direction: np.ndarray,
    alpha: float,
    intervention: Intervention,
) -> tuple[object, NormDiagnostics]:
    """Build a hook usable by TransformerLens and plain PyTorch modules."""
    direction_tensor = torch.from_numpy(unit(direction).astype(np.float32))
    diagnostics = NormDiagnostics()

    def hook(residual: torch.Tensor, hook=None) -> torch.Tensor:  # noqa: ARG001
        return intervene_tensor(
            residual, direction_tensor, alpha, intervention, diagnostics
        )

    return hook, diagnostics


def percentile_rank(null: np.ndarray, observed: float) -> float:
    """Finite-sample lower-tail percentile using the plus-one convention."""
    values = np.asarray(null, dtype=np.float64)
    if values.size == 0:
        raise ValueError("The null distribution is empty.")
    return float((1 + np.sum(values <= observed)) / (values.size + 1))


def descriptive_null_metrics(null: np.ndarray, observed: float) -> dict[str, float]:
    """Signed and magnitude percentiles without a scientific pass/fail gate."""
    values = np.asarray(null, dtype=np.float64)
    return {
        "signed_percentile": percentile_rank(values, observed),
        "absolute_percentile": percentile_rank(np.abs(values), abs(observed)),
        "random_median": float(np.median(values)),
        "target_minus_random_median": float(observed - np.median(values)),
    }


def paired_topic_difference_ci(
    raw_rows: list[dict[str, Any]],
    *,
    judgment_axis: str,
    steering_axis: str,
    intervention: str,
    endpoint_strength: float,
    seed: int,
    n_boot: int = 5000,
) -> tuple[float, float, float]:
    """Bootstrap target-minus-random-median endpoint effects by held-out topic."""
    selected = [
        row
        for row in raw_rows
        if row.get("mode") == "steering"
        and row.get("axis") == judgment_axis
        and row.get("intervention") == intervention
        and float(row.get("strength", 0.0)) == endpoint_strength
    ]
    topics = sorted({int(row["topic_idx"]) for row in selected})
    differences: list[float] = []
    for topic in topics:
        target_values = [
            float(row["delta_margin"])
            for row in selected
            if int(row["topic_idx"]) == topic and row["direction"] == steering_axis
        ]
        random_names = sorted(
            {
                str(row["direction"])
                for row in selected
                if int(row["topic_idx"]) == topic
                and str(row["direction"]).startswith("random_")
            }
        )
        random_effects = [
            float(
                np.mean(
                    [
                        float(row["delta_margin"])
                        for row in selected
                        if int(row["topic_idx"]) == topic and row["direction"] == name
                    ]
                )
            )
            for name in random_names
        ]
        if not target_values or not random_effects:
            raise ValueError(f"Incomplete topic-paired null data for topic {topic}.")
        differences.append(float(np.mean(target_values) - np.median(random_effects)))
    values = np.asarray(differences, dtype=np.float64)
    if values.size < 2:
        raise ValueError("At least two held-out topics are required for bootstrap.")
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )
