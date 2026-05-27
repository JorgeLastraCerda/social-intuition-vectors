from __future__ import annotations

import torch


HOOK_BACKEND = "transformer-lens"


def layer_from_fraction(n_layers: int, fraction: float) -> int:
    if not 0 <= fraction <= 1:
        raise ValueError("Layer fraction must be between 0 and 1.")
    return min(n_layers - 1, max(0, round((n_layers - 1) * fraction)))


def residual_hook_name(layer_index: int) -> str:
    return f"blocks.{layer_index}.hook_resid_post"


def mean_activation_after_token(activations: torch.Tensor, start_token: int) -> torch.Tensor:
    if activations.ndim != 3:
        raise ValueError("Expected activations with shape [batch, seq, d_model].")
    if start_token >= activations.shape[1]:
        start_token = max(0, activations.shape[1] - 1)
    return activations[:, start_token:, :].mean(dim=1)


def add_steering_vector(residual: torch.Tensor, vector: torch.Tensor, strength: float) -> torch.Tensor:
    return residual + strength * vector.to(device=residual.device, dtype=residual.dtype)
