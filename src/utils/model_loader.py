from __future__ import annotations

from importlib import metadata

import torch

from src.utils.config import ProjectConfig, require_model_name


def load_hooked_model(config: ProjectConfig):
    """Load raw Hugging Face weights through TransformerLens 3 Bridge.

    Full compatibility mode is intentionally not enabled: Gemma 4's PLE/MoE
    topology is not safe for legacy weight folding.  The returned bridge retains
    the HookedTransformer-style run_with_cache/run_with_hooks interface and hook
    aliases used by the pipeline.
    """
    model_name = require_model_name(config)
    if config.model.backend != "transformer-bridge":
        raise ValueError(
            "model.backend must be 'transformer-bridge' for production probing; "
            f"got {config.model.backend!r}."
        )

    try:
        from transformer_lens.model_bridge import TransformerBridge
    except ImportError as exc:
        raise ImportError(
            "Install transformer-lens>=3.5.1 before loading TransformerBridge models."
        ) from exc

    dtype = getattr(torch, config.model.dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype {config.model.dtype!r}.")
    return TransformerBridge.boot_transformers(
        model_name,
        device=config.model.device,
        dtype=dtype,
    )


def model_runtime_metadata(model) -> dict[str, str]:
    """Return reproducibility metadata without importing optional packages."""

    def version(package: str) -> str:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return "not-installed"

    parameters = iter(model.parameters())
    first_parameter = next(parameters, None)
    if first_parameter is None:
        first_parameter = next(model.original_model.parameters())
    return {
        "backend": "transformer-bridge",
        "transformer_lens_version": version("transformer-lens"),
        "transformers_version": version("transformers"),
        "torch_version": torch.__version__,
        "dtype": str(first_parameter.dtype),
    }
