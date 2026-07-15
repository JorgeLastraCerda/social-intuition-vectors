from __future__ import annotations

from importlib import metadata

import torch

from src.utils.config import ProjectConfig, require_model_name


def load_hooked_model(
    config: ProjectConfig,
    *,
    n_devices: int | None = None,
):
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
    if n_devices is not None and n_devices < 1:
        raise ValueError("n_devices must be at least 1 when provided.")

    load_kwargs: dict[str, object] = {"dtype": dtype}
    if n_devices is not None and n_devices > 1:
        # TransformerBridge resolves this to Accelerate's balanced device map.
        # Do not also pass ``device``: dispatched models must retain their map.
        load_kwargs["n_devices"] = n_devices
    else:
        load_kwargs["device"] = config.model.device

    model = TransformerBridge.boot_transformers(model_name, **load_kwargs)
    if model_name.startswith("google/gemma-4-") and not hasattr(
        getattr(model, "processor", None), "apply_chat_template"
    ):
        try:
            from transformers import AutoProcessor

            model.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the required Gemma 4 processor for {model_name!r}."
            ) from exc
        if not hasattr(model.processor, "apply_chat_template"):
            raise RuntimeError(
                f"Processor for {model_name!r} has no apply_chat_template()."
            )
    return model


def model_runtime_metadata(model) -> dict[str, object]:
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
    original_model = getattr(model, "original_model", model)
    raw_device_map = getattr(original_model, "hf_device_map", None)
    device_map = None
    if raw_device_map:
        device_map = {str(key): str(value) for key, value in raw_device_map.items()}

    parameter_devices = sorted(
        {str(parameter.device) for parameter in original_model.parameters()}
    )
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            {"index": index, "name": torch.cuda.get_device_name(index)}
            for index in range(torch.cuda.device_count())
        ]

    return {
        "backend": "transformer-bridge",
        "transformer_lens_version": version("transformer-lens"),
        "transformers_version": version("transformers"),
        "torch_version": torch.__version__,
        "dtype": str(first_parameter.dtype),
        "execution_topology": "dispatched" if device_map else "single-device",
        "parameter_devices": parameter_devices,
        "hf_device_map": device_map,
        "visible_cuda_devices": cuda_devices,
    }
