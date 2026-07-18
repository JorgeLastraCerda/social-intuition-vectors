from __future__ import annotations

import hashlib
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
    if config.model.revision:
        load_kwargs["revision"] = config.model.revision
    if n_devices is not None and n_devices > 1:
        # TransformerBridge resolves this to Accelerate's balanced device map.
        # Do not also pass ``device``: dispatched models must retain their map.
        load_kwargs["n_devices"] = n_devices
    else:
        load_kwargs["device"] = config.model.device

    pinned_processor = None
    if model_name.startswith("google/gemma-4-") and config.model.revision:
        try:
            from transformers import AutoProcessor, AutoTokenizer

            pinned_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=config.model.revision,
                add_bos_token=True,
            )
            pinned_processor = AutoProcessor.from_pretrained(
                model_name,
                revision=config.model.revision,
            )
            load_kwargs["tokenizer"] = pinned_tokenizer
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pinned Gemma 4 tokenizer/processor for {model_name!r} "
                f"at revision {config.model.revision!r}."
            ) from exc

    model = TransformerBridge.boot_transformers(model_name, **load_kwargs)
    if pinned_processor is not None:
        model.processor = pinned_processor
    elif model_name.startswith("google/gemma-4-") and not hasattr(
        getattr(model, "processor", None), "apply_chat_template"
    ):
        try:
            from transformers import AutoProcessor

            model.processor = AutoProcessor.from_pretrained(
                model_name,
                revision=config.model.revision,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the required Gemma 4 processor for {model_name!r}."
            ) from exc
    if model_name.startswith("google/gemma-4-"):
        if not hasattr(model.processor, "apply_chat_template"):
            raise RuntimeError(
                f"Processor for {model_name!r} has no apply_chat_template()."
            )

    original_model = getattr(model, "original_model", model)
    resolved_revision = getattr(
        getattr(original_model, "config", None), "_commit_hash", None
    )
    if config.model.revision and resolved_revision != config.model.revision:
        raise RuntimeError(
            f"Resolved model revision mismatch for {model_name!r}: requested "
            f"{config.model.revision!r}, resolved {resolved_revision!r}."
        )
    template = getattr(getattr(model, "processor", None), "chat_template", None)
    if template is None:
        template = getattr(getattr(model, "tokenizer", None), "chat_template", None)
    model._normalcy_revision_requested = config.model.revision
    model._normalcy_revision_resolved = resolved_revision
    model._normalcy_chat_template_sha256 = (
        hashlib.sha256(str(template).encode("utf-8")).hexdigest()
        if template is not None
        else None
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
        "model_revision_requested": getattr(
            model, "_normalcy_revision_requested", None
        ),
        "model_revision_resolved": getattr(model, "_normalcy_revision_resolved", None),
        "chat_template_sha256": getattr(model, "_normalcy_chat_template_sha256", None),
        "peak_allocated_vram_gib": (
            float(torch.cuda.max_memory_allocated() / 1024**3)
            if torch.cuda.is_available()
            else None
        ),
        "peak_reserved_vram_gib": (
            float(torch.cuda.max_memory_reserved() / 1024**3)
            if torch.cuda.is_available()
            else None
        ),
    }
