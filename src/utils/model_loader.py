from __future__ import annotations

from src.utils.config import ProjectConfig, require_model_name


def load_hooked_model(config: ProjectConfig):
    """Load the configured open-weights model with TransformerLens.

    This function intentionally stays small: all model identity and device choices
    come from config, and model loading should fail loudly if the environment is
    not ready.
    """
    model_name = require_model_name(config)

    try:
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise ImportError("Install transformer-lens before loading hooked models.") from exc

    return HookedTransformer.from_pretrained(
        model_name,
        device=config.model.device,
        dtype=config.model.dtype,
    )
