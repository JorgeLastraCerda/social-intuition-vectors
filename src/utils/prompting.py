"""Raw-passage and native-chat tokenization for probing experiments."""

from __future__ import annotations

from typing import Literal

import torch

PromptFormat = Literal["raw", "native-chat"]


def encode_passage(model, text: str) -> torch.Tensor:
    """Encode passive stimulus text exactly as in the original extraction runs."""
    return model.to_tokens(text, prepend_bos=True)


def render_decision_prompt(model, prompt: str, prompt_format: PromptFormat) -> str:
    if prompt_format == "raw":
        return prompt
    if prompt_format != "native-chat":
        raise ValueError(f"Unknown prompt format {prompt_format!r}.")

    processor = getattr(model, "processor", None)
    if processor is None or not hasattr(processor, "apply_chat_template"):
        raise ValueError(
            "native-chat requires a model processor with apply_chat_template()."
        )
    return processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_decision_prompt(
    model,
    prompt: str,
    prompt_format: PromptFormat,
) -> tuple[str, torch.Tensor]:
    rendered = render_decision_prompt(model, prompt, prompt_format)
    # Native templates already contain the model's BOS token.
    tokens = model.to_tokens(rendered, prepend_bos=prompt_format == "raw")
    return rendered, tokens


def candidate_token_id(model, candidate: str) -> int:
    """Legacy raw-prompt candidate check retained for existing experiments."""
    tokens = model.to_tokens(candidate, prepend_bos=False)
    if tokens.numel() != 1:
        raise ValueError(
            f"Candidate {candidate!r} must tokenize to exactly one token; "
            f"got shape {tuple(tokens.shape)}."
        )
    return int(tokens.item())


def decision_token_ids(
    model,
    rendered_prompt: str,
    prompt_format: PromptFormat,
) -> tuple[int, int]:
    """Resolve one-token Yes/No continuations relative to the exact prompt prefix."""
    if prompt_format == "raw":
        return candidate_token_id(model, " Yes"), candidate_token_id(model, " No")

    prefix = model.to_tokens(rendered_prompt, prepend_bos=False).reshape(-1)

    def continuation_id(candidate: str) -> int:
        combined = model.to_tokens(
            rendered_prompt + candidate,
            prepend_bos=False,
        ).reshape(-1)
        if combined.numel() <= prefix.numel() or not torch.equal(
            combined[: prefix.numel()], prefix
        ):
            raise ValueError(
                f"Adding {candidate!r} changed the native-chat token prefix."
            )
        continuation = combined[prefix.numel() :]
        if continuation.numel() != 1:
            raise ValueError(
                f"Native-chat continuation {candidate!r} must be one token; "
                f"got {continuation.numel()}."
            )
        return int(continuation.item())

    return continuation_id("Yes"), continuation_id("No")
