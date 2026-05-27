from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from openai import OpenAI

from scripts.config import ExperimentConfig


Provider = Literal["openai", "ollama"]


@dataclass(frozen=True)
class ModelSpec:
    provider: Provider
    model: str


class LLMRouter:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.openai_client = OpenAI(
            api_key=config.openai_api_key,
            timeout=config.timeout_seconds,
        )
        self.ollama_client = OpenAI(
            api_key="ollama",
            base_url=config.ollama_base_url,
            timeout=config.timeout_seconds,
        )

    def complete(self, spec: ModelSpec, messages: list[dict[str, str]]) -> str:
        if spec.provider == "openai":
            return self._complete_openai(spec.model, messages)
        if spec.provider == "ollama":
            return self._complete_ollama(spec.model, messages)
        raise ValueError(f"Unsupported provider: {spec.provider}")

    def _complete_openai(self, model: str, messages: list[dict[str, str]]) -> str:
        response = self.openai_client.responses.create(
            model=model,
            input=messages,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            store=False,
        )
        return response.output_text

    def _complete_ollama(self, model: str, messages: list[dict[str, str]]) -> str:
        response = self.ollama_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
        )
        return response.choices[0].message.content or ""


def resolve_model_specs(config: ExperimentConfig, requested: list[str]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for item in requested:
        if item == "openai":
            specs.append(ModelSpec(provider="openai", model=config.openai_model))
        elif item == "ollama":
            specs.append(ModelSpec(provider="ollama", model=config.ollama_model))
        elif item.startswith("openai:"):
            specs.append(ModelSpec(provider="openai", model=item.split(":", 1)[1]))
        elif item.startswith("ollama:"):
            specs.append(ModelSpec(provider="ollama", model=item.split(":", 1)[1]))
        else:
            raise ValueError(f"Unknown model spec {item!r}. Use openai, ollama, openai:<model>, or ollama:<model>.")
    return specs
