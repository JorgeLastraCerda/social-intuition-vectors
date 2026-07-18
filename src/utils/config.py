from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    dtype: str
    device: str
    backend: str = "transformer-bridge"
    revision: str | None = None


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    max_tokens: int
    temperature: float
    max_retries: int
    requests_per_minute: int


@dataclass(frozen=True)
class ProbingConfig:
    probe_layer_frac: float
    n_topics: int
    stories_per_topic: int
    start_token: int
    seed: int


@dataclass(frozen=True)
class SteeringConfig:
    strengths: list[float]


@dataclass(frozen=True)
class NeutralConfig:
    """Neutral corpus + PCA-denoising settings (Phase: valence denoising)."""
    corpus_path: str = "data/stimuli/neutral_corpus.jsonl"
    source_dataset: str = "wikimedia/wikipedia"
    source_config: str = "20231101.en"
    n_texts: int = 1500
    min_words: int = 90
    max_words: int = 200
    variance_threshold: float = 0.50


@dataclass(frozen=True)
class SmokeConfig:
    """Optional small-run settings for architecture compatibility checks."""

    label: str = "smoke"
    n_topics: int = 10
    expected_layers: int = 0
    expected_d_model: int = 0
    min_free_vram_gib: float = 0.0
    max_vram_fraction: float = 0.90


@dataclass(frozen=True)
class NativeHFConfig:
    """Optional production settings for native Hugging Face activation runs."""

    label: str = "native_hf"
    expected_layers: int = 0
    expected_d_model: int = 0
    min_free_vram_gib: float = 0.0
    max_vram_fraction: float = 0.90


@dataclass(frozen=True)
class PathConfig:
    papers: Path
    raw_data: Path
    stimuli: Path
    processed: Path
    results: Path
    logs: Path


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig
    generation: GenerationConfig
    probing: ProbingConfig
    steering: SteeringConfig
    paths: PathConfig
    neutral: NeutralConfig = field(default_factory=NeutralConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    native_hf: NativeHFConfig = field(default_factory=NativeHFConfig)


def load_config(path: str | Path = "config/config.yaml") -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    return ProjectConfig(
        model=ModelConfig(**raw["model"]),
        generation=GenerationConfig(**raw["generation"]),
        probing=ProbingConfig(**raw["probing"]),
        steering=SteeringConfig(**raw["steering"]),
        paths=PathConfig(**{key: Path(value) for key, value in raw["paths"].items()}),
        neutral=NeutralConfig(**(raw.get("neutral") or {})),
        smoke=SmokeConfig(**(raw.get("smoke") or {})),
        native_hf=NativeHFConfig(**(raw.get("native_hf") or {})),
    )


def require_model_name(config: ProjectConfig) -> str:
    if config.model.name == "REPLACE_ME":
        raise ValueError("Set model.name in config/config.yaml before loading a model.")
    return config.model.name
