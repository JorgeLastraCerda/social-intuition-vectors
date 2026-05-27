from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    dtype: str
    device: str


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
    probing: ProbingConfig
    steering: SteeringConfig
    paths: PathConfig


def load_config(path: str | Path = "config/config.yaml") -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    return ProjectConfig(
        model=ModelConfig(**raw["model"]),
        probing=ProbingConfig(**raw["probing"]),
        steering=SteeringConfig(**raw["steering"]),
        paths=PathConfig(**{key: Path(value) for key, value in raw["paths"].items()}),
    )


def require_model_name(config: ProjectConfig) -> str:
    if config.model.name == "REPLACE_ME":
        raise ValueError("Set model.name in config/config.yaml before loading a model.")
    return config.model.name
