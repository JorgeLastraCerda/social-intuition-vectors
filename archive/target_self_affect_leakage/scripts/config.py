from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class ExperimentConfig:
    openai_api_key: str
    openai_model: str
    ollama_model: str
    ollama_base_url: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float
    output_dir: Path


def load_config(env_path: Path = Path(".env")) -> ExperimentConfig:
    load_dotenv(env_path)

    return ExperimentConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma4:e4b").strip(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip(),
        temperature=float(os.getenv("EXPERIMENT_TEMPERATURE", "0")),
        max_output_tokens=int(os.getenv("EXPERIMENT_MAX_OUTPUT_TOKENS", "900")),
        timeout_seconds=float(os.getenv("EXPERIMENT_TIMEOUT_SECONDS", "90")),
        output_dir=Path(os.getenv("EXPERIMENT_OUTPUT_DIR", "runs")),
    )


def print_config_summary(config: ExperimentConfig) -> None:
    print("[config] OPENAI_API_KEY set:", bool(config.openai_api_key), flush=True)
    print("[config] OPENAI_MODEL:", config.openai_model, flush=True)
    print("[config] OLLAMA_MODEL:", config.ollama_model, flush=True)
    print("[config] OLLAMA_BASE_URL:", config.ollama_base_url, flush=True)
    print("[config] temperature:", config.temperature, flush=True)
    print("[config] max_output_tokens:", config.max_output_tokens, flush=True)
    print("[config] timeout_seconds:", config.timeout_seconds, flush=True)
    print("[config] output_dir:", config.output_dir, flush=True)
