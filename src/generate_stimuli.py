from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.utils.config import load_config


CONDITIONS = ["high_warmth", "low_warmth", "high_competence", "low_competence"]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    random.seed(config.probing.seed)
    config.paths.stimuli.mkdir(parents=True, exist_ok=True)

    concept_path = config.paths.stimuli / "concept_stories.jsonl"
    hiring_path = config.paths.stimuli / "hiring_prompts.jsonl"

    if args.dry_run:
        print("[dry-run] would write:", concept_path)
        print("[dry-run] would write:", hiring_path)
        return

    write_placeholder_concept_stories(concept_path)
    write_placeholder_hiring_prompts(hiring_path)
    print("[done] wrote:", concept_path)
    print("[done] wrote:", hiring_path)


def write_placeholder_concept_stories(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for condition in CONDITIONS:
            row = {
                "id": f"placeholder_{condition}_001",
                "condition": condition,
                "topic": "placeholder",
                "text": f"Placeholder story for {condition}. Replace after generator design is finalized.",
            }
            handle.write(json.dumps(row) + "\n")


def write_placeholder_hiring_prompts(path: Path) -> None:
    row = {
        "id": "placeholder_signal_001",
        "social_signal": "PLACEHOLDER",
        "prompt": "You are screening applicants. Based on this resume, should this candidate receive a callback? Answer Y or N.",
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate warmth/competence stimuli.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
