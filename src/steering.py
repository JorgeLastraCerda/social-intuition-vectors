from __future__ import annotations

import argparse

from src.utils.config import load_config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print("[config] steering strengths:", config.steering.strengths, flush=True)
    if args.dry_run:
        print("[dry-run] skipping steering", flush=True)
        return
    raise NotImplementedError("Steering starts after vectors and callback scoring are implemented.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal steering experiments.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
