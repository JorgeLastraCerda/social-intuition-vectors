from __future__ import annotations

import argparse

from src.utils.config import load_config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print("[config] processed:", config.paths.processed, flush=True)
    if args.dry_run:
        print("[dry-run] skipping validation", flush=True)
        return
    raise NotImplementedError("Probe validation starts after vectors and benchmark column mapping exist.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate warmth and competence probes.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
