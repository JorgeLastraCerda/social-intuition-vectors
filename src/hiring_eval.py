from __future__ import annotations

import argparse

from src.utils.config import load_config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print("[config] raw_data:", config.paths.raw_data, flush=True)
    if args.dry_run:
        print("[dry-run] skipping hiring evaluation", flush=True)
        return
    raise NotImplementedError("Hiring evaluation starts after benchmark data mapping is finalized.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hiring callback behavior.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
