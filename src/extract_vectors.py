from __future__ import annotations

import argparse

from src.utils.config import load_config
from src.utils.hooks import layer_from_fraction
from src.utils.model_loader import load_hooked_model


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print("[config] model:", config.model.name, flush=True)
    print("[config] probe_layer_frac:", config.probing.probe_layer_frac, flush=True)

    if args.dry_run:
        print("[dry-run] skipping model load", flush=True)
        return

    model = load_hooked_model(config)
    layer_index = layer_from_fraction(model.cfg.n_layers, config.probing.probe_layer_frac)
    print("[model] loaded:", config.model.name, flush=True)
    print("[model] selected layer:", layer_index, flush=True)
    raise NotImplementedError("Vector extraction implementation starts after source-method notes are finalized.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract warmth and competence vectors.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
