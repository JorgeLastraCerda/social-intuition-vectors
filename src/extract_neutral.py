from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.utils.config import load_config
from src.utils.hooks import layer_from_fraction, residual_hook_name
from src.utils.model_loader import load_hooked_model, model_runtime_metadata
from src.extract_vectors import extract_activations  # reuse identical extraction loop


def load_neutral(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    print(f"  [neutral] {len(texts)} texts loaded from {path}")
    return texts


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.model is not None:
        cfg = replace(cfg, model=replace(cfg.model, name=args.model))
    corpus = Path(cfg.neutral.corpus_path)

    if args.dry_run:
        load_neutral(corpus)
        print("[dry-run] neutral corpus check passed, skipping model load")
        return

    torch.manual_seed(cfg.probing.seed)
    np.random.seed(cfg.probing.seed)

    print(f"Loading model: {cfg.model.name} ...", flush=True)
    model = load_hooked_model(cfg)
    model.eval()

    layer = layer_from_fraction(model.cfg.n_layers, cfg.probing.probe_layer_frac)
    hook = residual_hook_name(layer)
    print(f"[model] n_layers={model.cfg.n_layers} d_model={model.cfg.d_model} "
          f"probe_layer={layer} hook={hook} start_token={cfg.probing.start_token}")

    texts = load_neutral(corpus)
    out_dir = Path(cfg.paths.processed) / args.vectors_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        X = extract_activations(model, texts, hook, cfg.probing.start_token)
    np.save(out_dir / "X_neutral.npy", X.numpy())

    meta = {
        "n_neutral": len(texts),
        "probe_layer": layer,
        "d_model": int(model.cfg.d_model),
        "start_token": cfg.probing.start_token,
        "corpus": str(corpus),
        "seed": cfg.probing.seed,
        "timestamp": int(time.time()),
        "model": cfg.model.name,
        "hook": hook,
        "input_format": "raw-passage",
        "runtime": model_runtime_metadata(model),
    }
    (out_dir / "neutral_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[DONE] saved X_neutral.npy shape={tuple(X.shape)} -> {out_dir}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract residual-stream activations for the neutral corpus.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=None, help="Override cfg.model.name.")
    ap.add_argument(
        "--vectors-subdir",
        default="concept_vectors",
        help="Model-specific output directory under cfg.paths.processed.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    main()
