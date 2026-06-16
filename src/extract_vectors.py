from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.utils.config import load_config
from src.utils.hooks import layer_from_fraction, mean_activation_after_token, residual_hook_name
from src.utils.model_loader import load_hooked_model

EXPECTED_CONDITIONS = ("high_warmth", "low_warmth", "high_competence", "low_competence")


def load_stories(stimuli_path: Path) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {c: [] for c in EXPECTED_CONDITIONS}
    with stimuli_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cond = record["condition"]
            if cond not in buckets:
                raise ValueError(f"Unknown condition {cond!r} in stimulus file.")
            buckets[cond].append(record["text"])
    for cond, texts in buckets.items():
        print(f"  [stimuli] {cond}: {len(texts)} stories", flush=True)
    return buckets


def extract_activations(
    model,
    texts: list[str],
    hook_name: str,
    start_token: int,
) -> torch.Tensor:
    vecs = []
    for i, text in enumerate(texts):
        tokens = model.to_tokens(text, prepend_bos=True)
        seq_len = tokens.shape[1]
        if seq_len <= start_token:
            print(
                f"  [warn] story {i}: seq_len={seq_len} <= start_token={start_token},"
                " falling back to full mean-pool",
                flush=True,
            )
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n == hook_name, return_type=None
        )
        acts = cache[hook_name]  # [1, seq, d_model]
        vec = mean_activation_after_token(acts, start_token).squeeze(0)  # [d_model]
        vecs.append(vec.float().cpu())
        if (i + 1) % 10 == 0:
            print(f"  [extract] {i + 1}/{len(texts)} done", flush=True)
    return torch.stack(vecs)  # [N, d_model]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print(f"[config] model: {cfg.model.name}")
    print(f"[config] probe_layer_frac: {cfg.probing.probe_layer_frac}, start_token: {cfg.probing.start_token}")

    stimuli_path = Path(cfg.paths.stimuli) / "concept_stories.jsonl"

    if args.dry_run:
        load_stories(stimuli_path)
        print("[dry-run] stimulus check passed, skipping model load")
        return

    torch.manual_seed(cfg.probing.seed)
    np.random.seed(cfg.probing.seed)

    print(f"Loading model: {cfg.model.name} ...", flush=True)
    model = load_hooked_model(cfg)
    model.eval()

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    layer = layer_from_fraction(n_layers, cfg.probing.probe_layer_frac)
    hook_name = residual_hook_name(layer)
    print(f"[model] n_layers={n_layers}, d_model={d_model}, probe_layer={layer}, hook={hook_name}")

    buckets = load_stories(stimuli_path)

    out_dir = Path(cfg.paths.processed) / "concept_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)

    X: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for cond in EXPECTED_CONDITIONS:
            print(f"\nExtracting: {cond} ({len(buckets[cond])} stories)...", flush=True)
            X[cond] = extract_activations(model, buckets[cond], hook_name, cfg.probing.start_token)
            np.save(out_dir / f"X_{cond}.npy", X[cond].numpy())
            print(f"  saved X_{cond}.npy  shape={X[cond].shape}")

    warmth_vec = X["high_warmth"].mean(dim=0) - X["low_warmth"].mean(dim=0)
    competence_vec = X["high_competence"].mean(dim=0) - X["low_competence"].mean(dim=0)
    np.save(out_dir / "warmth_vec.npy", warmth_vec.numpy())
    np.save(out_dir / "competence_vec.npy", competence_vec.numpy())
    print(f"\nwarmth_vec norm    : {warmth_vec.norm():.4f}")
    print(f"competence_vec norm: {competence_vec.norm():.4f}")

    meta = {
        "model": cfg.model.name,
        "probe_layer": layer,
        "n_layers": n_layers,
        "d_model": d_model,
        "hook": hook_name,
        "start_token": cfg.probing.start_token,
        "seed": cfg.probing.seed,
        "timestamp": int(time.time()),
        "n_per_condition": {c: len(buckets[c]) for c in EXPECTED_CONDITIONS},
        "warmth_vec_norm": round(float(warmth_vec.norm()), 6),
        "competence_vec_norm": round(float(competence_vec.norm()), 6),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[DONE] Outputs in {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract warmth and competence vectors.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
