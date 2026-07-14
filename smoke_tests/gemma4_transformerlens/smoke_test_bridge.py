"""SCCKN smoke gate for Gemma 4 TransformerLens Bridge runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.gemma_scope_causality import make_steering_hook, yes_no_margin
from src.hiring_steering import hiring_prompt
from src.utils.config import load_config
from src.utils.hooks import residual_hook_name
from src.utils.model_loader import load_hooked_model, model_runtime_metadata
from src.utils.prompting import decision_token_ids, encode_decision_prompt, encode_passage


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = replace(cfg, model=replace(cfg.model, name=args.model))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = load_hooked_model(cfg)
    model.eval()
    if model.cfg.n_layers != args.expected_layers:
        raise AssertionError((model.cfg.n_layers, args.expected_layers))
    if model.cfg.d_model != args.expected_d_model:
        raise AssertionError((model.cfg.d_model, args.expected_d_model))

    layer = round((model.cfg.n_layers - 1) * cfg.probing.probe_layer_frac)
    hook_name = residual_hook_name(layer)
    passage_tokens = encode_passage(model, "A person carefully completed a routine task.")
    with torch.no_grad():
        bridge_logits, cache = model.run_with_cache(
            passage_tokens,
            names_filter=hook_name,
        )
        hf_logits = model.original_model(input_ids=passage_tokens).logits
    activations = cache[hook_name]
    if activations.shape != (
        1,
        passage_tokens.shape[1],
        args.expected_d_model,
    ):
        raise AssertionError(f"Unexpected activation shape: {activations.shape}")
    if not torch.isfinite(activations).all() or not torch.isfinite(bridge_logits).all():
        raise FloatingPointError("Non-finite activation or logit in smoke test.")
    max_logit_diff = float((bridge_logits.float() - hf_logits.float()).abs().max())
    if max_logit_diff > args.max_logit_diff:
        raise AssertionError(f"Bridge/HF max logit difference {max_logit_diff:.6g}")

    prompt = hiring_prompt("Jordan Lee")
    rendered, _ = encode_decision_prompt(model, prompt, "native-chat")
    yes_id, no_id = decision_token_ids(model, rendered, "native-chat")
    baseline = yes_no_margin(
        model, prompt, hook_name, prompt_format="native-chat"
    )
    direction = np.ones(args.expected_d_model, dtype=np.float32)
    steered = yes_no_margin(
        model,
        prompt,
        hook_name,
        make_steering_hook(direction, 0.05),
        prompt_format="native-chat",
    )
    if not np.isfinite([baseline, steered]).all() or baseline == steered:
        raise AssertionError("Steering smoke check did not produce a finite margin change.")

    peak_gib = (
        torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    )
    result = {
        "model": args.model,
        "seed": args.seed,
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "probe_layer": layer,
        "hook": hook_name,
        "activation_shape": list(activations.shape),
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "bridge_hf_max_logit_diff": max_logit_diff,
        "baseline_margin": baseline,
        "steered_margin": steered,
        "peak_allocated_vram_gib": peak_gib,
        "prompt_format": "native-chat",
        "runtime": model_runtime_metadata(model),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-layers", required=True, type=int)
    parser.add_argument("--expected-d-model", required=True, type=int)
    parser.add_argument("--seed", default=20260527, type=int)
    parser.add_argument("--max-logit-diff", default=0.02, type=float)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
