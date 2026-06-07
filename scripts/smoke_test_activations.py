"""Smoke test: verify residual-stream activation extraction and steering work end-to-end.

Run:
    python scripts/smoke_test_activations.py
    python scripts/smoke_test_activations.py --device cpu --fallback-cpu-model gpt2

Exit 0 = pipeline is wired correctly and ready for Phase 4.
Exit 1 = something is broken; fix before proceeding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.hooks import (
    add_steering_vector,
    layer_from_fraction,
    mean_activation_after_token,
    residual_hook_name,
)
from src.utils.model_loader import load_hooked_model

WARM_PROMPT = "She listened patiently and offered to help carry the groceries home."
COLD_PROMPT = "He dismissed her question and walked away without answering."


def _encode(model, text: str) -> torch.Tensor:
    return model.to_tokens(text, prepend_bos=True)


def _get_residual(model, tokens: torch.Tensor, hook_name: str) -> torch.Tensor:
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda n: n == hook_name,
        return_type=None,
    )
    return cache[hook_name]


def run(config_path: str, device: str | None, fallback_cpu_model: str | None) -> dict:
    cfg = load_config(config_path)

    effective_device = device or cfg.model.device
    effective_model = cfg.model.name
    effective_dtype = cfg.model.dtype

    if effective_device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available.", end=" ")
        if fallback_cpu_model:
            print(f"Falling back to '{fallback_cpu_model}' on CPU.")
            effective_model = fallback_cpu_model
            effective_dtype = "float32"
            effective_device = "cpu"
        else:
            print("Pass --fallback-cpu-model to run on CPU instead.")
            sys.exit(1)

    # Rebuild config so all downstream helpers see the resolved device/model.
    cfg = cfg.__class__(
        model=cfg.model.__class__(name=effective_model, dtype=effective_dtype, device=effective_device),
        probing=cfg.probing,
        steering=cfg.steering,
        paths=cfg.paths,
    )

    torch.manual_seed(cfg.probing.seed)

    print(f"Loading model: {cfg.model.name} on {cfg.model.device}")
    model = load_hooked_model(cfg)
    model.eval()

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    layer = layer_from_fraction(n_layers, cfg.probing.probe_layer_frac)
    hook_name = residual_hook_name(layer)

    print(f"  n_layers={n_layers}, d_model={d_model}, probe_layer={layer}, hook={hook_name}")

    # --- activation extraction ---
    with torch.no_grad():
        warm_tokens = _encode(model, WARM_PROMPT)
        cold_tokens = _encode(model, COLD_PROMPT)

        warm_acts = _get_residual(model, warm_tokens, hook_name)
        cold_acts = _get_residual(model, cold_tokens, hook_name)

    warm_vec = mean_activation_after_token(warm_acts, cfg.probing.start_token).squeeze(0)
    cold_vec = mean_activation_after_token(cold_acts, cfg.probing.start_token).squeeze(0)

    diff = warm_vec - cold_vec
    diff_norm = diff.norm().item()
    cosine = torch.nn.functional.cosine_similarity(warm_vec, cold_vec, dim=0).item()

    print(f"  warm vec norm : {warm_vec.norm().item():.4f}")
    print(f"  cold vec norm : {cold_vec.norm().item():.4f}")
    print(f"  diff norm     : {diff_norm:.4f}")
    print(f"  cosine(w, c)  : {cosine:.6f}  (should be < 1.0)")

    assert diff_norm > 0, "Difference vector is zero — model returned identical activations for both prompts."
    assert cosine < 1.0 - 1e-6, "Cosine similarity is 1.0 — prompts produced identical activations."

    # --- steering test ---
    torch.manual_seed(cfg.probing.seed)
    rand_vec = torch.randn(d_model, device=warm_vec.device, dtype=warm_vec.dtype)
    rand_vec = rand_vec / rand_vec.norm()
    alpha = 0.5

    with torch.no_grad():
        baseline_logits = model(cold_tokens)[0, -1]

    def steer_hook(resid: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        return add_steering_vector(resid, rand_vec, alpha)

    with torch.no_grad():
        steered_logits = model.run_with_hooks(
            cold_tokens,
            fwd_hooks=[(hook_name, steer_hook)],
        )[0, -1]

    max_logit_delta = (steered_logits - baseline_logits).abs().max().item()
    print(f"  max logit delta after steering (α={alpha}): {max_logit_delta:.6f}")

    assert max_logit_delta > 1e-4, "Steering hook had no effect on logits."

    result = {
        "model": cfg.model.name,
        "probe_layer": layer,
        "n_layers": n_layers,
        "d_model": d_model,
        "hook": hook_name,
        "warm_vec_norm": round(warm_vec.norm().item(), 6),
        "cold_vec_norm": round(cold_vec.norm().item(), 6),
        "diff_norm": round(diff_norm, 6),
        "cosine_warm_cold": round(cosine, 6),
        "steering_alpha": alpha,
        "max_logit_delta": round(max_logit_delta, 6),
        "status": "PASS",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test residual-stream extraction and steering.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument(
        "--fallback-cpu-model",
        default=None,
        metavar="MODEL",
        help="If CUDA is unavailable, load this small model on CPU instead (e.g. gpt2).",
    )
    args = parser.parse_args()

    try:
        result = run(args.config, args.device, args.fallback_cpu_model)
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"smoke_test_{int(time.time())}.json"
    log_path.write_text(json.dumps(result, indent=2))

    print(f"\n[PASS] All checks passed. Log: {log_path}")


if __name__ == "__main__":
    main()
