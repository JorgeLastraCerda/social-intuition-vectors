"""Smoke test — Qwen2.5-1.5B-Instruct + TransformerLens.

Validates that warmth is linearly decodable from the residual stream of
Qwen2.5-1.5B-Instruct (the original pilot model).  Identical pipeline to the
baseline in scripts/smoke_test_probe.py but:
  - Sentences imported from smoke_tests/stimuli.py (shared with all three tests).
  - Adds an equal-magnitude random-vector steering control so warmth-direction
    steering can be compared fairly (fixes the misleading "18x" claim in README).
  - Results written to smoke_tests/qwen_transformerlens/results/.

Run from repo root:
    python smoke_tests/qwen_transformerlens/smoke_test_probe.py
    python smoke_tests/qwen_transformerlens/smoke_test_probe.py --model google/gemma-3-4b-it

Exit 0 = PASS.  Exit 1 = infrastructure or probe failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# -- path setup --------------------------------------------------------------
HERE  = Path(__file__).resolve().parent          # smoke_tests/qwen_transformerlens/
SMOKE = HERE.parent                              # smoke_tests/
ROOT  = SMOKE.parent                             # repo root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SMOKE))

from stimuli import COLD_SENTENCES, WARM_SENTENCES  # noqa: E402 (after sys.path)

from src.utils.config import load_config
from src.utils.hooks import (
    add_steering_vector,
    layer_from_fraction,
    mean_activation_after_token,
    residual_hook_name,
)
from src.utils.model_loader import load_hooked_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode(model, text: str) -> torch.Tensor:
    return model.to_tokens(text, prepend_bos=True)


def _get_residual(model, tokens: torch.Tensor, hook_name: str) -> torch.Tensor:
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: n == hook_name, return_type=None
    )
    return cache[hook_name]


def _extract_all(model, sentences: list[str], hook_name: str,
                 start_token: int) -> torch.Tensor:
    """Return [N, d_model] float32 activation matrix for a sentence list."""
    vecs = []
    for sent in sentences:
        tokens = _encode(model, sent)
        acts = _get_residual(model, tokens, hook_name)      # [1, seq, d_model]
        vec = mean_activation_after_token(acts, start_token).squeeze(0)  # [d_model]
        vecs.append(vec.float().cpu())
    return torch.stack(vecs)                                 # [N, d_model]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str, model_override: str | None, start_token: int,
        seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cfg = load_config(config_path)
    if model_override:
        cfg = cfg.__class__(
            model=cfg.model.__class__(
                name=model_override,
                dtype=cfg.model.dtype,
                device=cfg.model.device,
            ),
            generation=cfg.generation,
            probing=cfg.probing,
            steering=cfg.steering,
            paths=cfg.paths,
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Loading model: {cfg.model.name} on {cfg.model.device} ({cfg.model.dtype})")
    model = load_hooked_model(cfg)
    model.eval()

    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model
    layer    = layer_from_fraction(n_layers, cfg.probing.probe_layer_frac)
    hook_name = residual_hook_name(layer)
    print(f"  n_layers={n_layers}, d_model={d_model}, "
          f"probe_layer={layer}, hook={hook_name}, start_token={start_token}")

    # --- extract activations ------------------------------------------------
    print("Extracting warm activations (50 sentences)...")
    with torch.no_grad():
        X_warm = _extract_all(model, WARM_SENTENCES, hook_name, start_token)
        print("Extracting cold activations (50 sentences)...")
        X_cold = _extract_all(model, COLD_SENTENCES, hook_name, start_token)

    # --- mean direction -----------------------------------------------------
    mean_warm = X_warm.mean(dim=0)
    mean_cold = X_cold.mean(dim=0)
    diff      = mean_warm - mean_cold
    diff_norm = diff.norm().item()
    cosine    = torch.nn.functional.cosine_similarity(
        mean_warm.unsqueeze(0), mean_cold.unsqueeze(0)
    ).item()
    unit_dir  = diff / (diff.norm() + 1e-12)

    print(f"  diff norm   : {diff_norm:.4f}")
    print(f"  cosine(W,C) : {cosine:.6f}")
    assert diff_norm > 0, "Mean difference vector is zero."
    assert cosine < 1.0 - 1e-6, "Mean vectors are identical."

    # --- projection separation & Cohen's d ---------------------------------
    proj_warm = (X_warm @ unit_dir).numpy()
    proj_cold = (X_cold @ unit_dir).numpy()
    pw_mean, pw_std = float(proj_warm.mean()), float(proj_warm.std())
    pc_mean, pc_std = float(proj_cold.mean()), float(proj_cold.std())
    pooled_std = float(np.sqrt((proj_warm.var() + proj_cold.var()) / 2.0) + 1e-12)
    cohens_d   = (pw_mean - pc_mean) / pooled_std
    print(f"  proj warm  : {pw_mean:.4f} +/- {pw_std:.4f}")
    print(f"  proj cold  : {pc_mean:.4f} +/- {pc_std:.4f}")
    print(f"  Cohen's d  : {cohens_d:.4f}")
    assert pw_mean > pc_mean, "Warm projection <= cold — direction inverted or absent."

    # --- 5-fold CV probe ----------------------------------------------------
    X_np = torch.cat([X_warm, X_cold], dim=0).numpy()
    y_np = np.array([1] * 50 + [0] * 50)
    lr   = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores   = cross_val_score(lr, X_np, y_np, cv=cv, scoring="accuracy")
    cv_mean  = float(scores.mean())
    cv_std   = float(scores.std())
    print(f"  5-fold CV  : {cv_mean:.4f} +/- {cv_std:.4f}  "
          f"(folds: {[round(s, 3) for s in scores.tolist()]})")
    assert cv_mean > 0.8, f"Probe accuracy {cv_mean:.3f} <= 0.80."

    # --- steering: warmth direction vs equal-magnitude random ---------------
    # Both vectors are scaled to the same L2 magnitude (alpha) so the
    # comparison is apples-to-apples (fixes the misleading "18x" README claim).
    steer_sentence = COLD_SENTENCES[-1]
    with torch.no_grad():
        steer_tokens    = _encode(model, steer_sentence)
        baseline_logits = model(steer_tokens)[0, -1]

    mean_resid_norm = float(torch.cat([X_warm, X_cold], dim=0).norm(dim=1).mean())
    alpha = 0.5 * mean_resid_norm          # strength relative to residual norm

    # warmth direction (already unit-norm; scale to alpha magnitude)
    steer_vec = unit_dir.to(device=baseline_logits.device, dtype=baseline_logits.dtype)

    # equal-magnitude random direction
    torch.manual_seed(seed)
    rand_unit = torch.randn(d_model, device=steer_vec.device, dtype=steer_vec.dtype)
    rand_unit = rand_unit / (rand_unit.norm() + 1e-12)

    def make_hook(vec):
        def hook_fn(resid: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
            return add_steering_vector(resid, vec, alpha)
        return hook_fn

    with torch.no_grad():
        steered_warmth = model.run_with_hooks(
            steer_tokens, fwd_hooks=[(hook_name, make_hook(steer_vec))]
        )[0, -1]
        steered_random = model.run_with_hooks(
            steer_tokens, fwd_hooks=[(hook_name, make_hook(rand_unit))]
        )[0, -1]

    delta_warmth = (steered_warmth - baseline_logits).abs().max().item()
    delta_random = (steered_random - baseline_logits).abs().max().item()
    ratio        = delta_warmth / (delta_random + 1e-12)
    print(f"  steering alpha          : {alpha:.4f}  (0.5 × mean resid norm {mean_resid_norm:.4f})")
    print(f"  max logit delta warmth  : {delta_warmth:.6f}")
    print(f"  max logit delta random  : {delta_random:.6f}  (equal magnitude)")
    print(f"  warmth/random ratio     : {ratio:.2f}×")
    assert delta_warmth > 1e-4, "Warmth steering hook had no effect."

    result = {
        "test":                "qwen_transformerlens",
        "model":               cfg.model.name,
        "probe_layer":         layer,
        "n_layers":            n_layers,
        "d_model":             d_model,
        "hook":                hook_name,
        "start_token":         start_token,
        "seed":                seed,
        "n_warm":              50,
        "n_cold":              50,
        "diff_norm":           round(diff_norm, 6),
        "cosine_mean_warm_cold": round(cosine, 6),
        "proj_warm_mean":      round(pw_mean, 6),
        "proj_warm_std":       round(pw_std, 6),
        "proj_cold_mean":      round(pc_mean, 6),
        "proj_cold_std":       round(pc_std, 6),
        "cohens_d":            round(cohens_d, 6),
        "probe_cv_mean":       round(cv_mean, 6),
        "probe_cv_std":        round(cv_std, 6),
        "probe_cv_folds":      [round(s, 6) for s in scores.tolist()],
        "mean_resid_norm":     round(mean_resid_norm, 6),
        "steering_alpha":      round(alpha, 6),
        "max_logit_delta_warmth": round(delta_warmth, 6),
        "max_logit_delta_random": round(delta_random, 6),
        "warmth_random_ratio": round(ratio, 4),
        "status":              "PASS",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test — Qwen + TransformerLens, 50+50 probe."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model",  default=None,
                        help="Override model name (e.g. google/gemma-3-4b-it).")
    parser.add_argument("--start-token", type=int, default=1,
                        help="Token index to start mean-pooling from (default 1 = skip BOS).")
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    try:
        result = run(args.config, args.model, args.start_token, args.seed)
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"smoke_probe_{int(time.time())}.json"
    log_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[PASS] All checks passed. Log: {log_path}")


if __name__ == "__main__":
    main()
