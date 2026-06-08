"""Smoke test — Gemma 4 + nnsight.

Same probe pipeline as the TransformerLens tests but uses nnsight (v0.6+) to
hook into the original HuggingFace model weights.  nnsight wraps any HF model
with a context-manager trace API; it does not reimplementthe model, so weights
are bit-for-bit identical to the HF checkpoint.

Target model: google/gemma-4-e4b-it (or larger on SCCKN).
nnsight accesses the residual stream via:
    model.model.layers[N].output[0]   (hidden states after block N)

NOTE on Gemma 4 PLE (Per-Layer Embeddings):
    Gemma 4 adds position-dependent learned offsets to each layer's input,
    meaning the residual stream carries PLE information beyond the usual
    attention + MLP updates.  This does not prevent extraction or steering but
    means the "clean superposition" assumption is slightly weaker than for
    Gemma 3.  Treat this test as exploratory.

Run from repo root:
    python smoke_tests/gemma4_nnsight/smoke_test_probe.py \
        --model google/gemma-4-e4b-it

Exit 0 = PASS.  Exit 1 = failure.
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
HERE  = Path(__file__).resolve().parent
SMOKE = HERE.parent
ROOT  = SMOKE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SMOKE))

from stimuli import COLD_SENTENCES, WARM_SENTENCES  # noqa: E402

from src.utils.config import load_config


# ---------------------------------------------------------------------------
# nnsight extraction helpers
# ---------------------------------------------------------------------------

def _layer_from_fraction(n_layers: int, fraction: float) -> int:
    return min(n_layers - 1, max(0, round((n_layers - 1) * fraction)))


def _extract_all_nnsight(model, tokenizer, sentences: list[str],
                          layer_idx: int, device: str) -> torch.Tensor:
    """Return [N, d_model] float32 activation matrix using nnsight traces."""
    vecs = []
    for sent in sentences:
        ids = tokenizer(sent, return_tensors="pt").input_ids.to(device)
        with model.trace(ids):
            # hidden states after transformer block layer_idx, shape [1, seq, d_model]
            hidden = model.model.layers[layer_idx].output[0].save()
        acts = hidden.value.float().cpu()         # [1, seq, d_model]
        # mean-pool from token 1 onward (skip BOS)
        vec  = acts[0, 1:, :].mean(dim=0)         # [d_model]
        vecs.append(vec)
    return torch.stack(vecs)                       # [N, d_model]


def _logits_nnsight(model, tokenizer, sentence: str,
                    layer_idx: int, device: str,
                    steer_vec: torch.Tensor | None = None,
                    alpha: float = 0.0) -> torch.Tensor:
    """Return last-token logits, optionally steering at layer_idx."""
    ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
    if steer_vec is None:
        with model.trace(ids):
            logits = model.lm_head.output.save()
    else:
        sv = steer_vec.to(device=device, dtype=torch.bfloat16)
        with model.trace(ids):
            h = model.model.layers[layer_idx].output[0]
            model.model.layers[layer_idx].output[0][:] = h + alpha * sv
            logits = model.lm_head.output.save()
    return logits.value[0, -1].float().cpu()       # [vocab]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str, model_name: str, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    try:
        from nnsight import LanguageModel, VisionLanguageModel
    except ImportError:
        raise SystemExit(
            "nnsight is not installed in this environment.\n"
            "Install with: pip install nnsight"
        )

    cfg    = load_config(config_path)
    device = cfg.model.device
    dtype  = torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float32

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Gemma 4 is registered as a multimodal (image+text) model in HF; nnsight
    # requires VisionLanguageModel for such checkpoints, not LanguageModel.
    VISION_MODELS = ("gemma-4",)
    use_vlm = any(tag in model_name.lower() for tag in VISION_MODELS)
    ModelClass = VisionLanguageModel if use_vlm else LanguageModel

    print(f"Loading model: {model_name} on {device} ({dtype}) [{'VisionLanguageModel' if use_vlm else 'LanguageModel'}]")
    nns_model = ModelClass(
        model_name,
        device_map=device,
        dtype=dtype,
    )
    tokenizer = nns_model.tokenizer

    # Infer n_layers and d_model from the HF config
    hf_cfg  = nns_model.config
    n_layers = hf_cfg.num_hidden_layers
    d_model  = hf_cfg.hidden_size
    layer    = _layer_from_fraction(n_layers, cfg.probing.probe_layer_frac)
    print(f"  n_layers={n_layers}, d_model={d_model}, probe_layer={layer}")

    # --- extract activations -----------------------------------------------
    print("Extracting warm activations (50 sentences)...")
    X_warm = _extract_all_nnsight(nns_model, tokenizer, WARM_SENTENCES,
                                   layer, device)
    print("Extracting cold activations (50 sentences)...")
    X_cold = _extract_all_nnsight(nns_model, tokenizer, COLD_SENTENCES,
                                   layer, device)

    # --- mean direction ----------------------------------------------------
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
    assert diff_norm > 0
    assert cosine < 1.0 - 1e-6

    # --- projection & Cohen's d -------------------------------------------
    proj_warm  = (X_warm @ unit_dir).numpy()
    proj_cold  = (X_cold @ unit_dir).numpy()
    pw_mean, pw_std = float(proj_warm.mean()), float(proj_warm.std())
    pc_mean, pc_std = float(proj_cold.mean()), float(proj_cold.std())
    pooled_std = float(np.sqrt((proj_warm.var() + proj_cold.var()) / 2.0) + 1e-12)
    cohens_d   = (pw_mean - pc_mean) / pooled_std
    print(f"  proj warm  : {pw_mean:.4f} +/- {pw_std:.4f}")
    print(f"  proj cold  : {pc_mean:.4f} +/- {pc_std:.4f}")
    print(f"  Cohen's d  : {cohens_d:.4f}")
    assert pw_mean > pc_mean

    # --- 5-fold CV ---------------------------------------------------------
    X_np = torch.cat([X_warm, X_cold], dim=0).numpy()
    y_np = np.array([1] * 50 + [0] * 50)
    lr   = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores  = cross_val_score(lr, X_np, y_np, cv=cv, scoring="accuracy")
    cv_mean = float(scores.mean())
    cv_std  = float(scores.std())
    print(f"  5-fold CV  : {cv_mean:.4f} +/- {cv_std:.4f}  "
          f"(folds: {[round(s, 3) for s in scores.tolist()]})")
    assert cv_mean > 0.8, f"Probe accuracy {cv_mean:.3f} <= 0.80."

    # --- steering: warmth vs equal-magnitude random -------------------------
    steer_sentence  = COLD_SENTENCES[-1]
    mean_resid_norm = float(X_np.reshape(100, d_model).__class__(X_np).mean())  # fallback
    mean_resid_norm = float(
        torch.from_numpy(X_np).norm(dim=1).mean().item()
    )
    alpha = 0.5 * mean_resid_norm

    baseline_logits = _logits_nnsight(nns_model, tokenizer, steer_sentence,
                                       layer, device)
    steered_warmth  = _logits_nnsight(nns_model, tokenizer, steer_sentence,
                                       layer, device, steer_vec=unit_dir, alpha=alpha)

    torch.manual_seed(seed)
    rand_unit = torch.randn(d_model)
    rand_unit = rand_unit / (rand_unit.norm() + 1e-12)
    steered_random  = _logits_nnsight(nns_model, tokenizer, steer_sentence,
                                       layer, device, steer_vec=rand_unit, alpha=alpha)

    delta_warmth = (steered_warmth - baseline_logits).abs().max().item()
    delta_random = (steered_random - baseline_logits).abs().max().item()
    ratio        = delta_warmth / (delta_random + 1e-12)
    print(f"  steering alpha : {alpha:.4f}")
    print(f"  delta warmth   : {delta_warmth:.6f}")
    print(f"  delta random   : {delta_random:.6f}  (equal magnitude)")
    print(f"  ratio          : {ratio:.2f}×")
    assert delta_warmth > 1e-4

    result = {
        "test":                   "gemma4_nnsight",
        "model":                  model_name,
        "backend":                "nnsight",
        "probe_layer":            layer,
        "n_layers":               n_layers,
        "d_model":                d_model,
        "start_token":            1,
        "seed":                   seed,
        "n_warm":                 50,
        "n_cold":                 50,
        "diff_norm":              round(diff_norm, 6),
        "cosine_mean_warm_cold":  round(cosine, 6),
        "proj_warm_mean":         round(pw_mean, 6),
        "proj_warm_std":          round(pw_std, 6),
        "proj_cold_mean":         round(pc_mean, 6),
        "proj_cold_std":          round(pc_std, 6),
        "cohens_d":               round(cohens_d, 6),
        "probe_cv_mean":          round(cv_mean, 6),
        "probe_cv_std":           round(cv_std, 6),
        "probe_cv_folds":         [round(s, 6) for s in scores.tolist()],
        "mean_resid_norm":        round(mean_resid_norm, 6),
        "steering_alpha":         round(alpha, 6),
        "max_logit_delta_warmth": round(delta_warmth, 6),
        "max_logit_delta_random": round(delta_random, 6),
        "warmth_random_ratio":    round(ratio, 4),
        "gemma4_ple_note":        (
            "Gemma 4 uses Per-Layer Embeddings (PLE); residual-stream "
            "superposition assumption is slightly weaker. Treat as exploratory."
        ),
        "status": "PASS",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test — Gemma 4 + nnsight, 50+50 probe."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", required=True,
                        help="Gemma 4 model name (e.g. google/gemma-4-e4b-it).")
    parser.add_argument("--seed", type=int, default=20260527)
    args = parser.parse_args()

    try:
        result = run(args.config, args.model, args.seed)
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    out_dir  = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"smoke_probe_{int(time.time())}.json"
    log_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[PASS] All checks passed. Log: {log_path}")


if __name__ == "__main__":
    main()
