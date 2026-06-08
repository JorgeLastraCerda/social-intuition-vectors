"""GemmaScope 2 SAE decomposition — warmth direction vs. positive tone.

This script addresses the primary scientific risk identified in the smoke test
audit: the 50 warm sentences are also more positive in tone than the 50 cold
sentences, so the "warmth vector" might actually encode general positive
sentiment rather than warmth specifically.

GemmaScope 2 (Dec 2025) provides pretrained Sparse Autoencoders (SAEs) for
every layer of every Gemma 3 variant.  An SAE maps a residual-stream activation
to a small set of interpretable features.  Features labelled "warmth",
"friendliness", "care" etc. are genuinely warmth-specific; features labelled
"positivity", "joy", "good" are valence/tone.

Method:
  1. Load the warmth vector saved by smoke_test_probe.py.
  2. Encode it through the GemmaScope 2 SAE for the same layer.
  3. Report the top-activating features (by magnitude in the SAE encoding).
  4. Do the same for the per-sentence mean activations to get a feature-level
     warm vs. cold discrimination score.
  5. Output: JSON report + list of top features for human inspection.

Run after smoke_test_probe.py has saved results/:
    python smoke_tests/gemma3_transformerlens/sae_decompose.py \
        --model google/gemma-3-4b-it \
        --sae-release gemma-scope-2-4b-it-res \
        --sae-id layer_17_width_16k_l0_medium

SAE release names for common Gemma 3 sizes (residual stream, IT, 16k width):
    1B  -> gemma-scope-2-1b-it-res   sae_id prefix: layer_N_width_16k_l0_medium
    4B  -> gemma-scope-2-4b-it-res   sae_id prefix: layer_N_width_16k_l0_medium
    12B -> gemma-scope-2-12b-it-res  sae_id prefix: layer_N_width_16k_l0_medium
    27B -> gemma-scope-2-27b-it-res  sae_id prefix: layer_N_width_16k_l0_medium

Full catalogue: https://huggingface.co/google/gemma-scope-2

Exit 0 = decomposition ran.  Exit 1 = failure.
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
from src.utils.hooks import layer_from_fraction, mean_activation_after_token, residual_hook_name
from src.utils.model_loader import load_hooked_model


# ---------------------------------------------------------------------------
# SAE decomposition helpers
# ---------------------------------------------------------------------------

def load_sae(release: str, sae_id: str, device: str):
    """Load a GemmaScope 2 SAE via sae-lens."""
    try:
        from sae_lens import SAE
    except ImportError:
        raise SystemExit(
            "sae-lens is not installed in this environment.\n"
            "Install with: pip install sae-lens"
        )
    sae, cfg_dict, _sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()
    return sae, cfg_dict


def top_features(feature_acts: np.ndarray, k: int = 20) -> list[dict]:
    """Return the top-k features by absolute activation magnitude."""
    indices = np.argsort(np.abs(feature_acts))[::-1][:k]
    return [
        {"feature_idx": int(i), "activation": float(feature_acts[i])}
        for i in indices
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str, model_name: str, sae_release: str, sae_id: str,
        n_top_features: int, seed: int, results_dir: Path | None = None) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cfg = load_config(config_path)
    cfg = cfg.__class__(
        model=cfg.model.__class__(
            name=model_name,
            dtype=cfg.model.dtype,
            device=cfg.model.device,
        ),
        generation=cfg.generation,
        probing=cfg.probing,
        steering=cfg.steering,
        paths=cfg.paths,
    )

    device = cfg.model.device

    # -- load warmth vector saved by smoke_test_probe.py -------------------
    if results_dir is None:
        results_dir = HERE / "results"
    warmth_vec_path = results_dir / "warmth_vector.npy"
    if not warmth_vec_path.exists():
        raise FileNotFoundError(
            f"warmth_vector.npy not found at {warmth_vec_path}. "
            "Run smoke_test_probe.py first."
        )
    warmth_vec = torch.from_numpy(np.load(warmth_vec_path)).float().to(device)
    print(f"Loaded warmth vector from {warmth_vec_path}  shape={warmth_vec.shape}")

    # -- load model to get layer index + per-sentence activations ----------
    # (Only needed if X_warm/X_cold not already saved.)
    X_warm_path = results_dir / "X_warm.npy"
    X_cold_path = results_dir / "X_cold.npy"

    if X_warm_path.exists() and X_cold_path.exists():
        X_warm = torch.from_numpy(np.load(X_warm_path)).float()
        X_cold = torch.from_numpy(np.load(X_cold_path)).float()
        print("Loaded saved activations (skipping model load).")
        n_layers = None  # not needed below
    else:
        print(f"Loading model: {cfg.model.name} (needed to re-extract activations)")
        model = load_hooked_model(cfg)
        model.eval()
        n_layers  = model.cfg.n_layers
        layer     = layer_from_fraction(n_layers, cfg.probing.probe_layer_frac)
        hook_name = residual_hook_name(layer)
        vecs = []
        with torch.no_grad():
            for sent in WARM_SENTENCES + COLD_SENTENCES:
                toks = model.to_tokens(sent, prepend_bos=True)
                _, cache = model.run_with_cache(
                    toks, names_filter=lambda n: n == hook_name, return_type=None
                )
                acts = cache[hook_name]
                vec  = mean_activation_after_token(acts, 1).squeeze(0).float().cpu()
                vecs.append(vec)
        X_all  = torch.stack(vecs)
        X_warm = X_all[:50]
        X_cold = X_all[50:]

    # -- load SAE ----------------------------------------------------------
    print(f"Loading SAE: {sae_release} / {sae_id}")
    sae, sae_cfg = load_sae(sae_release, sae_id, device)
    n_features = sae.cfg.d_sae
    print(f"  SAE width: {n_features} features")

    # -- encode warmth vector through SAE ----------------------------------
    with torch.no_grad():
        warmth_sae = sae.encode(warmth_vec.unsqueeze(0)).squeeze(0).cpu().numpy()

    top_warmth = top_features(warmth_sae, k=n_top_features)
    print(f"\nTop {n_top_features} SAE features in the warmth DIRECTION vector:")
    for f in top_warmth[:10]:
        print(f"  feature {f['feature_idx']:6d}  activation={f['activation']:+.4f}")

    # -- encode per-sentence activations -----------------------------------
    X_all   = torch.cat([X_warm, X_cold], dim=0).to(device)  # [100, d_model]
    y_np    = np.array([1] * 50 + [0] * 50)

    chunk   = 20  # encode in small batches to avoid OOM
    sae_all = []
    with torch.no_grad():
        for i in range(0, len(X_all), chunk):
            batch = X_all[i:i+chunk]
            sae_all.append(sae.encode(batch).cpu())
    sae_all = torch.cat(sae_all, dim=0).numpy()  # [100, n_features]

    # -- feature-level probe: which SAE features discriminate warm vs cold?
    lr  = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    sae_cv_scores  = cross_val_score(lr, sae_all, y_np, cv=cv, scoring="accuracy")
    sae_cv_mean    = float(sae_cv_scores.mean())
    sae_cv_std     = float(sae_cv_scores.std())
    print(f"\nSAE-feature 5-fold probe CV: {sae_cv_mean:.4f} +/- {sae_cv_std:.4f}")
    print("  (If this matches or exceeds the residual-stream probe, the SAE captures the signal.)")

    # -- mean SAE activation per class (which features activate for warm vs cold?)
    mean_sae_warm = sae_all[:50].mean(axis=0)
    mean_sae_cold = sae_all[50:].mean(axis=0)
    sae_diff      = mean_sae_warm - mean_sae_cold
    top_diff_feats = top_features(sae_diff, k=n_top_features)
    print(f"\nTop {n_top_features} features by warm-minus-cold mean activation:")
    print("  (human inspection: are these warmth/care features or positivity/valence features?)")
    for f in top_diff_feats[:10]:
        direction = "WARM+" if f["activation"] > 0 else "COLD+"
        print(f"  feature {f['feature_idx']:6d}  diff={f['activation']:+.4f}  [{direction}]")
    print(f"  -> Full list in results JSON.  Look up on Neuronpedia: "
          f"https://www.neuronpedia.org/")

    result = {
        "test":                "gemma3_sae_decompose",
        "model":               cfg.model.name,
        "sae_release":         sae_release,
        "sae_id":              sae_id,
        "n_features":          n_features,
        "sae_cv_mean":         round(sae_cv_mean, 6),
        "sae_cv_std":          round(sae_cv_std, 6),
        "sae_cv_folds":        [round(s, 6) for s in sae_cv_scores.tolist()],
        "top_warmth_direction_features": top_warmth,
        "top_warm_minus_cold_features":  top_diff_feats,
        "interpretation_note": (
            "Look up feature indices on Neuronpedia "
            "(https://www.neuronpedia.org/) to determine whether top features "
            "represent warmth/care/friendliness (= warmth-specific) or "
            "positivity/joy/good (= valence confound)."
        ),
        "status": "DONE",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GemmaScope 2 SAE decomposition: warmth direction vs. tone."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", required=True,
                        help="Gemma 3 model name matching smoke_test_probe.py run.")
    parser.add_argument("--sae-release", required=True,
                        help="SAELens release string (e.g. gemma-scope-2-4b-it-res).")
    parser.add_argument("--sae-id", required=True,
                        help="SAE ID (e.g. layer_17_width_16k_l0_medium).")
    parser.add_argument("--n-top-features", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory where smoke_test_probe.py saved warmth_vector.npy / X_warm.npy / X_cold.npy, "
             "and where this script writes its JSON log (default: smoke_tests/gemma3_transformerlens/results). "
             "Must match the --out-dir used in the preceding probe run.",
    )
    args = parser.parse_args()

    results_dir = Path(args.out_dir) if args.out_dir else HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run(
            args.config, args.model, args.sae_release, args.sae_id,
            args.n_top_features, args.seed, results_dir=results_dir,
        )
    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    log_path = results_dir / f"sae_decompose_{int(time.time())}.json"
    log_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[DONE] SAE decomposition complete. Results: {log_path}")


if __name__ == "__main__":
    main()
