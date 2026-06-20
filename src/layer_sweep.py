"""layer_sweep.py — Sweep all residual-stream layers for warmth/competence probeability.

For each layer L, builds mean-diff warmth and competence vectors from the 200-story
corpus and evaluates: topic-holdout CV (GroupKFold), Cohen's d, cos(W,C), and
mean residual-stream norm (for scale normalisation, B4).

All 200 stories are processed in a single forward pass per story (all-layer cache),
keeping peak memory flat regardless of model depth.  No full activation tensors are
written to disk — only the per-layer metric CSV is persisted.

Usage (SCCKN / local):
    python src/layer_sweep.py --config config/config.yaml                          # Gemma
    python src/layer_sweep.py --config config/config.yaml --model Qwen/Qwen3-14B  \\
        --label qwen3_14b
    python src/layer_sweep.py --config config/config.yaml \\
        --model meta-llama/Llama-3.1-8B-Instruct --label llama31_8b
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.utils.config import load_config
from src.utils.hooks import layer_from_fraction, mean_activation_after_token
from src.utils.model_loader import load_hooked_model

# Import helpers from validate_probes (topic-holdout) and extract_vectors (story loader).
from src.validate_probes import load_topic_groups, topic_holdout_cv
from src.extract_vectors import load_stories

EXPECTED_CONDITIONS = ("high_warmth", "low_warmth", "high_competence", "low_competence")


# ---------------------------------------------------------------------------
# All-layer activation extraction
# ---------------------------------------------------------------------------

def extract_all_layers(
    model,
    texts: list[str],
    start_token: int,
    n_layers: int,
) -> np.ndarray:
    """Extract mean-pooled activations at every residual layer for each story.

    Returns:
        acts: float32 array [n_layers, n_stories, d_model]
    Memory strategy: process one story at a time; keep only mean-pooled vectors
    per layer (never accumulate full [seq, d_model] tensors across stories).
    """
    d_model = model.cfg.d_model
    acts = np.zeros((n_layers, len(texts), d_model), dtype=np.float32)

    for story_idx, text in enumerate(texts):
        tokens = model.to_tokens(text, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: n.endswith("hook_resid_post"),
                return_type=None,
            )
        for layer_idx in range(n_layers):
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            layer_acts = cache[hook_name]          # [1, seq, d_model]
            vec = mean_activation_after_token(layer_acts, start_token).squeeze(0)  # [d_model]
            acts[layer_idx, story_idx] = vec.float().cpu().numpy()

        if (story_idx + 1) % 20 == 0:
            print(f"  [sweep] {story_idx + 1}/{len(texts)} stories done", flush=True)

    return acts  # [n_layers, n_stories, d_model]


# ---------------------------------------------------------------------------
# Per-layer metrics
# ---------------------------------------------------------------------------

def cohens_d_1d(proj_high: np.ndarray, proj_low: np.ndarray) -> float:
    pooled_std = float(np.sqrt((proj_high.var() + proj_low.var()) / 2.0) + 1e-12)
    return float((proj_high.mean() - proj_low.mean()) / pooled_std)


def sweep_metrics_at_layer(
    layer_idx: int,
    acts: np.ndarray,          # [n_layers, n_stories, d_model]
    buckets: dict[str, list],  # condition -> list[story_index_in_concat]
    topic_groups: dict[str, np.ndarray],
    n_layers: int,
    probe_layer: int,
) -> dict:
    """Compute all metrics for a single layer from pre-extracted activations."""
    # Slice per-condition activations for this layer.
    X: dict[str, np.ndarray] = {}
    for cond in EXPECTED_CONDITIONS:
        indices = buckets[cond]
        X[cond] = acts[layer_idx, indices, :]   # [n_cond, d_model]

    # Mean-diff vectors (same as extract_vectors.py:108).
    warmth_vec = X["high_warmth"].mean(axis=0) - X["low_warmth"].mean(axis=0)
    comp_vec   = X["high_competence"].mean(axis=0) - X["low_competence"].mean(axis=0)
    wv = warmth_vec / (np.linalg.norm(warmth_vec) + 1e-12)
    cv = comp_vec   / (np.linalg.norm(comp_vec)   + 1e-12)

    # cos(W, C)
    cos_wc = float(np.dot(wv, cv))

    # Cohen's d (1-D projection onto unit vector).
    w_proj_h = X["high_warmth"] @ wv
    w_proj_l = X["low_warmth"]  @ wv
    c_proj_h = X["high_competence"] @ cv
    c_proj_l = X["low_competence"]  @ cv
    warmth_d  = cohens_d_1d(w_proj_h, w_proj_l)
    comp_d    = cohens_d_1d(c_proj_h, c_proj_l)

    # Topic-holdout CV (GroupKFold, deterministic — no seed needed).
    w_th_mean, w_th_std, _ = topic_holdout_cv(
        X["high_warmth"], X["low_warmth"],
        topic_groups["high_warmth"], topic_groups["low_warmth"],
    )
    c_th_mean, c_th_std, _ = topic_holdout_cv(
        X["high_competence"], X["low_competence"],
        topic_groups["high_competence"], topic_groups["low_competence"],
    )

    # Mean residual-stream norm across all 200 stories at this layer
    # (used for scale normalisation: proj / E[||resid||]).
    all_acts = acts[layer_idx]   # [200, d_model]
    mean_resid_norm = float(np.linalg.norm(all_acts, axis=1).mean())

    frac = round(layer_idx / max(n_layers - 1, 1), 4)
    return {
        "layer":             layer_idx,
        "frac":              frac,
        "is_probe_layer":    (layer_idx == probe_layer),
        "warmth_topic_cv":   round(w_th_mean, 6),
        "warmth_topic_cv_std": round(w_th_std, 6),
        "comp_topic_cv":     round(c_th_mean, 6),
        "comp_topic_cv_std": round(c_th_std, 6),
        "warmth_cohens_d":   round(warmth_d, 6),
        "comp_cohens_d":     round(comp_d, 6),
        "cos_wc":            round(cos_wc, 6),
        "mean_resid_norm":   round(mean_resid_norm, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.model is not None:
        cfg = replace(cfg, model=replace(cfg.model, name=args.model))
        print(f"[override] model set to: {cfg.model.name}")

    label = args.label or cfg.model.name.replace("/", "_")
    label_suffix = f"_{label}" if label else ""

    stimuli_path = Path(cfg.paths.stimuli) / "concept_stories.jsonl"
    print(f"[stimuli] {stimuli_path}")
    buckets_text = load_stories(stimuli_path)  # condition -> list[str]

    # Build per-condition integer indices into the concatenated-all-stories array
    # so we can slice acts[layer_idx, indices, :] cheaply.
    story_indices: dict[str, list[int]] = {}
    offset = 0
    for cond in EXPECTED_CONDITIONS:
        n = len(buckets_text[cond])
        story_indices[cond] = list(range(offset, offset + n))
        offset += n
    all_texts: list[str] = []
    for cond in EXPECTED_CONDITIONS:
        all_texts.extend(buckets_text[cond])
    n_all = len(all_texts)

    # Topic groups for holdout CV.
    topic_groups = load_topic_groups(stimuli_path)
    # Validate alignment.
    for cond in EXPECTED_CONDITIONS:
        if len(story_indices[cond]) != len(topic_groups[cond]):
            raise ValueError(
                f"Alignment error: {cond} has {len(story_indices[cond])} stories "
                f"but {len(topic_groups[cond])} topic-group entries."
            )

    print(f"[model] loading {cfg.model.name} ...", flush=True)
    model = load_hooked_model(cfg)
    model.eval()

    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model
    probe_layer = layer_from_fraction(n_layers, cfg.probing.probe_layer_frac)
    print(f"[model] n_layers={n_layers}, d_model={d_model}, probe_layer={probe_layer} (frac={cfg.probing.probe_layer_frac})")
    print(f"[sweep] extracting all {n_layers} layers for {n_all} stories ...", flush=True)

    t0 = time.time()
    acts = extract_all_layers(model, all_texts, cfg.probing.start_token, n_layers)
    print(f"[sweep] extraction done in {time.time()-t0:.1f}s  acts.shape={acts.shape}")

    # Free model VRAM before the CPU-bound metric computation.
    del model
    torch.cuda.empty_cache()

    # Per-layer metrics.
    rows: list[dict] = []
    print(f"[sweep] computing metrics per layer ...", flush=True)
    for layer_idx in range(n_layers):
        row = sweep_metrics_at_layer(
            layer_idx, acts, story_indices, topic_groups, n_layers, probe_layer
        )
        rows.append(row)
        marker = " <-- probe_layer_frac=0.66" if row["is_probe_layer"] else ""
        print(
            f"  L{layer_idx:03d}  frac={row['frac']:.2f}  "
            f"w_cv={row['warmth_topic_cv']:.3f}  c_cv={row['comp_topic_cv']:.3f}  "
            f"w_d={row['warmth_cohens_d']:.2f}  cos={row['cos_wc']:.3f}  "
            f"norm={row['mean_resid_norm']:.1f}{marker}",
            flush=True,
        )

    # Write CSV.
    out_csv_default = Path(cfg.paths.results) / "tables" / f"layer_sweep{label_suffix}.csv"
    out_csv = Path(args.out_csv) if args.out_csv else out_csv_default
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[DONE] layer sweep CSV: {out_csv}")

    # Summary: best layers per metric.
    best_w = max(rows, key=lambda r: r["warmth_topic_cv"])
    best_c = max(rows, key=lambda r: r["comp_topic_cv"])
    probe_row = next(r for r in rows if r["is_probe_layer"])
    print("\n--- SUMMARY ---")
    print(f"  Best warmth topic-CV   : L{best_w['layer']} (frac={best_w['frac']:.2f})  cv={best_w['warmth_topic_cv']:.4f}  d={best_w['warmth_cohens_d']:.2f}")
    print(f"  Best comp  topic-CV    : L{best_c['layer']} (frac={best_c['frac']:.2f})  cv={best_c['comp_topic_cv']:.4f}  d={best_c['comp_cohens_d']:.2f}")
    print(f"  probe_layer_frac=0.66  : L{probe_row['layer']} (frac={probe_row['frac']:.2f})  w_cv={probe_row['warmth_topic_cv']:.4f}  c_cv={probe_row['comp_topic_cv']:.4f}  cos={probe_row['cos_wc']:.3f}  norm={probe_row['mean_resid_norm']:.1f}")

    # Write meta alongside the CSV (same directory, for provenance).
    meta_out = out_csv.with_suffix(".meta.json")
    meta_out.write_text(json.dumps({
        "model": cfg.model.name,
        "n_layers": n_layers,
        "d_model": d_model,
        "probe_layer": probe_layer,
        "probe_layer_frac": cfg.probing.probe_layer_frac,
        "start_token": cfg.probing.start_token,
        "seed": cfg.probing.seed,
        "n_stories": n_all,
        "label": label,
        "timestamp": int(time.time()),
    }, indent=2), encoding="utf-8")
    print(f"[DONE] meta: {meta_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep all residual layers for warmth/competence probeability.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--model", default=None,
        help="Override cfg.model.name (e.g. Qwen/Qwen3-14B).  "
             "Use with --label to keep outputs separate.",
    )
    parser.add_argument(
        "--label", default=None,
        help="Short label for output filenames (e.g. qwen3_14b). "
             "Defaults to model name with / replaced by _.",
    )
    parser.add_argument(
        "--out-csv", default=None,
        help="Override default output path (results/tables/layer_sweep_<label>.csv).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
