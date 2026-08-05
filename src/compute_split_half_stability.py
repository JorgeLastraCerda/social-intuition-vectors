"""Split-half cosine stability check for warmth/competence direction vectors.

Source and background: written 2026-08-04 to close a reproducibility gap found
during a manuscript audit. The five-checks list in
`paper/paper/Ulu_Lastra.tex` (Methods) and the original findings report
`paper/2026-06-16_2001_concept_stories_probe_findings.md` both report
split-half cosine values (warmth 0.83, competence 0.88 for Gemma-3-12B-it),
but no corresponding code existed anywhere in the repository or its git
history (checked locally and on the SCCKN cluster, including
`git log -S` across all commits). This script reproduces the check from the
already-extracted concept-vector arrays under `data/processed/concept_vectors*/`
so the result is verifiable and re-runnable going forward.

Method: for each condition (high_warmth, low_warmth, high_competence,
low_competence), the 50 story vectors are split via a single random
permutation into two halves of 25. Half A's warmth direction is
mean(high_warmth half A) - mean(low_warmth half A); half B's warmth
direction is built the same way from the other half. The reported statistic
is the cosine similarity between the two independently-built half
directions. Same procedure for competence. No GPU or cluster access is
required; this only reads already-extracted `.npy` arrays.

Usage:
    python3 src/compute_split_half_stability.py \
        --vec-dir data/processed/concept_vectors \
        --label gemma3_12b \
        --out results/logs/split_half_stability_gemma3_12b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def split_half_direction_cosine(
    x_high: np.ndarray, x_low: np.ndarray, rng: np.random.Generator
) -> float:
    n_high, n_low = x_high.shape[0], x_low.shape[0]
    half_high = n_high // 2
    half_low = n_low // 2

    perm_high = rng.permutation(n_high)
    perm_low = rng.permutation(n_low)

    dir_a = (
        x_high[perm_high[:half_high]].mean(axis=0)
        - x_low[perm_low[:half_low]].mean(axis=0)
    )
    dir_b = (
        x_high[perm_high[half_high:]].mean(axis=0)
        - x_low[perm_low[half_low:]].mean(axis=0)
    )
    return cosine(dir_a, dir_b)


def run(vec_dir: Path, label: str, seed: int, out_path: Path) -> dict:
    meta = json.loads((vec_dir / "meta.json").read_text(encoding="utf-8"))
    x_high_warmth = np.load(vec_dir / "X_high_warmth.npy")
    x_low_warmth = np.load(vec_dir / "X_low_warmth.npy")
    x_high_comp = np.load(vec_dir / "X_high_competence.npy")
    x_low_comp = np.load(vec_dir / "X_low_competence.npy")

    rng = np.random.default_rng(seed)
    warmth_cos = split_half_direction_cosine(x_high_warmth, x_low_warmth, rng)
    comp_cos = split_half_direction_cosine(x_high_comp, x_low_comp, rng)

    result = {
        "label": label,
        "model": meta.get("model"),
        "seed": seed,
        "n_high_warmth": int(x_high_warmth.shape[0]),
        "n_low_warmth": int(x_low_warmth.shape[0]),
        "n_high_competence": int(x_high_comp.shape[0]),
        "n_low_competence": int(x_low_comp.shape[0]),
        "split_half_cosine_warmth": round(warmth_cos, 6),
        "split_half_cosine_competence": round(comp_cos, 6),
        "source_script": "src/compute_split_half_stability.py",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{label}] warmth cos={warmth_cos:.4f}  competence cos={comp_cos:.4f}  "
        f"-> {out_path}"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--vec-dir", type=Path, required=True)
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.vec_dir, args.label, args.seed, args.out)
