from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from src.utils.config import load_config

CONDITIONS = ("high_warmth", "low_warmth", "high_competence", "low_competence")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def project_out(vector: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Remove each (unit) PCA direction from the vector."""
    v = np.asarray(vector, np.float64).copy()
    for comp in components:
        v = v - (v @ comp) * comp
    return v


def select_k(explained_variance_ratio: np.ndarray, threshold: float) -> int:
    cumulative = np.cumsum(explained_variance_ratio)
    return int(np.searchsorted(cumulative, threshold) + 1)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    vdir = Path(cfg.paths.processed) / args.vectors_subdir

    Xn = np.load(vdir / "X_neutral.npy")
    warm = np.load(vdir / "warmth_vec.npy")
    comp = np.load(vdir / "competence_vec.npy")
    cond = {c: np.load(vdir / f"X_{c}.npy") for c in CONDITIONS}
    thr = cfg.neutral.variance_threshold

    pca = PCA(random_state=cfg.probing.seed).fit(Xn)
    k = select_k(pca.explained_variance_ratio_, thr)
    comps = pca.components_[:k]  # orthonormal directions
    kept = float(pca.explained_variance_ratio_[:k].sum())
    print(f"[pca] neutral n={Xn.shape[0]} d={Xn.shape[1]} -> k={k} components "
          f"cover {kept:.3f} variance (threshold {thr})")

    warm_d = project_out(warm, comps)
    comp_d = project_out(comp, comps)

    def report(tag: str, wv: np.ndarray, cv: np.ndarray) -> float:
        cw = cosine(wv, cv)
        d_w = cohens_d(cond["high_warmth"] @ wv, cond["low_warmth"] @ wv)
        d_c = cohens_d(cond["high_competence"] @ cv, cond["low_competence"] @ cv)
        # valence-leak diagnostic: warmth vector separating the competence pairs
        leak = cohens_d(cond["high_competence"] @ wv, cond["low_competence"] @ wv)
        print(f"  [{tag:8}] cos(w,c)={cw:+.3f}   d_warmth={d_w:5.2f}   "
              f"d_competence={d_c:5.2f}   warmth-on-competence(leak)={leak:5.2f}")
        return cw

    print("\nBEFORE vs AFTER denoising:")
    cos_before = report("raw", warm, comp)
    cos_after = report("denoised", warm_d, comp_d)

    np.savez(
        vdir / "concept_vectors_denoised.npz",
        warmth=warm_d, competence=comp_d, neutral_pca_components=comps,
        k=k, variance_threshold=thr, cosine_before=cos_before, cosine_after=cos_after,
    )
    json.dump(
        {"k": k, "variance_kept": kept, "cosine_before": cos_before, "cosine_after": cos_after},
        (vdir / "denoise_summary.json").open("w"), indent=2,
    )
    print(f"\n[DONE] saved concept_vectors_denoised.npz "
          f"(k={k}, cos {cos_before:+.3f} -> {cos_after:+.3f})")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PCA valence-denoising of warmth/competence vectors.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument(
        "--vectors-subdir",
        default="concept_vectors",
        help="Model-specific vector directory under cfg.paths.processed.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    main()
