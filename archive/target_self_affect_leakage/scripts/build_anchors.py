"""Build and save anchor embedding vectors for fear / calm projection axis.

Usage:
    python -m scripts.build_anchors [--fear data/anchors/fear_anchors.md] [--calm data/anchors/calm_anchors.md]
    python -m scripts.build_anchors --model all-MiniLM-L6-v2
    python -m scripts.build_anchors --verify   # print similarity diagnostics

Outputs:
    data/anchors/fear_mean.npy
    data/anchors/calm_mean.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_sentences(path: Path) -> list[str]:
    sentences = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("---"):
            sentences.append(line)
    if not sentences:
        raise ValueError(f"No sentences found in {path}")
    return sentences


def embed_sentences(sentences: list[str], model_name: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fear", default="data/anchors/fear_anchors.md")
    parser.add_argument("--calm", default="data/anchors/calm_anchors.md")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--verify", action="store_true", help="Print sanity-check similarities.")
    args = parser.parse_args()

    fear_path = Path(args.fear)
    calm_path = Path(args.calm)

    print(f"[anchors] loading fear sentences from {fear_path}", flush=True)
    fear_sentences = load_sentences(fear_path)
    print(f"[anchors] {len(fear_sentences)} fear sentences", flush=True)

    print(f"[anchors] loading calm sentences from {calm_path}", flush=True)
    calm_sentences = load_sentences(calm_path)
    print(f"[anchors] {len(calm_sentences)} calm sentences", flush=True)

    print(f"[anchors] embedding with model: {args.model}", flush=True)
    fear_vecs = embed_sentences(fear_sentences, args.model)
    calm_vecs = embed_sentences(calm_sentences, args.model)

    fear_mean = fear_vecs.mean(axis=0)
    calm_mean = calm_vecs.mean(axis=0)

    fear_mean /= np.linalg.norm(fear_mean)
    calm_mean /= np.linalg.norm(calm_mean)

    out_dir = fear_path.parent
    fear_out = out_dir / "fear_mean.npy"
    calm_out = out_dir / "calm_mean.npy"

    np.save(fear_out, fear_mean)
    np.save(calm_out, calm_mean)
    print(f"[anchors] saved: {fear_out}", flush=True)
    print(f"[anchors] saved: {calm_out}", flush=True)

    if args.verify:
        _verify(fear_sentences, calm_sentences, fear_mean, calm_mean, args.model)


def _verify(fear_sents: list[str], calm_sents: list[str], fear_mean: np.ndarray, calm_mean: np.ndarray, model_name: str) -> None:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    test_cases = [
        ("He lay in the mud waiting for a bullet he felt was inevitable.", "fear"),
        ("The ceasefire held and the silence was restful.", "calm"),
        ("The artillery never stopped and he could not remember sleep.", "fear"),
        ("She put down the letter and felt, finally, at peace.", "calm"),
    ]

    print("\n[verify] projection diagnostics:", flush=True)
    for text, expected in test_cases:
        vec = model.encode([text], normalize_embeddings=True)[0]
        fear_score = float(np.dot(vec, fear_mean))
        calm_score = float(np.dot(vec, calm_mean))
        efs_fear = fear_score - calm_score
        status = "OK" if (efs_fear > 0) == (expected == "fear") else "WARN"
        print(f"  [{status}] expected={expected} efs_fear={efs_fear:+.3f} | {text[:60]}", flush=True)


if __name__ == "__main__":
    main()
