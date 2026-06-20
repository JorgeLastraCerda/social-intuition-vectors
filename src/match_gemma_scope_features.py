from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from src.gemma_scope_utils import check_file_size
from src.utils.config import load_config


VECTOR_NAMES = (
    "warmth",
    "competence",
    "shared",
    "warmth_specific",
    "competence_specific",
)


def top_nonzero_indices(vector: np.ndarray, top_k: int) -> np.ndarray:
    order = np.argsort(np.abs(vector))[::-1]
    nonzero = order[np.abs(vector[order]) > 0]
    return nonzero[:top_k].astype(np.int64)


def normalized_profiles(
    matrix: sparse.csr_matrix,
    feature_indices: np.ndarray,
) -> np.ndarray:
    profiles = matrix[:, feature_indices].toarray().T.astype(np.float64)
    profiles -= profiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    return profiles / np.maximum(norms, 1e-12)


def neuronpedia_url(base: str | None, feature_idx: int) -> str:
    if not base:
        return ""
    return f"https://www.neuronpedia.org/{base}/{feature_idx}"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    processed = Path(cfg.paths.processed)
    table_dir = Path(cfg.paths.results) / "tables"
    log_dir = Path(cfg.paths.logs)
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    left_dir = processed / args.left_subdir
    right_dir = processed / args.right_subdir
    left_matrix = sparse.load_npz(left_dir / "story_features_65k.npz").tocsr()
    right_matrix = sparse.load_npz(right_dir / "story_features_65k.npz").tocsr()
    if left_matrix.shape[0] != right_matrix.shape[0]:
        raise ValueError(
            "Feature matrices must contain the same stories in the same order: "
            f"{left_matrix.shape[0]} != {right_matrix.shape[0]}"
        )

    left_vectors = np.load(left_dir / "vectors_65k.npz")
    right_vectors = np.load(right_dir / "vectors_65k.npz")
    rows: list[dict] = []
    for vector_name in VECTOR_NAMES:
        left_vector = left_vectors[f"feature_{vector_name}"]
        right_vector = right_vectors[f"feature_{vector_name}"]
        left_indices = top_nonzero_indices(left_vector, args.top_k)
        right_indices = top_nonzero_indices(right_vector, args.top_k)
        left_profiles = normalized_profiles(left_matrix, left_indices)
        right_profiles = normalized_profiles(right_matrix, right_indices)
        similarity = left_profiles @ right_profiles.T
        left_assignment, right_assignment = linear_sum_assignment(-similarity)
        matched = sorted(
            zip(left_assignment, right_assignment),
            key=lambda pair: similarity[pair[0], pair[1]],
            reverse=True,
        )
        for rank, (left_pos, right_pos) in enumerate(matched, start=1):
            left_idx = int(left_indices[left_pos])
            right_idx = int(right_indices[right_pos])
            left_effect = float(left_vector[left_idx])
            right_effect = float(right_vector[right_idx])
            rows.append(
                {
                    "vector": vector_name,
                    "rank": rank,
                    "left_label": args.left_label,
                    "left_feature_idx": left_idx,
                    "left_effect": left_effect,
                    "left_neuronpedia_url": neuronpedia_url(
                        args.left_neuronpedia_base,
                        left_idx,
                    ),
                    "right_label": args.right_label,
                    "right_feature_idx": right_idx,
                    "right_effect": right_effect,
                    "right_neuronpedia_url": neuronpedia_url(
                        args.right_neuronpedia_base,
                        right_idx,
                    ),
                    "story_profile_correlation": float(
                        similarity[left_pos, right_pos]
                    ),
                    "effect_sign_agreement": bool(
                        np.sign(left_effect) == np.sign(right_effect)
                    ),
                }
            )

    output_path = table_dir / "gemma_scope_feature_matches_12b_27b.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    check_file_size(output_path)

    log = {
        "left_label": args.left_label,
        "right_label": args.right_label,
        "left_subdir": args.left_subdir,
        "right_subdir": args.right_subdir,
        "width": "65k",
        "top_k_per_vector": args.top_k,
        "matching": (
            "One-to-one Hungarian matching that maximizes centered cosine "
            "similarity of each feature's activation profile over the same stories."
        ),
        "output": str(output_path),
    }
    log_path = log_dir / "gemma_scope_feature_matching_12b_27b.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[done] {output_path}")
    print(f"[done] {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match Gemma Scope features across model scales by story profiles."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--left-subdir", required=True)
    parser.add_argument("--right-subdir", required=True)
    parser.add_argument("--left-label", default="gemma3_12b")
    parser.add_argument("--right-label", default="gemma3_27b")
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--left-neuronpedia-base")
    parser.add_argument("--right-neuronpedia-base")
    return parser.parse_args()


if __name__ == "__main__":
    main()
