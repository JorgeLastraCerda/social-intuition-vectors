from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.gemma_scope_utils import (
    CONDITIONS,
    check_file_size,
    cohens_d,
    condition_slices,
    cosine,
    decompose_feature_axes,
    load_story_records,
    sparse_mean,
)
from src.utils.config import load_config
from src.validate_probes import projected_cv_accuracy


def load_sae(release: str, sae_id: str, device: str):
    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise ImportError("Install sae-lens before running Gemma Scope analysis.") from exc
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()
    return sae, cfg_dict, sparsity


def encode_activations(
    sae,
    activations: np.ndarray,
    device: str,
    chunk_size: int,
) -> tuple[sparse.csr_matrix, dict[str, float]]:
    chunks: list[sparse.csr_matrix] = []
    relative_errors: list[np.ndarray] = []
    reconstruction_cosines: list[np.ndarray] = []
    active_counts: list[np.ndarray] = []
    sae_dtype = sae.W_dec.dtype
    with torch.no_grad():
        for start in range(0, len(activations), chunk_size):
            batch = torch.from_numpy(activations[start:start + chunk_size]).to(
                device=device,
                dtype=sae_dtype,
            )
            features = sae.encode(batch)
            reconstruction = sae.decode(features)
            residual = reconstruction - batch
            rel = residual.norm(dim=-1) / (batch.norm(dim=-1) + 1e-12)
            cos = torch.nn.functional.cosine_similarity(reconstruction, batch, dim=-1)
            relative_errors.append(rel.float().cpu().numpy())
            reconstruction_cosines.append(cos.float().cpu().numpy())
            active_counts.append((features != 0).sum(dim=-1).float().cpu().numpy())
            chunks.append(sparse.csr_matrix(features.float().cpu().numpy()))
            del batch, features, reconstruction, residual
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    matrix = sparse.vstack(chunks, format="csr")
    return matrix, {
        "relative_l2_error_mean": float(np.concatenate(relative_errors).mean()),
        "relative_l2_error_std": float(np.concatenate(relative_errors).std()),
        "reconstruction_cosine_mean": float(np.concatenate(reconstruction_cosines).mean()),
        "active_features_mean": float(np.concatenate(active_counts).mean()),
        "active_features_std": float(np.concatenate(active_counts).std()),
    }


def full_feature_cv(
    high: sparse.csr_matrix,
    low: sparse.csr_matrix,
    seed: int,
    groups_high: np.ndarray | None = None,
    groups_low: np.ndarray | None = None,
) -> tuple[float, float]:
    X = sparse.vstack([high, low], format="csr")
    y = np.array([1] * high.shape[0] + [0] * low.shape[0])
    estimator = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            max_iter=3000,
            C=1.0,
            solver="liblinear",
            random_state=seed,
        ),
    )
    standard = cross_val_score(
        estimator,
        X,
        y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring="accuracy",
    )
    topic_mean = float("nan")
    if groups_high is not None and groups_low is not None:
        groups = np.concatenate([groups_high, groups_low])
        topic = cross_val_score(
            estimator,
            X,
            y,
            groups=groups,
            cv=GroupKFold(n_splits=5),
            scoring="accuracy",
        )
        topic_mean = float(topic.mean())
    return float(standard.mean()), topic_mean


def decode_direction(sae, vector: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(vector, dtype=np.float32)).to(
        device=sae.W_dec.device,
        dtype=sae.W_dec.dtype,
    )
    with torch.no_grad():
        decoded = tensor @ sae.W_dec
    return decoded.float().cpu().numpy()


def top_feature_rows(
    feature_matrix: sparse.csr_matrix,
    vectors: dict[str, np.ndarray],
    story_records: list[dict],
    model: str,
    layer: int,
    width_label: str,
    neuronpedia_base: str | None,
    top_k: int,
) -> list[dict]:
    rows: list[dict] = []
    for vector_name, vector in vectors.items():
        order = np.argsort(np.abs(vector))[::-1][:top_k]
        for rank, feature_idx in enumerate(order, start=1):
            profile = feature_matrix[:, feature_idx].toarray().ravel()
            top_story_order = np.argsort(profile)[::-1][:5]
            url = (
                f"https://www.neuronpedia.org/{neuronpedia_base}/{int(feature_idx)}"
                if neuronpedia_base
                else ""
            )
            rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "width": width_label,
                    "vector": vector_name,
                    "rank": rank,
                    "feature_idx": int(feature_idx),
                    "effect": float(vector[feature_idx]),
                    "mean_activation": float(profile.mean()),
                    "activation_frequency": float(np.mean(profile != 0)),
                    "top_story_ids": ";".join(
                        story_records[i]["id"] for i in top_story_order
                    ),
                    "top_story_conditions": ";".join(
                        story_records[i]["condition"] for i in top_story_order
                    ),
                    "neuronpedia_url": url,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg.probing.seed
    vectors_dir = Path(cfg.paths.processed) / args.vectors_subdir
    output_dir = Path(cfg.paths.processed) / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = Path(cfg.paths.results) / "tables"
    log_dir = Path(cfg.paths.logs)
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((vectors_dir / "meta.json").read_text(encoding="utf-8"))
    layer = int(meta["probe_layer"])
    model = str(meta["model"])
    expected_prefix = f"layer_{layer}_"
    sae_ids = [item.strip() for item in args.sae_ids.split(",") if item.strip()]
    if any(not sae_id.startswith(expected_prefix) for sae_id in sae_ids):
        raise ValueError(
            f"All SAE IDs must match probe layer {layer}: {sae_ids}"
        )

    records_by_condition = load_story_records(
        Path(cfg.paths.stimuli) / "concept_stories.jsonl"
    )
    counts = {condition: len(records_by_condition[condition]) for condition in CONDITIONS}
    slices = condition_slices(counts)
    story_records = [
        record
        for condition in CONDITIONS
        for record in records_by_condition[condition]
    ]
    activations_by_condition = {
        condition: np.load(vectors_dir / f"X_{condition}.npy").astype(np.float32)
        for condition in CONDITIONS
    }
    activations = np.concatenate(
        [activations_by_condition[condition] for condition in CONDITIONS],
        axis=0,
    )
    raw_warmth = np.load(vectors_dir / "warmth_vec.npy").astype(np.float32)
    raw_competence = np.load(vectors_dir / "competence_vec.npy").astype(np.float32)
    topic_groups = {
        condition: np.array(
            [int(record["topic_idx"]) for record in records_by_condition[condition]]
        )
        for condition in CONDITIONS
    }

    metrics_rows: list[dict] = []
    all_top_rows: list[dict] = []
    width_outputs: dict[str, dict] = {}
    for sae_id in sae_ids:
        width_label = sae_id.split("_width_", 1)[1].split("_", 1)[0]
        print(f"[sae] loading {args.sae_release} / {sae_id}", flush=True)
        sae, sae_cfg, _ = load_sae(args.sae_release, sae_id, cfg.model.device)
        if int(sae.cfg.d_in) != int(meta["d_model"]):
            raise ValueError(
                f"SAE d_in={sae.cfg.d_in} does not match activation width "
                f"{meta['d_model']}"
            )
        feature_matrix, reconstruction = encode_activations(
            sae,
            activations,
            cfg.model.device,
            args.chunk_size,
        )
        feature_path = output_dir / f"story_features_{width_label}.npz"
        sparse.save_npz(feature_path, feature_matrix, compressed=True)
        check_file_size(feature_path)

        condition_features = {
            condition: feature_matrix[slices[condition]]
            for condition in CONDITIONS
        }
        warmth = (
            sparse_mean(condition_features["high_warmth"])
            - sparse_mean(condition_features["low_warmth"])
        )
        competence = (
            sparse_mean(condition_features["high_competence"])
            - sparse_mean(condition_features["low_competence"])
        )
        vectors = decompose_feature_axes(warmth, competence)
        decoded = {
            name: decode_direction(sae, vector)
            for name, vector in vectors.items()
        }
        vector_path = output_dir / f"vectors_{width_label}.npz"
        np.savez_compressed(
            vector_path,
            **{f"feature_{name}": value for name, value in vectors.items()},
            **{f"residual_{name}": value for name, value in decoded.items()},
        )
        check_file_size(vector_path)

        warmth_scores_high = condition_features["high_warmth"] @ vectors["warmth"]
        warmth_scores_low = condition_features["low_warmth"] @ vectors["warmth"]
        comp_scores_high = (
            condition_features["high_competence"] @ vectors["competence"]
        )
        comp_scores_low = condition_features["low_competence"] @ vectors["competence"]
        warmth_cv, warmth_topic_cv = full_feature_cv(
            condition_features["high_warmth"],
            condition_features["low_warmth"],
            seed,
            topic_groups["high_warmth"],
            topic_groups["low_warmth"],
        )
        comp_cv, comp_topic_cv = full_feature_cv(
            condition_features["high_competence"],
            condition_features["low_competence"],
            seed,
            topic_groups["high_competence"],
            topic_groups["low_competence"],
        )
        cross_w_on_c = projected_cv_accuracy(
            np.asarray(
                condition_features["high_competence"] @ vectors["warmth"]
            ).reshape(-1, 1),
            np.asarray(
                condition_features["low_competence"] @ vectors["warmth"]
            ).reshape(-1, 1),
            np.array([1.0]),
            seed,
        )
        cross_c_on_w = projected_cv_accuracy(
            np.asarray(
                condition_features["high_warmth"] @ vectors["competence"]
            ).reshape(-1, 1),
            np.asarray(
                condition_features["low_warmth"] @ vectors["competence"]
            ).reshape(-1, 1),
            np.array([1.0]),
            seed,
        )
        neuronpedia_base = (
            args.neuronpedia_base.format(width=width_label)
            if args.neuronpedia_base
            else None
        )
        if isinstance(sae_cfg, dict):
            neuronpedia_base = neuronpedia_base or sae_cfg.get("neuronpedia")
        alignment_w = cosine(decoded["warmth"], raw_warmth)
        alignment_c = cosine(decoded["competence"], raw_competence)
        metrics = {
            "label": args.label,
            "model": model,
            "layer": layer,
            "sae_release": args.sae_release,
            "sae_id": sae_id,
            "width": width_label,
            "n_features": int(sae.cfg.d_sae),
            **reconstruction,
            "warmth_cohens_d": cohens_d(warmth_scores_high, warmth_scores_low),
            "competence_cohens_d": cohens_d(comp_scores_high, comp_scores_low),
            "warmth_cv": warmth_cv,
            "competence_cv": comp_cv,
            "warmth_topic_cv": warmth_topic_cv,
            "competence_topic_cv": comp_topic_cv,
            "cross_warmth_on_competence_cv": cross_w_on_c,
            "cross_competence_on_warmth_cv": cross_c_on_w,
            "feature_cosine": cosine(warmth, competence),
            "decoded_warmth_alignment": alignment_w,
            "decoded_competence_alignment": alignment_c,
            "shared_energy_fraction_warmth": float(
                np.square(vectors["shared"]).sum()
                / (np.square(vectors["warmth"]).sum() + 1e-12)
            ),
            "shared_energy_fraction_competence": float(
                np.square(vectors["shared"]).sum()
                / (np.square(vectors["competence"]).sum() + 1e-12)
            ),
        }
        metrics_rows.append(metrics)
        all_top_rows.extend(
            top_feature_rows(
                feature_matrix,
                vectors,
                story_records,
                model,
                layer,
                width_label,
                neuronpedia_base,
                args.top_k,
            )
        )
        width_outputs[width_label] = {
            "feature_file": str(feature_path),
            "vector_file": str(vector_path),
            "metrics": metrics,
        }
        del sae, feature_matrix, condition_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_path = table_dir / f"gemma_scope_metrics_{args.label}.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    top_path = table_dir / f"gemma_scope_top_features_{args.label}.csv"
    with top_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_top_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_top_rows)
    log = {
        "label": args.label,
        "model": model,
        "probe_layer": layer,
        "seed": seed,
        "sae_release": args.sae_release,
        "sae_ids": sae_ids,
        "condition_order": list(CONDITIONS),
        "story_ids": [record["id"] for record in story_records],
        "outputs": width_outputs,
    }
    log_path = log_dir / f"gemma_scope_analysis_{args.label}.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[done] {metrics_path}")
    print(f"[done] {top_path}")
    print(f"[done] {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse concept activations with Gemma Scope 2 residual SAEs."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--vectors-subdir", required=True)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-ids", required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--neuronpedia-base",
        help=(
            "Optional Neuronpedia model/SAE path, without feature index. "
            "The placeholder {width} is replaced with 16k, 65k, or 262k. "
            "Leave unset when the release has no published mapping."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
