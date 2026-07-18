from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config


EXPECTED_CONDITIONS = ("high_warmth", "low_warmth", "high_competence", "low_competence")


def load_topic_groups(stimuli_path: Path) -> dict[str, np.ndarray]:
    """Return topic_idx arrays per condition, in the same row order as X_<cond>.npy.

    Reads concept_stories.jsonl sequentially and buckets topic_idx by condition,
    mirroring the exact pass that extract_vectors.load_stories uses.  Row i of the
    returned array for a condition == row i of X_<cond>.npy.
    """
    buckets: dict[str, list[int]] = {c: [] for c in EXPECTED_CONDITIONS}
    with stimuli_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cond = record["condition"]
            if cond in buckets:
                buckets[cond].append(int(record["topic_idx"]))
    groups: dict[str, np.ndarray] = {}
    for cond, idxs in buckets.items():
        if not idxs:
            raise ValueError(f"load_topic_groups: no stories found for condition {cond!r} in {stimuli_path}")
        groups[cond] = np.array(idxs, dtype=np.int64)
    return groups


def topic_holdout_cv(
    X_high: np.ndarray,
    X_low: np.ndarray,
    groups_high: np.ndarray,
    groups_low: np.ndarray,
    n_splits: int = 5,
) -> tuple[float, float, list[float]]:
    """Full-feature topic-holdout cross-validation (GroupKFold, deterministic).

    Groups ensure a topic's high and low stories are always in the same fold,
    preventing the probe from exploiting shared topic vocabulary.
    """
    X = np.concatenate([X_high, X_low], axis=0)
    y = np.array([1] * len(X_high) + [0] * len(X_low), dtype=np.int64)
    groups = np.concatenate([groups_high, groups_low])
    n_distinct = len(set(groups.tolist()))
    if n_distinct < n_splits:
        raise ValueError(
            f"topic_holdout_cv: only {n_distinct} distinct topics but n_splits={n_splits}; "
            "need at least as many topics as folds."
        )
    lr = LogisticRegression(max_iter=1000, C=1.0)
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(lr, X, y, cv=cv, groups=groups, scoring="accuracy")
    return float(scores.mean()), float(scores.std()), [round(float(s), 6) for s in scores.tolist()]


def direction_topic_holdout_cv(
    X_high: np.ndarray,
    X_low: np.ndarray,
    groups_high: np.ndarray,
    groups_low: np.ndarray,
    seed: int,
    n_splits: int = 5,
) -> tuple[float, float, list[float]]:
    """Validate a mean-difference direction rebuilt inside each topic fold.

    Unlike ``topic_holdout_cv``, which fits an unrestricted full-feature linear
    classifier, this routine evaluates the exact Stage 1 construction rule. The
    direction, projection standardisation, and decision boundary are learned
    only from training topics before scoring the held-out topics.
    """
    X = np.concatenate([X_high, X_low], axis=0)
    y = np.array([1] * len(X_high) + [0] * len(X_low), dtype=np.int64)
    groups = np.concatenate([groups_high, groups_low])
    cv = GroupKFold(n_splits=n_splits)
    scores: list[float] = []
    for train_idx, test_idx in cv.split(X, y, groups):
        train_high = X[train_idx][y[train_idx] == 1]
        train_low = X[train_idx][y[train_idx] == 0]
        direction = train_high.mean(axis=0) - train_low.mean(axis=0)
        unit_direction = direction / (np.linalg.norm(direction) + 1e-12)
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed, C=1.0),
        )
        estimator.fit((X[train_idx] @ unit_direction).reshape(-1, 1), y[train_idx])
        scores.append(
            float(estimator.score((X[test_idx] @ unit_direction).reshape(-1, 1), y[test_idx]))
        )
    scores_np = np.asarray(scores, dtype=np.float64)
    return (
        float(scores_np.mean()),
        float(scores_np.std()),
        [round(float(score), 6) for score in scores],
    )


def topic_cross_axis_transfer_cv(
    source_high: np.ndarray,
    source_low: np.ndarray,
    target_high: np.ndarray,
    target_low: np.ndarray,
    source_groups_high: np.ndarray,
    source_groups_low: np.ndarray,
    target_groups_high: np.ndarray,
    target_groups_low: np.ndarray,
    seed: int,
    n_splits: int = 5,
) -> tuple[float, float, list[float]]:
    """Train on one axis and test the other on held-out topics without recalibration."""
    source_X = np.concatenate([source_high, source_low], axis=0)
    source_y = np.array([1] * len(source_high) + [0] * len(source_low), dtype=np.int64)
    source_groups = np.concatenate([source_groups_high, source_groups_low])
    target_topics = set(target_groups_high.tolist()) | set(target_groups_low.tolist())
    if set(source_groups.tolist()) != target_topics:
        raise ValueError("Source and target axes must contain the same topic set for transfer CV.")

    cv = GroupKFold(n_splits=n_splits)
    scores: list[float] = []
    for train_idx, heldout_idx in cv.split(source_X, source_y, source_groups):
        heldout_topics = np.unique(source_groups[heldout_idx])
        train_high = source_X[train_idx][source_y[train_idx] == 1]
        train_low = source_X[train_idx][source_y[train_idx] == 0]
        direction = train_high.mean(axis=0) - train_low.mean(axis=0)
        unit_direction = direction / (np.linalg.norm(direction) + 1e-12)
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed, C=1.0),
        )
        estimator.fit(
            (source_X[train_idx] @ unit_direction).reshape(-1, 1),
            source_y[train_idx],
        )

        target_high_mask = np.isin(target_groups_high, heldout_topics)
        target_low_mask = np.isin(target_groups_low, heldout_topics)
        target_X = np.concatenate(
            [target_high[target_high_mask], target_low[target_low_mask]], axis=0
        )
        target_y = np.array(
            [1] * int(target_high_mask.sum()) + [0] * int(target_low_mask.sum()),
            dtype=np.int64,
        )
        scores.append(
            float(estimator.score((target_X @ unit_direction).reshape(-1, 1), target_y))
        )

    scores_np = np.asarray(scores, dtype=np.float64)
    return (
        float(scores_np.mean()),
        float(scores_np.std()),
        [round(float(score), 6) for score in scores],
    )


def load_vectors(processed_dir: Path, subdir: str = "concept_vectors") -> dict:
    vec_dir = processed_dir / subdir
    data: dict = {}
    data["warmth_vec"] = np.load(vec_dir / "warmth_vec.npy")
    data["competence_vec"] = np.load(vec_dir / "competence_vec.npy")
    for cond in ("high_warmth", "low_warmth", "high_competence", "low_competence"):
        data[cond] = np.load(vec_dir / f"X_{cond}.npy")
    with (vec_dir / "meta.json").open(encoding="utf-8") as f:
        data["meta"] = json.load(f)
    return data


def probe_axis(
    X_high: np.ndarray,
    X_low: np.ndarray,
    vec: np.ndarray,
    label: str,
    seed: int,
    groups_high: np.ndarray | None = None,
    groups_low: np.ndarray | None = None,
) -> dict:
    unit_vec = vec / (np.linalg.norm(vec) + 1e-12)
    mean_high = X_high.mean(axis=0)
    mean_low = X_low.mean(axis=0)
    diff_norm = float(np.linalg.norm(mean_high - mean_low))
    denom = np.linalg.norm(mean_high) * np.linalg.norm(mean_low) + 1e-12
    cosine = float(np.dot(mean_high, mean_low) / denom)

    proj_high = X_high @ unit_vec
    proj_low = X_low @ unit_vec
    pooled_std = float(np.sqrt((proj_high.var() + proj_low.var()) / 2.0) + 1e-12)
    cohens_d = float((proj_high.mean() - proj_low.mean()) / pooled_std)

    X_np = np.concatenate([X_high, X_low], axis=0)
    y_np = np.array([1] * len(X_high) + [0] * len(X_low))
    lr = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(lr, X_np, y_np, cv=skf, scoring="accuracy")

    print(f"\n[{label}]")
    print(f"  diff_norm      : {diff_norm:.4f}")
    print(f"  cosine(H,L)    : {cosine:.6f}")
    print(f"  Cohen's d      : {cohens_d:.4f}")
    print(f"  5-fold CV      : {scores.mean():.4f} +/- {scores.std():.4f}  {[round(s, 3) for s in scores.tolist()]}")

    result = {
        "axis": label,
        "diff_norm": round(diff_norm, 6),
        "cosine_high_low": round(cosine, 6),
        "proj_high_mean": round(float(proj_high.mean()), 6),
        "proj_high_std": round(float(proj_high.std()), 6),
        "proj_low_mean": round(float(proj_low.mean()), 6),
        "proj_low_std": round(float(proj_low.std()), 6),
        "cohens_d": round(cohens_d, 6),
        "cv_mean": round(float(scores.mean()), 6),
        "cv_std": round(float(scores.std()), 6),
        "cv_folds": [round(float(s), 6) for s in scores.tolist()],
    }

    # Topic-level holdout (GroupKFold) — discriminative metric; avoids topic vocabulary leakage.
    if groups_high is not None and groups_low is not None:
        th_mean, th_std, th_folds = topic_holdout_cv(X_high, X_low, groups_high, groups_low)
        print(f"  topic-holdout  : {th_mean:.4f} +/- {th_std:.4f}  {[round(s, 3) for s in th_folds]}")
        result["topic_cv_mean"] = round(th_mean, 6)
        result["topic_cv_std"] = round(th_std, 6)
        result["topic_cv_folds"] = th_folds
        direction_mean, direction_std, direction_folds = direction_topic_holdout_cv(
            X_high, X_low, groups_high, groups_low, seed
        )
        print(
            f"  direction-topic: {direction_mean:.4f} +/- {direction_std:.4f}  "
            f"{[round(s, 3) for s in direction_folds]}"
        )
        result["direction_topic_cv_mean"] = round(direction_mean, 6)
        result["direction_topic_cv_std"] = round(direction_std, 6)
        result["direction_topic_cv_folds"] = direction_folds

    return result


def cross_axis_accuracy(
    X_high: np.ndarray,
    X_low: np.ndarray,
    vec: np.ndarray,
    label: str,
    seed: int,
) -> float:
    acc = projected_cv_accuracy(X_high, X_low, vec, seed)
    print(f"  [{label}]: {acc:.4f}")
    return acc


def projected_cv_accuracy(
    X_high: np.ndarray,
    X_low: np.ndarray,
    vec: np.ndarray,
    seed: int,
) -> float:
    """Return scale-invariant 1-D CV accuracy along a fixed direction.

    Standardisation is fitted inside each training fold. Residual-stream
    projections differ by orders of magnitude across model families; without
    scaling, logistic regression can stop at its constant initial prediction
    and report a spurious 0.50 accuracy.
    """
    unit_vec = vec / (np.linalg.norm(vec) + 1e-12)
    proj_high = X_high @ unit_vec
    proj_low = X_low @ unit_vec
    X_proj = np.concatenate([proj_high, proj_low]).reshape(-1, 1)
    y = np.array([1] * len(proj_high) + [0] * len(proj_low))
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed, C=1.0),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, X_proj, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print(f"[config] processed: {cfg.paths.processed}")

    if args.dry_run:
        print("[dry-run] skipping validation")
        return

    seed = cfg.probing.seed
    data = load_vectors(Path(cfg.paths.processed), subdir=args.vectors_subdir)
    meta = data["meta"]
    print(f"[meta] model={meta['model']}, layer={meta['probe_layer']}, d_model={meta['d_model']}")

    warmth_vec = data["warmth_vec"]
    competence_vec = data["competence_vec"]

    # Load topic groups for holdout CV (groups by topic_idx prevent vocabulary leakage).
    stimuli_path = Path(cfg.paths.stimuli) / "concept_stories.jsonl"
    topic_groups: dict[str, np.ndarray] | None = None
    if stimuli_path.exists():
        topic_groups = load_topic_groups(stimuli_path)
        # Validate alignment: group arrays must match loaded activation matrices.
        for cond in EXPECTED_CONDITIONS:
            n_stories = len(data[cond])
            n_groups = len(topic_groups[cond])
            if n_stories != n_groups:
                raise ValueError(
                    f"Alignment error for condition {cond!r}: "
                    f"X_{cond}.npy has {n_stories} rows but topic_groups has {n_groups} entries. "
                    "Ensure concept_stories.jsonl was not modified after extraction."
                )
        print(f"[topic-groups] loaded from {stimuli_path}")
    else:
        print(f"[topic-groups] WARNING: {stimuli_path} not found; skipping topic-holdout CV")

    warmth_metrics = probe_axis(
        data["high_warmth"], data["low_warmth"], warmth_vec, "warmth", seed,
        groups_high=topic_groups["high_warmth"] if topic_groups else None,
        groups_low=topic_groups["low_warmth"] if topic_groups else None,
    )
    competence_metrics = probe_axis(
        data["high_competence"], data["low_competence"], competence_vec, "competence", seed,
        groups_high=topic_groups["high_competence"] if topic_groups else None,
        groups_low=topic_groups["low_competence"] if topic_groups else None,
    )

    wv = warmth_vec / (np.linalg.norm(warmth_vec) + 1e-12)
    cv = competence_vec / (np.linalg.norm(competence_vec) + 1e-12)
    axis_cosine = float(np.dot(wv, cv))
    print(f"\n[orthogonality] cos(warmth_vec, competence_vec) = {axis_cosine:.6f}  (target: |val| < 0.3)")

    print("\n[cross-axis probe accuracy]")
    cross_w_on_c = cross_axis_accuracy(
        data["high_competence"], data["low_competence"], warmth_vec,
        "warmth_vec on competence stories", seed,
    )
    cross_c_on_w = cross_axis_accuracy(
        data["high_warmth"], data["low_warmth"], competence_vec,
        "competence_vec on warmth stories", seed,
    )

    transfer_w_to_c: tuple[float, float, list[float]] | None = None
    transfer_c_to_w: tuple[float, float, list[float]] | None = None
    if topic_groups is not None:
        transfer_w_to_c = topic_cross_axis_transfer_cv(
            data["high_warmth"], data["low_warmth"],
            data["high_competence"], data["low_competence"],
            topic_groups["high_warmth"], topic_groups["low_warmth"],
            topic_groups["high_competence"], topic_groups["low_competence"],
            seed,
        )
        transfer_c_to_w = topic_cross_axis_transfer_cv(
            data["high_competence"], data["low_competence"],
            data["high_warmth"], data["low_warmth"],
            topic_groups["high_competence"], topic_groups["low_competence"],
            topic_groups["high_warmth"], topic_groups["low_warmth"],
            seed,
        )
        print("\n[cross-axis topic-held-out transfer; no target recalibration]")
        print(
            f"  warmth -> competence: {transfer_w_to_c[0]:.4f} +/- "
            f"{transfer_w_to_c[1]:.4f}  {transfer_w_to_c[2]}"
        )
        print(
            f"  competence -> warmth: {transfer_c_to_w[0]:.4f} +/- "
            f"{transfer_c_to_w[1]:.4f}  {transfer_c_to_w[2]}"
        )

    # Label suffix keeps outputs for different models from clobbering each other.
    label_suffix = f"_{args.label}" if args.label else ""

    table_dir = Path(cfg.paths.results) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / f"probe_metrics{label_suffix}.csv"
    fieldnames = list(warmth_metrics.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(warmth_metrics)
        writer.writerow(competence_metrics)
    print(f"\n[table] {csv_path}")

    log = {
        "meta": meta,
        "warmth": warmth_metrics,
        "competence": competence_metrics,
        "axis_cosine": round(axis_cosine, 6),
        # Backward-compatible aliases: these fixed directions are recalibrated
        # on the target axis inside each CV fold and are not zero-shot transfer.
        "cross_warmth_on_competence_cv": round(cross_w_on_c, 6),
        "cross_competence_on_warmth_cv": round(cross_c_on_w, 6),
        "cross_warmth_on_competence_calibrated_cv": round(cross_w_on_c, 6),
        "cross_competence_on_warmth_calibrated_cv": round(cross_c_on_w, 6),
        "pass_warmth_cv": warmth_metrics["cv_mean"] > 0.8,
        "pass_competence_cv": competence_metrics["cv_mean"] > 0.8,
        "pass_orthogonality": abs(axis_cosine) < 0.3,
        "pass_warmth_topic_cv": warmth_metrics.get("topic_cv_mean", 0.0) > 0.8,
        "pass_competence_topic_cv": competence_metrics.get("topic_cv_mean", 0.0) > 0.8,
    }
    if transfer_w_to_c is not None and transfer_c_to_w is not None:
        log.update({
            "cross_warmth_to_competence_topic_transfer_mean": round(transfer_w_to_c[0], 6),
            "cross_warmth_to_competence_topic_transfer_std": round(transfer_w_to_c[1], 6),
            "cross_warmth_to_competence_topic_transfer_folds": transfer_w_to_c[2],
            "cross_competence_to_warmth_topic_transfer_mean": round(transfer_c_to_w[0], 6),
            "cross_competence_to_warmth_topic_transfer_std": round(transfer_c_to_w[1], 6),
            "cross_competence_to_warmth_topic_transfer_folds": transfer_c_to_w[2],
        })
    log_dir = Path(cfg.paths.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_label = args.label or "default"
    log_path = log_dir / f"validate_probes_{log_label}.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"[log] {log_path}")

    print("\n--- SUMMARY ---")
    print(f"  warmth CV (5-fold)     : {warmth_metrics['cv_mean']:.4f}  {'PASS' if log['pass_warmth_cv'] else 'FAIL'}  (threshold 0.80)")
    print(f"  competence CV (5-fold) : {competence_metrics['cv_mean']:.4f}  {'PASS' if log['pass_competence_cv'] else 'FAIL'}  (threshold 0.80)")
    if "topic_cv_mean" in warmth_metrics:
        print(f"  warmth topic-holdout   : {warmth_metrics['topic_cv_mean']:.4f}  {'PASS' if log['pass_warmth_topic_cv'] else 'FAIL'}  (discriminative; threshold 0.80)")
    if "topic_cv_mean" in competence_metrics:
        print(f"  competence topic-hold  : {competence_metrics['topic_cv_mean']:.4f}  {'PASS' if log['pass_competence_topic_cv'] else 'FAIL'}  (discriminative; threshold 0.80)")
    print(f"  |cos(W,C)|             : {abs(axis_cosine):.4f}  {'PASS' if log['pass_orthogonality'] else 'FAIL'}  (threshold 0.30)")
    print(f"  cross-W->C CV          : {cross_w_on_c:.4f}")
    print(f"  cross-C->W CV          : {cross_c_on_w:.4f}")
    if transfer_w_to_c is not None and transfer_c_to_w is not None:
        print(f"  transfer W->C topic     : {transfer_w_to_c[0]:.4f}")
        print(f"  transfer C->W topic     : {transfer_c_to_w[0]:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate warmth and competence probes.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--vectors-subdir",
        default="concept_vectors",
        help="Subdirectory under cfg.paths.processed where vectors were saved "
             "(default: concept_vectors).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Short label to suffix output filenames (e.g. qwen3_14b). "
             "Prevents overwriting Gemma outputs. If omitted, original filenames are used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
