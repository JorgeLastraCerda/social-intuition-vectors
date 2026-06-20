from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

from src.utils.config import load_config
from src.utils.plotting import save_figure


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

    return result


def cross_axis_accuracy(
    X_high: np.ndarray,
    X_low: np.ndarray,
    vec: np.ndarray,
    label: str,
    seed: int,
) -> float:
    unit_vec = vec / (np.linalg.norm(vec) + 1e-12)
    proj_high = X_high @ unit_vec
    proj_low = X_low @ unit_vec
    X_proj = np.concatenate([proj_high, proj_low]).reshape(-1, 1)
    y = np.array([1] * len(proj_high) + [0] * len(proj_low))
    lr = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(lr, X_proj, y, cv=cv, scoring="accuracy")
    acc = float(scores.mean())
    print(f"  [{label}]: {acc:.4f}  (expect ~0.50)")
    return acc


def plot_separation(
    proj_high: np.ndarray,
    proj_low: np.ndarray,
    label: str,
    fig_path: Path,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(proj_high, bins=20, alpha=0.6, label="high", color="steelblue")
    plt.hist(proj_low, bins=20, alpha=0.6, label="low", color="tomato")
    plt.xlabel(f"{label} projection")
    plt.ylabel("count")
    plt.title(f"{label} probe separation")
    plt.legend()
    save_figure(fig_path)
    plt.close()


def plot_orthogonality(orth: dict, fig_path: Path) -> None:
    labels = ["warmth→warmth", "comp→comp", "warmth→comp", "comp→warmth"]
    accs = [
        orth["warmth_cv"],
        orth["competence_cv"],
        orth["cross_warmth_on_competence"],
        orth["cross_competence_on_warmth"],
    ]
    colors = ["steelblue", "steelblue", "tomato", "tomato"]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, accs, color=colors)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    plt.axhline(0.8, color="green", linestyle="--", linewidth=1, label="threshold (0.8)")
    plt.ylim(0, 1.05)
    plt.ylabel("5-fold CV accuracy")
    plt.title("Axis orthogonality")
    plt.xticks(rotation=15, ha="right")
    plt.legend(fontsize=8)
    save_figure(fig_path)
    plt.close()


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

    # Label suffix keeps outputs for different models from clobbering each other.
    label_suffix = f"_{args.label}" if args.label else ""
    fig_dir = Path(cfg.paths.results) / "figures"
    if args.label:
        fig_dir = fig_dir / args.label
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_separation(data["high_warmth"] @ wv, data["low_warmth"] @ wv, "warmth", fig_dir / "probe_separation_warmth.png")
    plot_separation(data["high_competence"] @ cv, data["low_competence"] @ cv, "competence", fig_dir / "probe_separation_competence.png")

    orth = {
        "warmth_cv": warmth_metrics["cv_mean"],
        "competence_cv": competence_metrics["cv_mean"],
        "cross_warmth_on_competence": cross_w_on_c,
        "cross_competence_on_warmth": cross_c_on_w,
    }
    plot_orthogonality(orth, fig_dir / "axis_orthogonality.png")

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
        "timestamp": int(time.time()),
        "meta": meta,
        "warmth": warmth_metrics,
        "competence": competence_metrics,
        "axis_cosine": round(axis_cosine, 6),
        "cross_warmth_on_competence_cv": round(cross_w_on_c, 6),
        "cross_competence_on_warmth_cv": round(cross_c_on_w, 6),
        "pass_warmth_cv": warmth_metrics["cv_mean"] > 0.8,
        "pass_competence_cv": competence_metrics["cv_mean"] > 0.8,
        "pass_orthogonality": abs(axis_cosine) < 0.3,
        "pass_warmth_topic_cv": warmth_metrics.get("topic_cv_mean", 0.0) > 0.8,
        "pass_competence_topic_cv": competence_metrics.get("topic_cv_mean", 0.0) > 0.8,
    }
    log_dir = Path(cfg.paths.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"validate_probes_{int(time.time())}.json"
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
    print(f"  cross-W->C CV          : {cross_w_on_c:.4f}  (expect ~0.50)")
    print(f"  cross-C->W CV          : {cross_c_on_w:.4f}  (expect ~0.50)")


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
