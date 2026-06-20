from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse


CONDITIONS = (
    "high_warmth",
    "low_warmth",
    "high_competence",
    "low_competence",
)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pooled = np.sqrt((a.var() + b.var()) / 2.0)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def decompose_feature_axes(
    warmth: np.ndarray,
    competence: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split two feature contrasts into shared and axis-specific components.

    For features with the same sign on both axes, the shared component receives
    the smaller absolute effect. Opposite-sign effects remain axis-specific.
    The decomposition is exact: warmth == shared + warmth_specific and likewise
    for competence.
    """
    warmth = np.asarray(warmth, dtype=np.float32)
    competence = np.asarray(competence, dtype=np.float32)
    same_sign = np.sign(warmth) == np.sign(competence)
    shared = np.where(
        same_sign,
        np.sign(warmth) * np.minimum(np.abs(warmth), np.abs(competence)),
        0.0,
    ).astype(np.float32)
    return {
        "warmth": warmth,
        "competence": competence,
        "shared": shared,
        "warmth_specific": (warmth - shared).astype(np.float32),
        "competence_specific": (competence - shared).astype(np.float32),
    }


def smallest_energy_feature_set(
    vector: np.ndarray,
    fraction: float = 0.50,
) -> np.ndarray:
    """Return the smallest descending-magnitude feature set covering energy."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    vector = np.asarray(vector, dtype=np.float64)
    energy = np.square(vector)
    total = float(energy.sum())
    if total == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(energy)[::-1]
    k = int(np.searchsorted(np.cumsum(energy[order]), fraction * total) + 1)
    return order[:k].astype(np.int64)


def load_story_records(path: Path) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {condition: [] for condition in CONDITIONS}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            condition = record["condition"]
            if condition in buckets:
                buckets[condition].append(record)
    for condition, records in buckets.items():
        if not records:
            raise ValueError(f"No records found for condition {condition!r}.")
    return buckets


def condition_slices(counts: dict[str, int]) -> dict[str, slice]:
    result: dict[str, slice] = {}
    start = 0
    for condition in CONDITIONS:
        stop = start + int(counts[condition])
        result[condition] = slice(start, stop)
        start = stop
    return result


def sparse_mean(matrix: sparse.spmatrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel().astype(np.float32)


def bootstrap_mean_ci(
    values: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Paired/group bootstrap confidence interval for a mean."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    grouped = np.array([values[groups == group].mean() for group in unique])
    estimate = float(grouped.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(grouped, size=len(grouped), replace=True)
        draws[i] = sample.mean()
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return estimate, float(low), float(high)


def check_file_size(path: Path, max_bytes: int = 50 * 1024 * 1024) -> None:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"Output file exceeds {max_bytes / 1024**2:.0f} MB guard: "
            f"{path} ({size / 1024**2:.1f} MB)"
        )
