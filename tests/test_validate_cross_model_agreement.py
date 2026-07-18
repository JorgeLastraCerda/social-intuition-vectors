from pathlib import Path

import numpy as np

from src.validate_cross_model_agreement import compute_agreement_records


def _write_model(root: Path, name: str, within_orders: list[np.ndarray]) -> Path:
    model_dir = root / name
    model_dir.mkdir()
    np.save(model_dir / "warmth_vec.npy", np.array([1.0]))
    np.save(model_dir / "competence_vec.npy", np.array([1.0]))
    offsets = [20.0, -20.0, 10.0, -10.0]
    for condition, values, offset in zip(
        ("high_warmth", "low_warmth", "high_competence", "low_competence"),
        within_orders,
        offsets,
    ):
        np.save(model_dir / f"X_{condition}.npy", (values + offset).reshape(-1, 1))
    return model_dir


def test_overall_agreement_can_exceed_within_condition_agreement(tmp_path: Path):
    ascending = [np.arange(20, dtype=float) / 100 for _ in range(4)]
    mixed = [values[::-1] if idx % 2 else values for idx, values in enumerate(ascending)]
    first = _write_model(tmp_path, "first", ascending)
    second = _write_model(tmp_path, "second", mixed)

    records = compute_agreement_records([first, second], ["first", "second"])

    assert len(records) == 2
    assert records[0]["overall_rho"] > 0.8
    assert abs(records[0]["within_condition_rho"]) < 0.1
