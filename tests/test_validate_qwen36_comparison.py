from pathlib import Path

import numpy as np

from src.validate_qwen36_comparison import CONDITIONS, agreement_records


def _write_model(root: Path, name: str, orders: list[np.ndarray]) -> Path:
    directory = root / name
    directory.mkdir()
    for axis in ("warmth", "competence"):
        np.save(directory / f"{axis}_vec.npy", np.array([1.0]))
    for condition, values, offset in zip(CONDITIONS, orders, (20, -20, 10, -10)):
        np.save(directory / f"X_{condition}.npy", (values + offset).reshape(-1, 1))
    return directory


def test_agreement_separates_overall_and_within_condition(tmp_path: Path) -> None:
    ascending = [np.arange(20, dtype=float) / 100 for _ in CONDITIONS]
    mixed = [values[::-1] if index % 2 else values for index, values in enumerate(ascending)]
    first = _write_model(tmp_path, "first", ascending)
    second = _write_model(tmp_path, "second", mixed)

    records = agreement_records([first, second], ["first", "second"])

    assert len(records) == 2
    assert records[0]["overall_rho"] > 0.8
    assert abs(records[0]["within_condition_rho"]) < 0.1
