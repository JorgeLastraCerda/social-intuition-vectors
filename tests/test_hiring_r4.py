import numpy as np
import pandas as pd

from src.hiring_r4 import (
    group_statistics,
    load_and_join,
    margin_diagnostic,
    name_level_statistics,
)


def test_r4_join_requires_first_name_and_matching_study(tmp_path):
    audit = pd.DataFrame(
        {
            "name": ["Aisha Smith", "Aisha Jones", "Beth Lee"],
            "study": ["s1", "s2", "s1"],
            "callback_margin": [1.0, 2.0, 3.0],
            "model_warmth": [0.1, 0.2, 0.3],
            "model_competence": [0.4, 0.5, 0.6],
        }
    )
    human = pd.DataFrame(
        {
            "name": ["aisha", "aisha", "beth"],
            "study": ["s1", "s2", "s9"],
            "race": ["Black", "Black", "White"],
            "gender": ["Female", "Female", "Female"],
            "callback": [0.1, 0.2, 0.3],
        }
    )
    audit_path, human_path = tmp_path / "audit.csv", tmp_path / "human.csv"
    audit.to_csv(audit_path, index=False)
    human.to_csv(human_path, index=False)
    matched = load_and_join(audit_path, human_path)
    assert matched["name"].tolist() == ["Aisha Smith", "Aisha Jones"]


def test_r4_statistics_schema_and_quantisation_warning():
    n = 24
    values = np.arange(n) / 8.0
    matched = pd.DataFrame(
        {
            "name": [f"name-{i}" for i in range(n)],
            "race": ["Black", "White"] * (n // 2),
            "gender": ["Female", "Male"] * (n // 2),
            "callback_margin": values,
            "human_callback": np.linspace(0.1, 0.9, n),
            "model_warmth": np.linspace(-1, 1, n),
            "model_competence": np.linspace(1, -0.5, n),
        }
    )
    stats, diagnostic = name_level_statistics(matched, "test")
    assert stats["predictor"].tolist() == [
        "human_callback",
        "model_warmth",
        "model_competence",
    ]
    assert set(group_statistics(matched, "test")["model"]) == {"test"}
    assert diagnostic["fraction_on_0.125_grid"] == 1.0
    narrow = margin_diagnostic(pd.Series([0.0, 0.125] * 12))
    assert narrow["quantisation_warning"] is True
