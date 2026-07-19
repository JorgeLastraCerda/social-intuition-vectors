from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.qwen36_hiring import select_names
from src.utils.config import PathConfig, load_config
from src.validate_qwen36_hiring import validate

ROOT = Path(__file__).resolve().parents[1]


def test_name_selection_matches_seeded_legacy_panel() -> None:
    ratings = pd.DataFrame({"name": [f"name_{index:03d}" for index in range(282)]})
    first = select_names(ratings, n_names=60, seed=20260527)
    second = select_names(ratings, n_names=60, seed=20260527)
    assert first.equals(second)
    assert len(first) == 60
    assert first["name"].nunique() == 60
    assert len(select_names(ratings, n_names=0, seed=20260527)) == 282


def test_audit_validator_accepts_complete_native_hf_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = load_config(ROOT / "config/qwen36_27b.yaml")
    cfg = replace(
        cfg,
        paths=PathConfig(
            papers=tmp_path / "paper",
            raw_data=tmp_path / "raw",
            stimuli=tmp_path / "stimuli",
            processed=tmp_path / "processed",
            results=tmp_path / "results",
            logs=tmp_path / "results" / "logs",
        ),
    )
    monkeypatch.setattr("src.validate_qwen36_hiring.load_config", lambda _path: cfg)
    tables = cfg.paths.results / "tables"
    tables.mkdir(parents=True)
    cfg.paths.logs.mkdir(parents=True)
    label = "qwen36_27b"
    pd.DataFrame(
        {
            "name": [f"name_{index:03d}" for index in range(282)],
            "human_warm": np.linspace(0, 1, 282),
            "human_competent": np.linspace(1, 2, 282),
            "model_warmth": np.linspace(2, 3, 282),
            "model_competence": np.linspace(3, 4, 282),
            "callback_margin": np.linspace(-1, 1, 282),
        }
    ).to_csv(tables / f"hiring_audit_{label}.csv", index=False)
    (cfg.paths.logs / f"hiring_probe_vs_human_{label}.json").write_text(
        json.dumps(
            {
                "model": cfg.model.name,
                "revision": cfg.model.revision,
                "transformer_lens_imported": False,
                "runtime": {"model_revision_resolved": cfg.model.revision},
                "correlations": [{} for _ in range(6)],
            }
        ),
        encoding="utf-8",
    )
    assert validate("unused", "audit", label)["rows"] == 282


def test_qwen_hiring_source_is_resumable_and_native_hf() -> None:
    source = (ROOT / "src/qwen36_hiring.py").read_text(encoding="utf-8")
    assert "CheckpointStore" in source
    assert "checkpoint.consolidate" in source
    assert "TransformerLens was imported" in source
    assert "register_forward_hook" in source
    runner = (ROOT / "jobs/ccu/run_qwen36_hiring.sh").read_text(encoding="utf-8")
    assert "audit|neutral|local|broad|denoised_local" in runner
    assert "--checkpoint-origin-commit" in runner
    assert "Expected H100" in runner
    subprocess.run(
        ["bash", "-n", "jobs/ccu/run_qwen36_hiring.sh"], cwd=ROOT, check=True
    )
