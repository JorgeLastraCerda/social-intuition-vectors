from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.qwen36_smoke import (
    CONDITIONS,
    encode_raw_passage,
    mean_pool_after_token,
    select_topic_records,
    smoke_paths,
)
from src.utils.config import load_config
from src.validate_qwen36_smoke import validate


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "jobs/sge/submit_qwen36_smoke.sh"
RUNNER = ROOT / "jobs/sge/qwen36_smoke.sh"
FINALIZER = ROOT / "jobs/sge/qwen36_smoke_finalize.sh"


class FakeTokenizer:
    def __call__(self, _text, *, add_special_tokens, return_tensors):
        assert add_special_tokens is False
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[11, 12, 13]])}


def test_qwen_config_is_isolated_and_pinned() -> None:
    baseline = load_config(ROOT / "config/config.yaml")
    qwen = load_config(ROOT / "config/qwen36_smoke.yaml")
    assert baseline.model.name == "google/gemma-3-12b-it"
    assert qwen.model.name == "Qwen/Qwen3.6-27B"
    assert qwen.model.revision == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    assert qwen.model.backend == "huggingface-native"
    assert qwen.smoke.expected_layers == 64
    assert qwen.smoke.expected_d_model == 5120


def test_topic_selection_is_balanced_and_deterministic() -> None:
    buckets, topics = select_topic_records(
        ROOT / "data/stimuli/concept_stories.jsonl",
        n_topics=10,
        seed=20260527,
    )
    assert topics == [1, 3, 8, 18, 26, 37, 38, 78, 85, 86]
    assert set(buckets) == set(CONDITIONS)
    for condition in CONDITIONS:
        assert [row["topic_idx"] for row in buckets[condition]] == topics


def test_raw_encoder_adds_exactly_one_bos() -> None:
    encoded = encode_raw_passage(FakeTokenizer(), "text", bos_token_id=99)
    assert encoded["input_ids"].tolist() == [[99, 11, 12, 13]]
    assert encoded["attention_mask"].tolist() == [[1, 1, 1, 1]]


def test_mean_pool_rejects_short_sequences() -> None:
    with pytest.raises(ValueError, match="does not exceed"):
        mean_pool_after_token(torch.ones(1, 5, 3), start_token=5)


def write_valid_smoke_outputs(cfg) -> None:
    paths = smoke_paths(cfg)
    paths.vectors_dir.mkdir(parents=True)
    paths.probe_table.parent.mkdir(parents=True)
    paths.probe_log.parent.mkdir(parents=True)
    d_model = cfg.smoke.expected_d_model
    n_topics = cfg.smoke.n_topics
    probe_layer = 42
    selected_topics = list(range(n_topics))
    runtime = {
        "backend": "huggingface-native",
        "transformer_lens_imported": False,
        "transformer_lens_version": "not-installed",
        "parameter_devices": ["cuda:0"],
        "peak_reserved_vram_fraction": 0.7,
    }
    technical = {
        "status": "pass",
        "model": cfg.model.name,
        "revision": cfg.model.revision,
        "seed": cfg.probing.seed,
        "n_layers": cfg.smoke.expected_layers,
        "d_model": d_model,
        "probe_layer": probe_layer,
        "vision_forward_calls": 0,
        "n_stories": n_topics * 4,
        "hook_hidden_max_diff": 0.0,
        "passive_hook_max_logit_diff": 0.0,
        "token_length_min": 60,
        "runtime": runtime,
    }
    paths.technical_log.write_text(json.dumps(technical), encoding="utf-8")
    stage1_meta = {
        "model": cfg.model.name,
        "revision": cfg.model.revision,
        "probe_layer": probe_layer,
        "n_layers": cfg.smoke.expected_layers,
        "d_model": d_model,
        "seed": cfg.probing.seed,
        "input_format": "raw-passage-explicit-bos",
        "smoke": True,
        "selected_topics": selected_topics,
    }
    (paths.vectors_dir / "meta.json").write_text(json.dumps(stage1_meta), encoding="utf-8")
    for condition in CONDITIONS:
        np.save(paths.vectors_dir / f"X_{condition}.npy", np.ones((n_topics, d_model)))
    np.save(paths.vectors_dir / "warmth_vec.npy", np.ones(d_model))
    np.save(paths.vectors_dir / "competence_vec.npy", np.ones(d_model))

    pd.DataFrame(
        [
            {"axis": "warmth", "cohens_d": 1.0, "cv_mean": 0.6},
            {"axis": "competence", "cohens_d": 2.0, "cv_mean": 0.7},
        ]
    ).to_csv(paths.probe_table, index=False)
    paths.probe_log.write_text(
        json.dumps(
            {
                "axis_cosine": 0.25,
                "scientific_flags_are_non_gating": True,
                "pass_warmth_cv": False,
                "pass_competence_cv": False,
                "pass_orthogonality": True,
                "pass_warmth_topic_cv": False,
                "pass_competence_topic_cv": False,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "layer": range(64),
            "frac": np.arange(64) / 63,
            "is_probe_layer": [index == probe_layer for index in range(64)],
            "warmth_topic_cv": 0.6,
            "warmth_topic_cv_std": 0.1,
            "comp_topic_cv": 0.7,
            "comp_topic_cv_std": 0.1,
            "warmth_cohens_d": [1.0 if i == probe_layer else 0.5 for i in range(64)],
            "comp_cohens_d": [2.0 if i == probe_layer else 0.5 for i in range(64)],
            "cos_wc": [0.25 if i == probe_layer else 0.1 for i in range(64)],
            "mean_resid_norm": 10.0,
        }
    ).to_csv(paths.sweep_table, index=False)
    paths.sweep_meta.write_text(
        json.dumps(
            {
                "model": cfg.model.name,
                "revision": cfg.model.revision,
                "n_layers": 64,
                "d_model": d_model,
                "probe_layer": probe_layer,
                "seed": cfg.probing.seed,
                "n_stories": n_topics * 4,
                "smoke": True,
            }
        ),
        encoding="utf-8",
    )


def test_validator_accepts_technical_pass_with_negative_scientific_flags(
    tmp_path, monkeypatch
) -> None:
    cfg = load_config(ROOT / "config/qwen36_smoke.yaml")
    cfg = replace(
        cfg,
        paths=replace(
            cfg.paths,
            processed=tmp_path / "processed",
            results=tmp_path / "results",
            logs=tmp_path / "results/logs",
        ),
    )
    write_valid_smoke_outputs(cfg)
    monkeypatch.setattr("src.validate_qwen36_smoke.load_config", lambda _path: cfg)
    validate("unused.yaml")


def test_submitter_dry_run_and_runner_topology() -> None:
    env = os.environ.copy()
    env["RUN_ID"] = "20990101T000000Z"
    result = subprocess.run(
        ["bash", str(SUBMITTER), "--dry-run"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "availability policy=submit-even-when-zero" in result.stdout
    assert "gpu=1,rtx_6000=1" in result.stdout
    runner = RUNNER.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    assert "git pull" not in runner
    assert "sync_outputs.sh" not in runner
    assert "src.qwen36_smoke" in runner
    assert finalizer.count('bash jobs/sync_outputs.sh "$REPO_PATH"') == 1
