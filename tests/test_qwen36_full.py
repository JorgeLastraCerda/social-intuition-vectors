from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.qwen36_pipeline import CONDITIONS, load_full_records, stage_paths
from src.utils.config import load_config
from src.validate_qwen36_stage import (
    cross_stage_audit,
    validate_stage1,
    validate_stage2,
    validate_stage3,
)


ROOT = Path(__file__).resolve().parents[1]
FULL_SUBMITTER = ROOT / "jobs/sge/submit_qwen36_full.sh"
FOLLOWUP_SUBMITTER = ROOT / "jobs/sge/submit_qwen36_followup.sh"
RUNNER = ROOT / "jobs/sge/qwen36_stage.sh"


def test_production_configs_are_pinned_and_architecture_specific() -> None:
    dense = load_config(ROOT / "config/qwen36_27b.yaml")
    moe = load_config(ROOT / "config/qwen36_35b_a3b.yaml")
    assert dense.model.name == "Qwen/Qwen3.6-27B"
    assert dense.model.revision == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    assert (dense.native_hf.expected_layers, dense.native_hf.expected_d_model) == (64, 5120)
    assert moe.model.name == "Qwen/Qwen3.6-35B-A3B"
    assert moe.model.revision == "995ad96eacd98c81ed38be0c5b274b04031597b0"
    assert (moe.native_hf.expected_layers, moe.native_hf.expected_d_model) == (40, 2048)
    for cfg in (dense, moe):
        assert cfg.model.backend == "huggingface-native"
        assert cfg.probing.probe_layer_frac == 0.66
        assert cfg.probing.seed == 20260527


def test_full_corpus_contract_and_hash() -> None:
    buckets, digest = load_full_records(ROOT / "data/stimuli/concept_stories.jsonl")
    assert {key: len(value) for key, value in buckets.items()} == {
        condition: 50 for condition in CONDITIONS
    }
    assert len(digest) == 64


def _temporary_config(tmp_path: Path):
    cfg = load_config(ROOT / "config/qwen36_35b_a3b.yaml")
    return replace(
        cfg,
        native_hf=replace(cfg.native_hf, expected_d_model=4, expected_layers=4),
        paths=replace(
            cfg.paths,
            processed=tmp_path / "processed",
            results=tmp_path / "results",
            logs=tmp_path / "results/logs",
        ),
    )


def _runtime(cfg) -> dict:
    return {
        "backend": "huggingface-native",
        "transformer_lens_version": "not-installed",
        "transformer_lens_imported": False,
        "model_revision_requested": cfg.model.revision,
        "model_revision_resolved": cfg.model.revision,
        "parameter_devices": ["cuda:0"],
        "vision_forward_calls": 0,
        "hook_hidden_max_diff": 0.0,
        "passive_hook_max_logit_diff": 0.0,
        "cuda_device_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "peak_reserved_vram_fraction": 0.75,
    }


def _write_outputs(cfg) -> None:
    paths = stage_paths(cfg)
    paths.vectors_dir.mkdir(parents=True)
    paths.probe_table.parent.mkdir(parents=True)
    paths.technical_logs[1].parent.mkdir(parents=True)
    probe_layer = 2
    meta = {
        "model": cfg.model.name,
        "revision": cfg.model.revision,
        "n_layers": 4,
        "d_model": 4,
        "probe_layer": probe_layer,
        "probe_layer_frac": 0.66,
        "start_token": 50,
        "seed": 20260527,
        "n_stories": 200,
        "input_format": "raw-passage-explicit-bos",
        "stimuli_sha256": "a" * 64,
        "runtime": _runtime(cfg),
    }
    for condition in CONDITIONS:
        np.save(paths.vectors_dir / f"X_{condition}.npy", np.ones((50, 4)))
    np.save(paths.vectors_dir / "warmth_vec.npy", np.ones(4))
    np.save(paths.vectors_dir / "competence_vec.npy", np.ones(4))
    (paths.vectors_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    paths.technical_logs[1].write_text(
        json.dumps({"status": "pass", "stage": 1, "label": cfg.native_hf.label, **meta}),
        encoding="utf-8",
    )

    pd.DataFrame(
        [{"axis": "warmth", "cohens_d": 1.0}, {"axis": "competence", "cohens_d": 2.0}]
    ).to_csv(paths.probe_table, index=False)
    probe_log = {
        "meta": meta,
        "axis_cosine": 0.25,
        "scientific_flags_are_non_gating": True,
        "pass_warmth_cv": False,
        "pass_competence_cv": False,
        "pass_orthogonality": True,
        "pass_warmth_topic_cv": False,
        "pass_competence_topic_cv": False,
    }
    paths.probe_log.write_text(json.dumps(probe_log), encoding="utf-8")
    paths.technical_logs[2].write_text(
        json.dumps(
            {
                "status": "pass",
                "stage": 2,
                "label": cfg.native_hf.label,
                "model": cfg.model.name,
                "revision": cfg.model.revision,
                "seed": 20260527,
                "stimuli_sha256": "a" * 64,
                "backend": "numpy-scikit-learn-cpu",
            }
        ),
        encoding="utf-8",
    )

    rows = pd.DataFrame(
        {
            "layer": range(4),
            "frac": np.arange(4) / 3,
            "is_probe_layer": [False, False, True, False],
            "warmth_topic_cv": 0.7,
            "warmth_topic_cv_std": 0.1,
            "comp_topic_cv": 0.8,
            "comp_topic_cv_std": 0.1,
            "warmth_cohens_d": [0.5, 0.5, 1.0, 0.5],
            "comp_cohens_d": [0.5, 0.5, 2.0, 0.5],
            "cos_wc": [0.1, 0.1, 0.25, 0.1],
            "mean_resid_norm": 10.0,
        }
    )
    rows.to_csv(paths.sweep_table, index=False)
    paths.sweep_meta.write_text(json.dumps(meta), encoding="utf-8")
    paths.technical_logs[3].write_text(
        json.dumps({"status": "pass", "stage": 3, "label": cfg.native_hf.label, **meta}),
        encoding="utf-8",
    )


def test_stage_validators_and_cross_stage_audit_accept_negative_scientific_flags(
    tmp_path,
) -> None:
    cfg = _temporary_config(tmp_path)
    _write_outputs(cfg)
    validate_stage1(cfg)
    validate_stage2(cfg)
    validate_stage3(cfg)
    audit = cross_stage_audit(cfg)
    assert audit["pass"] is True
    assert audit["non_gating_reproducibility_audit"] is True


def test_submitters_are_independent_and_dry_runnable() -> None:
    env = os.environ.copy()
    env["RUN_ID"] = "20990101T000000Z"
    full = subprocess.run(
        ["bash", str(FULL_SUBMITTER), "--dry-run"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    followup = subprocess.run(
        ["bash", str(FOLLOWUP_SUBMITTER), "config/qwen36_27b.yaml", "--dry-run"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "hold_jid=none" in full.stdout
    assert "independent stages=2,3" in followup.stdout
    assert "stage2 queue=scc gpu=0" in followup.stdout
    for script in (FULL_SUBMITTER, FOLLOWUP_SUBMITTER):
        text = script.read_text(encoding="utf-8")
        assert "-hold_jid" not in text
        assert "PREDECESSOR_SENTINEL" not in text
    runner = RUNNER.read_text(encoding="utf-8")
    assert "git pull" not in runner
    assert "sync_outputs.sh" not in runner
