import json

import numpy as np
import pandas as pd
import pytest

from src.validate_gemma4_stage import (
    require_targets_absent,
    validate_stage1,
    validate_stage2,
    validate_stage3,
    validate_stage3b,
)


MODEL = "google/gemma-4-12B-it"
LAYERS = 48
D_MODEL = 4
PROBE_LAYER = 31


def write_stage1(root, *, finite=True, vector_nonzero=True):
    root.mkdir()
    meta = {
        "model": MODEL,
        "n_layers": LAYERS,
        "d_model": D_MODEL,
        "probe_layer": PROBE_LAYER,
        "seed": 20260527,
        "input_format": "raw-passage",
    }
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    value = 1.0 if finite else np.nan
    for condition in (
        "high_warmth",
        "low_warmth",
        "high_competence",
        "low_competence",
    ):
        np.save(root / f"X_{condition}.npy", np.full((50, D_MODEL), value))
    vector = np.ones(D_MODEL) if vector_nonzero else np.zeros(D_MODEL)
    np.save(root / "warmth_vec.npy", vector)
    np.save(root / "competence_vec.npy", vector)


def common():
    return dict(
        model=MODEL,
        expected_layers=LAYERS,
        expected_d_model=D_MODEL,
        expected_layer=PROBE_LAYER,
    )


def test_stage1_accepts_complete_finite_outputs(tmp_path):
    vectors = tmp_path / "vectors"
    write_stage1(vectors)
    validate_stage1(vectors, **common())


@pytest.mark.parametrize("finite,nonzero", [(False, True), (True, False)])
def test_stage1_rejects_nonfinite_or_zero_vectors(tmp_path, finite, nonzero):
    vectors = tmp_path / "vectors"
    write_stage1(vectors, finite=finite, vector_nonzero=nonzero)
    with pytest.raises(AssertionError):
        validate_stage1(vectors, **common())


def test_stage2_preserves_negative_scientific_flags(tmp_path):
    table = tmp_path / "probe.csv"
    pd.DataFrame(
        [
            {"axis": "warmth", "cv_mean": 0.55, "cohens_d": 0.1},
            {"axis": "competence", "cv_mean": 0.60, "cohens_d": 0.2},
        ]
    ).to_csv(table, index=False)
    log = tmp_path / "probe.json"
    log.write_text(
        json.dumps(
            {
                "meta": {
                    "model": MODEL,
                    "n_layers": LAYERS,
                    "d_model": D_MODEL,
                    "probe_layer": PROBE_LAYER,
                },
                "pass_warmth_cv": False,
                "pass_competence_cv": False,
                "pass_orthogonality": False,
                "pass_warmth_topic_cv": False,
                "pass_competence_topic_cv": False,
            }
        ),
        encoding="utf-8",
    )
    validate_stage2(table, log, **common())


def test_stage3_requires_complete_layer_rows(tmp_path):
    table = tmp_path / "layers.csv"
    pd.DataFrame(
        {
            "layer": range(LAYERS - 1),
            "frac": np.arange(LAYERS - 1) / (LAYERS - 1),
            "is_probe_layer": [index == PROBE_LAYER for index in range(LAYERS - 1)],
            "warmth_topic_cv": 0.9,
            "comp_topic_cv": 0.9,
            "warmth_cohens_d": 1.0,
            "comp_cohens_d": 1.0,
            "cos_wc": 0.2,
            "mean_resid_norm": 10.0,
        }
    ).to_csv(table, index=False)
    meta = tmp_path / "layers.meta.json"
    meta.write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="expected 48 rows"):
        validate_stage3(table, meta, **common())


def test_stage3_accepts_complete_finite_sweep(tmp_path):
    table = tmp_path / "layers.csv"
    pd.DataFrame(
        {
            "layer": range(LAYERS),
            "frac": np.arange(LAYERS) / (LAYERS - 1),
            "is_probe_layer": [index == PROBE_LAYER for index in range(LAYERS)],
            "warmth_topic_cv": 0.9,
            "comp_topic_cv": 0.9,
            "warmth_cohens_d": 1.0,
            "comp_cohens_d": 1.0,
            "cos_wc": 0.2,
            "mean_resid_norm": 10.0,
        }
    ).to_csv(table, index=False)
    meta = tmp_path / "layers.meta.json"
    meta.write_text(
        json.dumps(
            {
                "model": MODEL,
                "n_layers": LAYERS,
                "d_model": D_MODEL,
                "probe_layer": PROBE_LAYER,
                "seed": 20260527,
                "n_stories": 200,
                "input_format": "raw-passage",
            }
        ),
        encoding="utf-8",
    )
    validate_stage3(table, meta, **common())


def test_stage3b_accepts_enhanced_artifacts(tmp_path):
    table = tmp_path / "layers.csv"
    frame = pd.DataFrame(
        {
            "layer": range(LAYERS),
            "frac": np.arange(LAYERS) / (LAYERS - 1),
            "is_probe_layer": [index == PROBE_LAYER for index in range(LAYERS)],
            "warmth_topic_cv": 0.9,
            "comp_topic_cv": 0.9,
            "warmth_cohens_d": 1.0,
            "comp_cohens_d": 1.0,
            "cos_wc": 0.2,
            "mean_resid_norm": 10.0,
            "warmth_direction_topic_cv": 0.9,
            "warmth_direction_topic_cv_std": 0.02,
            "comp_direction_topic_cv": 0.9,
            "comp_direction_topic_cv_std": 0.02,
            "warmth_to_comp_topic_transfer": 0.8,
            "warmth_to_comp_topic_transfer_std": 0.03,
            "comp_to_warmth_topic_transfer": 0.8,
            "comp_to_warmth_topic_transfer_std": 0.03,
            "warmth_cohens_d_ci_low": 0.8,
            "warmth_cohens_d_ci_high": 1.2,
            "comp_cohens_d_ci_low": 0.8,
            "comp_cohens_d_ci_high": 1.2,
            "cos_wc_ci_low": 0.1,
            "cos_wc_ci_high": 0.3,
        }
    )
    frame.to_csv(table, index=False)
    meta = tmp_path / "layers.meta.json"
    meta.write_text(
        json.dumps(
            {
                "model": MODEL,
                "n_layers": LAYERS,
                "d_model": D_MODEL,
                "probe_layer": PROBE_LAYER,
                "seed": 20260527,
                "n_stories": 200,
                "input_format": "raw-passage",
                "analysis_profile": "stage3b",
                "n_bootstrap": 1000,
                "git_commit": "abc123",
                "stimuli_sha256": "deadbeef",
            }
        ),
        encoding="utf-8",
    )
    fold_metrics = {
        "warmth_direction_topic_cv": [0.9] * 5,
        "comp_direction_topic_cv": [0.9] * 5,
        "warmth_to_comp_topic_transfer": [0.8] * 5,
        "comp_to_warmth_topic_transfer": [0.8] * 5,
    }
    probability = [0.0] * LAYERS
    probability[10] = 1.0
    peak = {"layer_probabilities": probability}
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "analysis_profile": "stage3b",
                "n_layers": LAYERS,
                "folds_by_layer": {str(i): fold_metrics for i in range(LAYERS)},
                "bootstrap": {
                    "n_bootstrap": 1000,
                    "n_topics": 50,
                    "peaks": {
                        "warmth_cohens_d": peak,
                        "comp_cohens_d": peak,
                        "cos_wc": peak,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    validate_stage3b(table, meta, audit, **common())


def test_no_clobber_rejects_existing_target(tmp_path):
    target = tmp_path / "existing"
    target.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        require_targets_absent((target, tmp_path / "missing"))
