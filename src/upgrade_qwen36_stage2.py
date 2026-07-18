"""Add strict scientific validation fields to existing Qwen3.6 Stage 2 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from src.qwen36_pipeline import (
    _sha256,
    _write_csv,
    _write_json,
    compute_stage2_outputs,
    stage_paths,
)
from src.utils.config import load_config


ADDED_AXIS_FIELDS = {
    "direction_topic_cv_mean",
    "direction_topic_cv_std",
    "direction_topic_cv_folds",
}
ADDED_LOG_FIELDS = {
    "cross_warmth_on_competence_calibrated_cv",
    "cross_competence_on_warmth_calibrated_cv",
    "cross_warmth_to_competence_topic_transfer_mean",
    "cross_warmth_to_competence_topic_transfer_std",
    "cross_warmth_to_competence_topic_transfer_folds",
    "cross_competence_to_warmth_topic_transfer_mean",
    "cross_competence_to_warmth_topic_transfer_std",
    "cross_competence_to_warmth_topic_transfer_folds",
}


def _equal(old: Any, new: Any, path: str = "root") -> None:
    """Require every pre-existing value to survive the additive upgrade."""
    if isinstance(old, dict):
        if not isinstance(new, dict):
            raise AssertionError(f"{path}: expected mapping in regenerated output.")
        for key, value in old.items():
            if key not in new:
                raise AssertionError(f"{path}.{key}: missing from regenerated output.")
            _equal(value, new[key], f"{path}.{key}")
        return
    if isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            raise AssertionError(f"{path}: list shape changed.")
        for index, (old_value, new_value) in enumerate(zip(old, new, strict=True)):
            _equal(old_value, new_value, f"{path}[{index}]")
        return
    if isinstance(old, (int, float)) and not isinstance(old, bool):
        if not isinstance(new, (int, float)) or not math.isclose(
            float(old), float(new), rel_tol=0.0, abs_tol=1e-5
        ):
            raise AssertionError(f"{path}: changed from {old!r} to {new!r}.")
        return
    if old != new:
        raise AssertionError(f"{path}: changed from {old!r} to {new!r}.")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _merge_table(
    old_rows: list[dict[str, str]], new_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(old_rows) != len(new_rows):
        raise AssertionError("Stage 2 row count changed during regeneration.")
    merged_rows: list[dict[str, Any]] = []
    for old, new in zip(old_rows, new_rows, strict=True):
        for key, old_value in old.items():
            if key in ADDED_AXIS_FIELDS:
                continue
            if key not in new:
                raise AssertionError(f"Stage 2 column {key!r} disappeared.")
            new_value = new[key]
            if isinstance(new_value, list):
                parsed_old = json.loads(old_value)
                _equal(parsed_old, new_value, f"table.{old['axis']}.{key}")
            elif key == "axis":
                _equal(old_value, new_value, f"table.{key}")
            else:
                _equal(float(old_value), float(new_value), f"table.{old['axis']}.{key}")
        legacy = {key: value for key, value in old.items() if key not in ADDED_AXIS_FIELDS}
        merged_rows.append({**new, **legacy})
    return merged_rows


def _additive_merge(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Keep canonical old values and add only fields absent from the old payload."""
    _equal(old, new)
    merged = dict(new)
    for key, value in old.items():
        if isinstance(value, dict) and isinstance(new.get(key), dict):
            merged[key] = _additive_merge(value, new[key])
        else:
            merged[key] = value
    return merged


def _legacy_log(payload: dict[str, Any]) -> dict[str, Any]:
    legacy = {key: value for key, value in payload.items() if key not in ADDED_LOG_FIELDS}
    for axis in ("warmth", "competence"):
        if isinstance(legacy.get(axis), dict):
            legacy[axis] = {
                key: value
                for key, value in legacy[axis].items()
                if key not in ADDED_AXIS_FIELDS
            }
    return legacy


def upgrade(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    paths = stage_paths(cfg)
    old_rows = _read_csv(paths.probe_table)
    old_log = json.loads(paths.probe_log.read_text(encoding="utf-8"))
    before = {
        "probe_table_sha256": _sha256(paths.probe_table),
        "probe_log_sha256": _sha256(paths.probe_log),
    }
    audit_path = Path(cfg.paths.logs) / f"{cfg.native_hf.label}_stage2_strict_audit.json"
    prior_audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.exists()
        else None
    )
    started = time.time()
    new_rows, new_log, stimuli_sha256 = compute_stage2_outputs(cfg, paths)
    merged_rows = _merge_table(old_rows, new_rows)
    merged_log = _additive_merge(_legacy_log(old_log), new_log)
    _write_csv(paths.probe_table, merged_rows)
    _write_json(paths.probe_log, merged_log)
    audit = {
        "status": "pass",
        "upgrade": "qwen36-stage2-strict-scientific-validation-v1",
        "model": cfg.model.name,
        "revision": cfg.model.revision,
        "label": cfg.native_hf.label,
        "seed": cfg.probing.seed,
        "stimuli_sha256": stimuli_sha256,
        "backend": "numpy-scikit-learn-cpu",
        "prior_values_preserved": True,
        "before": prior_audit.get("before", before) if prior_audit else before,
        "upgrade_input": before,
        "after": {
            "probe_table_sha256": _sha256(paths.probe_table),
            "probe_log_sha256": _sha256(paths.probe_log),
        },
        "direction_topic_cv": {
            axis: {
                "mean": new_log[axis]["direction_topic_cv_mean"],
                "std": new_log[axis]["direction_topic_cv_std"],
                "folds": new_log[axis]["direction_topic_cv_folds"],
            }
            for axis in ("warmth", "competence")
        },
        "strict_cross_axis_topic_transfer": {
            "warmth_to_competence": {
                "mean": new_log["cross_warmth_to_competence_topic_transfer_mean"],
                "std": new_log["cross_warmth_to_competence_topic_transfer_std"],
                "folds": new_log["cross_warmth_to_competence_topic_transfer_folds"],
            },
            "competence_to_warmth": {
                "mean": new_log["cross_competence_to_warmth_topic_transfer_mean"],
                "std": new_log["cross_competence_to_warmth_topic_transfer_std"],
                "folds": new_log["cross_competence_to_warmth_topic_transfer_folds"],
            },
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_json(audit_path, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(upgrade(args.config), indent=2))


if __name__ == "__main__":
    main()
