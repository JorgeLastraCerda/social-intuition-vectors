from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import pytest

from src.steering_checkpoint import CheckpointStore

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_resume_consolidates_without_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    fingerprint = {"model": "example", "seed": 7}
    first = CheckpointStore(root, fingerprint, resume=False)
    first.write(0, {"axis": "warmth"}, [{"row": 1}, {"row": 2}])
    resumed = CheckpointStore(root, fingerprint, resume=True)
    assert resumed.read(0, {"axis": "warmth"}) == [{"row": 1}, {"row": 2}]
    resumed.write(1, {"axis": "competence"}, [{"row": 3}])
    assert resumed.consolidate(2) == [{"row": 1}, {"row": 2}, {"row": 3}]


def test_resumed_and_uninterrupted_consolidations_have_same_checksum(
    tmp_path: Path,
) -> None:
    fingerprint = {"model": "example", "seed": 7}
    rows = [[{"row": 1}], [{"row": 2}], [{"row": 3}]]
    uninterrupted = CheckpointStore(
        tmp_path / "uninterrupted", fingerprint, resume=False
    )
    for sequence, unit in enumerate(rows):
        uninterrupted.write(sequence, {"unit": sequence}, unit)
    interrupted = CheckpointStore(tmp_path / "resumed", fingerprint, resume=False)
    interrupted.write(0, {"unit": 0}, rows[0])
    resumed = CheckpointStore(tmp_path / "resumed", fingerprint, resume=True)
    for sequence, unit in enumerate(rows[1:], start=1):
        resumed.write(sequence, {"unit": sequence}, unit)

    def checksum(payload: list[dict]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    assert checksum(uninterrupted.consolidate(3)) == checksum(resumed.consolidate(3))


def test_checkpoint_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    CheckpointStore(root, {"seed": 1}, resume=False)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        CheckpointStore(root, {"seed": 2}, resume=True)


def test_checkpoint_does_not_treat_atomic_temp_as_complete(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    store = CheckpointStore(root, {"seed": 1}, resume=False)
    (root / "shards" / ".000000.json.interrupted").write_text(
        json.dumps({"rows": [{"row": 1}]}), encoding="utf-8"
    )
    assert store.read(0, {"axis": "warmth"}) is None
    with pytest.raises(RuntimeError, match="incomplete"):
        store.consolidate(1)


def test_ccu_queue_is_serial_fail_closed_and_h100_only() -> None:
    runner = (ROOT / "jobs/ccu/run_gemma4_calibrated.sh").read_text()
    queue = (ROOT / "jobs/ccu/run_gemma4_calibrated_queue.sh").read_text()
    handoff = (ROOT / "jobs/ccu/handoff_12b_to_31b.sh").read_text()
    bootstrap = (ROOT / "jobs/ccu/bootstrap_gemma4.sh").read_text()
    requirements = (ROOT / "jobs/ccu/requirements-gemma4.txt").read_text()
    assert "*H100*" in runner
    assert "--checkpoint-dir" in runner and "--resume" in runner
    assert "12b 26b_a4b 31b" in queue
    assert "stopped after technical failure" in queue
    assert "26b_a4b=delegated_scckn" in handoff
    assert "run_gemma4_calibrated.sh 31b" in handoff
    assert "torch==2.6.0+cu124" in bootstrap
    assert "scikit-learn==1.8.0" in requirements
    assert "scipy==1.17.0" in requirements
    for script in (
        "jobs/ccu/bootstrap_gemma4.sh",
        "jobs/ccu/run_gemma4_calibrated.sh",
        "jobs/ccu/run_gemma4_calibrated_queue.sh",
        "jobs/ccu/handoff_12b_to_31b.sh",
    ):
        subprocess.run(["bash", "-n", script], cwd=ROOT, check=True)
