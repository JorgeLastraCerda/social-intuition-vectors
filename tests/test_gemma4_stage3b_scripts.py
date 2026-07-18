from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "jobs/sge/submit_gemma4_stage3b.sh"
RUNNER = ROOT / "jobs/sge/gemma4_stage3b.sh"
FINALIZER = ROOT / "jobs/sge/gemma4_stage3b_finalize.sh"
POSTFLIGHT = ROOT / "jobs/sge/finalize_gemma4_stage3b_provenance.sh"


def test_submitter_dry_run_has_three_independent_held_jobs() -> None:
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
    output = result.stdout
    assert output.count("held model=") == 3
    assert output.count("predecessor=none") == 3
    assert "queue=gpu@scc192 hardware=NVIDIA_L40" in output
    assert output.count("queue=gpu@scc214 hardware=RTX_PRO_6000") == 2
    assert "finalizer queue=scc hold=12b,26b,31b gpu=0" in output


def test_runner_is_write_once_and_gpu_jobs_do_not_mutate_git() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "--validation-profile stage3b" in text
    assert "--require-absent" in text
    assert "--n-bootstrap 1000" in text
    assert "git pull" not in text
    assert "git push" not in text
    assert "sync_outputs.sh" not in text


def test_finalizer_validates_all_models_before_postflight_sync() -> None:
    text = FINALIZER.read_text(encoding="utf-8")
    assert text.count("--analysis-profile stage3b") == 3
    assert "sync_outputs.sh" not in text
    postflight = POSTFLIGHT.read_text(encoding="utf-8")
    assert 'qacct", "-j"' in postflight
    assert "sha256" in postflight
    assert "bash jobs/sync_outputs.sh" in postflight
