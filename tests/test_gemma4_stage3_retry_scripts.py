from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "jobs/sge/submit_gemma4_stage3_retry.sh"
RUNNER = ROOT / "jobs/sge/gemma4_stage3_retry.sh"
FINALIZER = ROOT / "jobs/sge/gemma4_stage3_finalize.sh"


def test_retry_submitter_dry_run_has_two_independent_held_gpu_jobs() -> None:
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
    assert "held model=26b" in output
    assert "held model=31b" in output
    assert output.count("predecessor=none") == 2
    assert output.count("gpu=1,rtx_6000=1") == 2
    assert "finalizer queue=scc hold=26b,31b gpu=0" in output


def test_gpu_runner_has_no_git_pull_or_output_sync() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "git pull" not in text
    assert "sync_outputs.sh" not in text
    assert "--stage 1" in text
    assert "--stage 2" in text
    assert "--stage 3" in text


def test_finalizer_requires_both_sentinels_before_one_sync() -> None:
    text = FINALIZER.read_text(encoding="utf-8")
    assert '[[ -f "$SENTINEL_26B" ]]' in text
    assert '[[ -f "$SENTINEL_31B" ]]' in text
    assert text.count('bash jobs/sync_outputs.sh "$REPO_PATH"') == 1
