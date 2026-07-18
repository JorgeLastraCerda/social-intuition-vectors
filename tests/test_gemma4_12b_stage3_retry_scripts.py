from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "jobs/sge/submit_gemma4_12b_stage3_retry.sh"
RUNNER = ROOT / "jobs/sge/gemma4_12b_stage3_retry.sh"
FINALIZER = ROOT / "jobs/sge/gemma4_12b_stage3_finalize.sh"


def test_submitter_dry_run_describes_one_held_l40_job() -> None:
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
    assert "held model=12b hardware=L40" in output
    assert "queue=gpu@scc192,gpu@scc213" in output
    assert "gpu=1,h_vmem=32G,h_rt=01:00:00" in output
    assert "rtx_6000" not in output
    assert "finalizer queue=scc hold=12b gpu=0" in output


def test_runner_requires_l40_headroom_and_has_no_git_mutation() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'if "L40" not in name:' in text
    assert "free_gib < 30.0" in text
    assert "git pull" not in text
    assert "sync_outputs.sh" not in text
    assert "--stage 1" in text
    assert "--stage 2" in text
    assert "--stage 3" in text


def test_finalizer_requires_sentinel_before_one_sync() -> None:
    text = FINALIZER.read_text(encoding="utf-8")
    assert '[[ ! -f "$SUCCESS_SENTINEL" ]]' in text
    assert text.count('bash jobs/sync_outputs.sh "$REPO_PATH"') == 1


def test_submitter_verifies_both_job_shapes_before_submission() -> None:
    text = SUBMITTER.read_text(encoding="utf-8")
    assert text.count("qsub -w v") == 2
    assert "-hold_jid \"$job_gpu\"" in text
    assert "gemma4_stage3_retry_submission_12b_${RUN_ID}.json" in text
