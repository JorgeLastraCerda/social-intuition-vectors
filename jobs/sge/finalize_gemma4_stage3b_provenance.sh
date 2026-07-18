#!/bin/bash
# Login-node postflight: capture qacct, hashes, raw logs, then sync once.
set -euo pipefail

RUN_ID="${1:?usage: bash jobs/sge/finalize_gemma4_stage3b_provenance.sh RUN_ID}"
cd "$(git rev-parse --show-toplevel)"
REPO_PATH="$(pwd)"
MANIFEST="results/logs/gemma4_stage3b_submission_${RUN_ID}.json"
OUTCOME="results/logs/gemma4_stage3b_outcome_${RUN_ID}.json"
[[ -f "$MANIFEST" ]] || { echo "Missing submission manifest: $MANIFEST" >&2; exit 2; }
[[ ! -e "$OUTCOME" ]] || { echo "Refusing to overwrite: $OUTCOME" >&2; exit 3; }

RUN_ID="$RUN_ID" MANIFEST="$MANIFEST" OUTCOME="$OUTCOME" python - <<'PY'
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

run_id = os.environ["RUN_ID"]
manifest_path = Path(os.environ["MANIFEST"])
outcome_path = Path(os.environ["OUTCOME"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
final = next(job for job in manifest["jobs"] if job.get("role") == "finalizer")
if not Path(final["success_sentinel"]).exists():
    raise SystemExit(f"Missing final success sentinel: {final['success_sentinel']}")

def qacct(job_id: str) -> dict[str, str]:
    result = subprocess.run(
        ["qacct", "-j", str(job_id)], check=True, capture_output=True, text=True
    )
    parsed = {}
    for line in result.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            parsed[parts[0]] = parts[1].strip()
    if parsed.get("failed") != "0" or parsed.get("exit_status") != "0":
        raise SystemExit(f"Job {job_id} failed accounting gate: {parsed}")
    return {key: parsed.get(key, "") for key in (
        "jobnumber", "qname", "hostname", "failed", "exit_status",
        "ru_wallclock", "maxvmem", "start_time", "end_time",
    )}

log_paths = []
for suffix in ("12b", "26b", "31b", "final"):
    for extension in ("out", "err"):
        log_paths.append(Path(f"results/logs/gemma4_stage3b_{run_id}_{suffix}.{extension}"))
artifact_paths = []
for job in manifest["jobs"]:
    artifact_paths.extend(Path(path) for path in job.get("outputs", []))
for path in [*artifact_paths, *log_paths]:
    if not path.exists():
        raise SystemExit(f"Missing Stage 3B artifact: {path}")

def digest(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

payload = {
    "run_id": run_id,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "complete",
    "git_commit": manifest["git_commit"],
    "submission_manifest": str(manifest_path),
    "accounting": {
        str(job["job_id"]): qacct(str(job["job_id"])) for job in manifest["jobs"]
    },
    "artifacts": [digest(path) for path in artifact_paths],
    "raw_logs": [digest(path) for path in log_paths],
}
outcome_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"[outcome] {outcome_path}")
PY

bash jobs/sync_outputs.sh "$REPO_PATH"
