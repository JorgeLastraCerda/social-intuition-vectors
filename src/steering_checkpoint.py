"""Atomic, fingerprinted checkpoint shards for long steering runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: Any) -> None:
    """Write JSON with fsync and atomic replacement in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class CheckpointStore:
    """One immutable JSON shard per deterministic unit of steering work."""

    MANIFEST_VERSION = 1

    def __init__(self, root: Path, fingerprint: dict[str, Any], *, resume: bool):
        self.root = root
        self.shards = root / "shards"
        self.manifest_path = root / "manifest.json"
        expected = {
            "manifest_version": self.MANIFEST_VERSION,
            "fingerprint": fingerprint,
        }
        if resume:
            if not self.manifest_path.exists():
                raise FileNotFoundError(
                    f"Cannot resume without checkpoint manifest: {self.manifest_path}"
                )
            observed = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if observed != expected:
                raise ValueError(
                    "Checkpoint fingerprint mismatch; refusing to mix incompatible runs."
                )
        else:
            if root.exists():
                raise FileExistsError(
                    f"Checkpoint directory already exists; use --resume: {root}"
                )
            self.shards.mkdir(parents=True)
            atomic_json_write(self.manifest_path, expected)
        self.shards.mkdir(parents=True, exist_ok=True)

    def shard_path(self, sequence: int) -> Path:
        return self.shards / f"{sequence:06d}.json"

    def read(self, sequence: int, key: dict[str, Any]) -> list[dict[str, Any]] | None:
        path = self.shard_path(sequence)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("sequence") != sequence or payload.get("key") != key:
            raise ValueError(f"Checkpoint shard identity mismatch: {path}")
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Checkpoint shard is empty or malformed: {path}")
        return rows

    def write(
        self, sequence: int, key: dict[str, Any], rows: list[dict[str, Any]]
    ) -> None:
        if not rows:
            raise ValueError("Refusing to checkpoint an empty work unit.")
        path = self.shard_path(sequence)
        if path.exists():
            observed = self.read(sequence, key)
            if observed != rows:
                raise FileExistsError(f"Checkpoint shard is immutable: {path}")
            return
        atomic_json_write(path, {"sequence": sequence, "key": key, "rows": rows})

    def consolidate(self, expected_units: int) -> list[dict[str, Any]]:
        paths = sorted(self.shards.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"))
        expected_names = [f"{index:06d}.json" for index in range(expected_units)]
        if [path.name for path in paths] != expected_names:
            raise RuntimeError(
                f"Checkpoint is incomplete: expected {expected_units} contiguous shards, "
                f"found {len(paths)}."
            )
        output: list[dict[str, Any]] = []
        for sequence, path in enumerate(paths):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("sequence") != sequence or not payload.get("rows"):
                raise ValueError(f"Checkpoint shard is malformed: {path}")
            output.extend(payload["rows"])
        return output
