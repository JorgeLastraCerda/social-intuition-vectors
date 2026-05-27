from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ALLOWED_TEXT_TYPES = {"psywar", "neutral", "emotional_non_propaganda"}
PLACEHOLDER_MARKERS = (
    "[Paste the quote",
    "[Alıntıyı",
    "[TEXT]",
)


@dataclass(frozen=True)
class TextRecord:
    text_id: str
    text_type: str
    title: str
    source: str
    date_or_period: str
    conflict: str
    speaker_or_origin: str
    target_audience: str
    notes: str
    text: str


def load_quote_markdown(path: Path) -> list[TextRecord]:
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^---\s*$", content)
    records: list[TextRecord] = []

    for block in blocks:
        heading = re.search(r"(?m)^##\s+(TEXT_\d+)\s*$", block)
        if not heading:
            continue

        metadata = _parse_metadata(block)
        text = _parse_text_fence(block)
        text_id = metadata.get("text_id") or heading.group(1)
        text_type = metadata.get("text_type", "")

        records.append(
            TextRecord(
                text_id=text_id,
                text_type=text_type,
                title=metadata.get("title", ""),
                source=metadata.get("source", ""),
                date_or_period=metadata.get("date_or_period", ""),
                conflict=metadata.get("conflict", ""),
                speaker_or_origin=metadata.get("speaker_or_origin", ""),
                target_audience=metadata.get("target_audience", ""),
                notes=metadata.get("notes", ""),
                text=text,
            )
        )

    return records


def validate_records(records: list[TextRecord], allow_placeholders: bool = False) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()

    if not records:
        errors.append("No TEXT_### blocks found.")

    for record in records:
        if record.text_id in seen:
            errors.append(f"{record.text_id}: duplicate text_id.")
        seen.add(record.text_id)

        if record.text_type not in ALLOWED_TEXT_TYPES:
            errors.append(
                f"{record.text_id}: text_type must be one of {sorted(ALLOWED_TEXT_TYPES)}, got {record.text_type!r}."
            )
        if not record.text.strip():
            errors.append(f"{record.text_id}: text block is empty.")
        if not allow_placeholders and any(marker in record.text for marker in PLACEHOLDER_MARKERS):
            errors.append(f"{record.text_id}: placeholder text is still present.")

    return errors


def print_corpus_summary(records: list[TextRecord]) -> None:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.text_type] = counts.get(record.text_type, 0) + 1

    print("[corpus] loaded records:", len(records), flush=True)
    for key in sorted(counts):
        print(f"[corpus] {key}: {counts[key]}", flush=True)


def _parse_metadata(block: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for match in re.finditer(r"(?m)^\*\*([a-zA-Z0-9_]+):\*\*\s*(.*?)\s*$", block):
        metadata[match.group(1)] = match.group(2).strip()
    return metadata


def _parse_text_fence(block: str) -> str:
    match = re.search(r"```text\s*(.*?)\s*```", block, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()
