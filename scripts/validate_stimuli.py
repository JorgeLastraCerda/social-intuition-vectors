#!/usr/bin/env python3
"""
validate_stimuli.py
-------------------
Mechanical quality-control for the concept story corpus
(data/stimuli/concept_stories.jsonl).

This script does NOT trust that the generator (API or a chat model) followed
the rules. It re-checks every story and refuses to pass rule violations.
Run it after every generation batch.

Checks: (1) valid JSON + required fields; (2) known condition; (3) unique id;
(4) no forbidden word/variant for the condition (the quality must be SHOWN,
not named); (5) word count in a sane band so length is not a confound.
Exit code is non-zero on any hard violation (1-4) so it can gate a pipeline.

Usage:
    python scripts/validate_stimuli.py
    python scripts/validate_stimuli.py --min-words 110 --max-words 190
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CONDITIONS = ["high_warmth", "low_warmth", "high_competence", "low_competence"]
REQUIRED_FIELDS = ["id", "condition", "topic_idx", "topic", "text", "generation_model"]

TARGET_PER_CONDITION = 1200
TARGET_TOTAL = 4800

# Forbidden stems per condition, matched with a word-boundary prefix (\b<stem>)
# so variants are caught ("warm" -> warmth/warmer/warmly). Note "friendl" (not
# "friend") so "friendly/friendliness" is caught but the noun "friend" is not.
FORBIDDEN_STEMS = {
    "high_warmth": ["warm", "kind", "caring", "empath", "compassion", "friendl", "tender", "nurtur"],
    "low_warmth": ["cold", "unfriendly", "distant", "aloof", "uncaring", "indifferen", "hostile", "callous"],
    "high_competence": ["competent", "competence", "skill", "capable", "capabilit", "expert",
                        "proficien", "efficien", "adept", "masterful"],
    "low_competence": ["incompetent", "incompetence", "unskilled", "incapable", "inefficien",
                       "clumsy", "inept", "bungl"],
}


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def find_forbidden(text: str, condition: str) -> list:
    hits = []
    low = text.lower()
    for stem in FORBIDDEN_STEMS.get(condition, []):
        if re.search(r"\b" + re.escape(stem), low):
            hits.append(stem)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="data/stimuli/concept_stories.jsonl")
    ap.add_argument("--min-words", type=int, default=90)
    ap.add_argument("--max-words", type=int, default=200)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[error] file not found: {path}")
        return 2

    violations = []
    length_warnings = []
    seen_ids = set()
    counts = defaultdict(int)
    topics_seen = defaultdict(set)
    lengths = defaultdict(list)

    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                violations.append(f"line {n}: invalid JSON ({e})")
                continue

            missing = [k for k in REQUIRED_FIELDS if k not in rec]
            if missing:
                violations.append(f"line {n}: missing fields {missing}")
                continue

            cond = rec["condition"]
            sid = rec["id"]
            if cond not in CONDITIONS:
                violations.append(f"{sid}: unknown condition '{cond}'")
                continue
            if sid in seen_ids:
                violations.append(f"{sid}: duplicate id")
                continue
            seen_ids.add(sid)

            forbidden = find_forbidden(rec["text"], cond)
            if forbidden:
                violations.append(f"{sid}: forbidden word(s) for {cond}: {forbidden}")

            wc = count_words(rec["text"])
            if wc < args.min_words or wc > args.max_words:
                length_warnings.append(f"{sid}: {wc} words (outside {args.min_words}-{args.max_words})")

            counts[cond] += 1
            topics_seen[cond].add(rec["topic_idx"])
            lengths[cond].append(wc)

    total = sum(counts.values())
    print("=" * 64)
    print(f"STIMULI VALIDATION  --  {path}")
    print("=" * 64)
    print(f"{'condition':<18}{'count':>7}{'target':>8}{'remain':>8}{'topics':>8}{'words(min/mean/max)':>22}")
    for c in CONDITIONS:
        ls = lengths[c]
        wstat = f"{min(ls)}/{round(sum(ls)/len(ls))}/{max(ls)}" if ls else "-"
        print(f"{c:<18}{counts[c]:>7}{TARGET_PER_CONDITION:>8}"
              f"{TARGET_PER_CONDITION - counts[c]:>8}{len(topics_seen[c]):>8}{wstat:>22}")
    print("-" * 64)
    print(f"{'TOTAL':<18}{total:>7}{TARGET_TOTAL:>8}{TARGET_TOTAL - total:>8}")
    print()

    if length_warnings:
        print(f"[length warnings: {len(length_warnings)}]")
        for w in length_warnings:
            print("  ~", w)
        print()

    if violations:
        print(f"[HARD VIOLATIONS: {len(violations)}]  --> fix or regenerate these stories")
        for v in violations:
            print("  X", v)
        print()
        print("RESULT: FAIL")
        return 1

    print("RESULT: PASS  (no rule violations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
