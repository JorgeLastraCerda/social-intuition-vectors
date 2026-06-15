#!/usr/bin/env python3
"""
audit_stimuli.py
----------------
Advisory coverage + balance report for the concept story corpus.
Unlike validate_stimuli.py (which is pass/fail on hard rules), this script is
informational: it tells you (a) how topic coverage is progressing toward the
12-stories-per-topic target, and (b) whether protagonist demographics are
balanced across the four conditions.

Why the demographic audit matters
----------------------------------
The concept stories teach the model the *warmth* and *competence* directions.
If, say, the "low" conditions are full of one gender/name-origin and the "high"
conditions full of another, those directions get entangled with demographics
and the whole study is confounded. This script exists to catch that — including
bias introduced by whichever model generated the stories.

Inputs
------
- data/stimuli/concept_stories.jsonl       (condition, topic_idx)
- data/stimuli/protagonist_metadata.jsonl  (gender, name_origin, age/disability/religion cues)
  joined on `id`. If metadata is missing, only the topic report runs.

Usage
-----
    python scripts/audit_stimuli.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CONDITIONS = ["high_warmth", "low_warmth", "high_competence", "low_competence"]
STORIES_PER_TOPIC = 12   # per (topic, condition), from config
N_TOPICS = 100
PROTECTED = ["gender", "name_origin", "age_cue", "disability_cue", "religion_cue"]
SKEW_THRESHOLD = 0.70    # warn if one level exceeds this share within a condition


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def bar(n: int, total: int, width: int = 20) -> str:
    if total == 0:
        return ""
    filled = round(width * n / total)
    return "#" * filled + "." * (width - filled)


def topic_report(stories: list[dict]) -> None:
    print("=" * 64)
    print("TOPIC COVERAGE")
    print("=" * 64)
    # per topic: count per condition
    per_topic: dict[int, Counter] = defaultdict(Counter)
    for s in stories:
        per_topic[s["topic_idx"]][s["condition"]] += 1

    touched = sorted(per_topic)
    full = [t for t in touched
            if all(per_topic[t][c] >= STORIES_PER_TOPIC for c in CONDITIONS)]
    print(f"topics touched: {len(touched)} / {N_TOPICS}    "
          f"topics at full depth ({STORIES_PER_TOPIC}/condition): {len(full)}")
    print()
    print(f"{'topic':>5} | {'HW':>3} {'LW':>3} {'HC':>3} {'LC':>3} | total")
    for t in touched:
        c = per_topic[t]
        row = [c['high_warmth'], c['low_warmth'], c['high_competence'], c['low_competence']]
        print(f"{t:>5} | {row[0]:>3} {row[1]:>3} {row[2]:>3} {row[3]:>3} | {sum(row):>4}")
    print()


def demographic_report(stories: list[dict], meta: list[dict]) -> None:
    if not meta:
        print("[no protagonist_metadata.jsonl found — skipping demographic audit]")
        return
    meta_by_id = {m["id"]: m for m in meta}
    cond_by_id = {s["id"]: s["condition"] for s in stories}

    print("=" * 64)
    print("DEMOGRAPHIC BALANCE  (protagonist cues, by condition)")
    print("=" * 64)
    print("note: name_origin is a coarse INFERRED cue from the name, used only")
    print("to audit spread — not a claim about any real person's identity.")
    print()

    warnings = []
    for attr in PROTECTED:
        print(f"--- {attr} ---")
        # condition -> Counter of levels
        table: dict[str, Counter] = {c: Counter() for c in CONDITIONS}
        overall = Counter()
        for m in meta:
            cid = m["id"]
            cond = cond_by_id.get(cid)
            if cond is None:
                continue
            level = m.get(attr, "missing")
            table[cond][level] += 1
            overall[level] += 1
        levels = sorted(overall)
        header = "condition".ljust(18) + "".join(f"{lv[:12]:>14}" for lv in levels)
        print(header)
        for c in CONDITIONS:
            n = sum(table[c].values())
            cells = "".join(f"{table[c][lv]:>14}" for lv in levels)
            print(f"{c:<18}{cells}")
            # skew check
            if n:
                top_level, top_n = table[c].most_common(1)[0]
                if top_n / n > SKEW_THRESHOLD and len(levels) > 1:
                    warnings.append(
                        f"{attr}: '{c}' is {round(100*top_n/n)}% '{top_level}' "
                        f"({top_n}/{n}) — possible confound with condition")
        print(f"{'OVERALL':<18}" + "".join(f"{overall[lv]:>14}" for lv in levels))
        print()

    print("=" * 64)
    if warnings:
        print(f"BALANCE WARNINGS: {len(warnings)}")
        for w in warnings:
            print("  !", w)
        print()
        print("If a demographic level lines up with a condition, the warmth/")
        print("competence vector will partly encode that demographic. Rebalance")
        print("by assigning names from data/stimuli/name_roster.csv evenly across")
        print("all four conditions.")
    else:
        print("No demographic skew above threshold. Conditions look balanced.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stories", default="data/stimuli/concept_stories.jsonl")
    ap.add_argument("--meta", default="data/stimuli/protagonist_metadata.jsonl")
    args = ap.parse_args()

    stories = load_jsonl(Path(args.stories))
    meta = load_jsonl(Path(args.meta))
    if not stories:
        print(f"[error] no stories at {args.stories}")
        return 2

    topic_report(stories)
    demographic_report(stories, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
