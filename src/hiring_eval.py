"""Hiring evaluation dispatcher (Phase 7).

This module is retained for backwards compatibility.  The actual Phase 7
hiring evaluation is split across three model-agnostic scripts:

  src/hiring_steering.py  — causal sweep (GPU); replaces notebook 06
  src/hiring_audit.py     — probe-vs-human validation + baseline (GPU); replaces notebook 07
  src/hiring_disparity.py — disparity analysis + bootstrap mediation (CPU only)

Run them in order for each model:

    python -m src.hiring_steering  --vectors-subdir <dir> --label <label>
    python -m src.hiring_audit     --vectors-subdir <dir> --label <label>
    python -m src.hiring_disparity --label <label>

SGE job scripts are in jobs/sge/hiring_*.sh.
"""
from __future__ import annotations

import argparse

from src.utils.config import load_config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print("[config] raw_data:", config.paths.raw_data, flush=True)
    if args.dry_run:
        print("[dry-run] hiring_eval is a dispatcher stub.", flush=True)
        print(
            "[dry-run] Use src/hiring_steering.py, src/hiring_audit.py, "
            "and src/hiring_disparity.py for Phase 7.",
            flush=True,
        )
        return
    raise SystemExit(
        "src/hiring_eval.py is a compatibility stub.\n"
        "Run src/hiring_steering.py + src/hiring_audit.py + src/hiring_disparity.py instead."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hiring evaluation dispatcher (stub). "
            "Phase 7 logic lives in src/hiring_{steering,audit,disparity}.py."
        )
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
