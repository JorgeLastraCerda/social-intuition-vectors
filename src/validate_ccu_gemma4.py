"""Validate CCU H100 Gemma 4 smoke provenance and resource gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.config import load_config


def validate_smoke(config: str, path: str, *, total_vram_gib: float) -> dict:
    cfg = load_config(config)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    runtime = payload["runtime"]
    if payload["model"] != cfg.model.name:
        raise AssertionError("Smoke model mismatch.")
    if payload["revision"] != cfg.model.revision:
        raise AssertionError("Smoke requested revision mismatch.")
    if runtime["model_revision_resolved"] != cfg.model.revision:
        raise AssertionError("Smoke resolved revision mismatch.")
    if (payload["n_layers"], payload["d_model"]) != (
        cfg.smoke.expected_layers,
        cfg.smoke.expected_d_model,
    ):
        raise AssertionError("Smoke architecture mismatch.")
    if payload["bridge_hf_max_logit_diff"] > 0.02:
        raise AssertionError("Bridge/HF logit parity exceeds 0.02.")
    if payload["baseline_margin"] == payload["steered_margin"]:
        raise AssertionError("Smoke steering did not change the output.")
    visible = runtime.get("visible_cuda_devices", [])
    if len(visible) != 1 or "H100" not in visible[0].get("name", ""):
        raise AssertionError(f"Expected one H100 in runtime metadata: {visible!r}")
    fraction = float(payload["peak_allocated_vram_gib"]) / total_vram_gib
    if fraction >= cfg.smoke.max_vram_fraction:
        raise AssertionError(
            f"Peak VRAM fraction {fraction:.3f} exceeds {cfg.smoke.max_vram_fraction:.3f}."
        )
    return {
        "status": "pass",
        "model": payload["model"],
        "revision": payload["revision"],
        "bridge_hf_max_logit_diff": payload["bridge_hf_max_logit_diff"],
        "peak_vram_fraction": fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-path", required=True)
    parser.add_argument("--total-vram-gib", required=True, type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_smoke(
                args.config, args.smoke_path, total_vram_gib=args.total_vram_gib
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
