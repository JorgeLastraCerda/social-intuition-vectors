# Gemma 4 26B-A4B Remaining-Test Smoke

- **Produced:** 2026-07-18 16:12 Europe/Berlin
- **Model:** `google/gemma-4-26B-A4B-it`
- **Scope:** Pinned TransformerLens Bridge, residual hook, native-chat margin, steering, hardware, and memory smoke
- **Status:** Complete; all technical gates passed

## Artifacts

- **Scripts:** `smoke_tests/gemma4_transformerlens/smoke_test_bridge.py`, `src/utils/model_loader.py`, `src/validate_gemma4_remaining.py`, `jobs/sge/gemma4_remaining_run.sh`
- **Inputs:** `config/gemma4_26b_a4b.yaml`
- **Outputs:** `results/logs/smoke_gemma4_26b_a4b_pinned.json`, `results/logs/gemma4_26b_a4b_smoke_20260718T140550Z_26b.{out,err}`, `results/logs/gemma4_remaining_submission_26b_a4b_smoke_20260718T140550Z_26b.json`, `results/logs/gemma4_remaining_smoke_outcome_20260718T141000Z.json`

## Results

Job `1145320` completed on one NVIDIA RTX PRO 6000 Blackwell Server Edition with `failed=0` and `exit_status=0` in 37 seconds. The requested and resolved checkpoint revisions both equal `01e5b3ee840d3a9e0b0b493c593e85398a30ef75`.

| Gate | Result |
|---|---:|
| Layers / residual width | 30 / 2,816 |
| Probe hook | `blocks.19.hook_resid_post` |
| Activation shape | `[1, 9, 2816]` |
| Bridge–HF maximum logit difference | 0.0 |
| Baseline / steered Yes-No margin | 21.3750 / 21.9375 |
| Peak allocated / reserved VRAM | 48.35 / 48.45 GiB |

The intervention changes the technical-test margin and leaves approximately 46.6 GiB between peak reserved memory and physical capacity. This is a compatibility and capacity result, not an estimate of warmth or competence causality.

## Decision

The pinned 26B-A4B checkpoint is accepted for the full SAE-free remaining-test pipeline on one RTX PRO 6000. No multi-GPU, quantized, or automatic fallback path is authorized.
