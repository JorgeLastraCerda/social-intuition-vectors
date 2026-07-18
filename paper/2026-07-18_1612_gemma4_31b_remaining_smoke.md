# Gemma 4 31B Remaining-Test Smoke

- **Produced:** 2026-07-18 16:12 Europe/Berlin
- **Model:** `google/gemma-4-31B-it`
- **Scope:** Pinned TransformerLens Bridge, residual hook, native-chat margin, steering, hardware, and memory smoke
- **Status:** Complete; all technical gates passed

## Artifacts

- **Scripts:** `smoke_tests/gemma4_transformerlens/smoke_test_bridge.py`, `src/utils/model_loader.py`, `src/validate_gemma4_remaining.py`, `jobs/sge/gemma4_remaining_run.sh`
- **Inputs:** `config/gemma4_31b.yaml`
- **Outputs:** `results/logs/smoke_gemma4_31b_pinned.json`, `results/logs/gemma4_31b_smoke_20260718T140550Z_31b.{out,err}`, `results/logs/gemma4_remaining_submission_31b_smoke_20260718T140550Z_31b.json`, `results/logs/gemma4_remaining_smoke_outcome_20260718T141000Z.json`

## Results

Job `1145322` completed on one NVIDIA RTX PRO 6000 Blackwell Server Edition with `failed=0` and `exit_status=0` in 52 seconds. The requested and resolved checkpoint revisions both equal `b9ea41a2887d8607f594846523f94c6cc75ac8a4`.

| Gate | Result |
|---|---:|
| Layers / residual width | 60 / 5,376 |
| Probe hook | `blocks.39.hook_resid_post` |
| Activation shape | `[1, 9, 5376]` |
| Bridge–HF maximum logit difference | 0.0 |
| Baseline / steered Yes-No margin | 25.7500 / 25.4688 |
| Peak allocated / reserved VRAM | 58.50 / 58.87 GiB |

The technical intervention changes the margin without producing nonfinite values. Approximately 36.1 GiB remains between peak reserved memory and physical capacity. The single-prompt sign is not interpreted as a warmth or competence effect.

## Decision

The pinned 31B checkpoint is accepted for the full SAE-free remaining-test pipeline on one RTX PRO 6000. Full runs must preserve the exact revision, native chat template, bfloat16 parity path, and write-once output gates.
