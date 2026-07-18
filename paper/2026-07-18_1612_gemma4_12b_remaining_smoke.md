# Gemma 4 12B Remaining-Test Smoke

- **Produced:** 2026-07-18 16:12 Europe/Berlin
- **Model:** `google/gemma-4-12B-it`
- **Scope:** Pinned TransformerLens Bridge, residual hook, native-chat margin, steering, hardware, and memory smoke
- **Status:** Complete; all technical gates passed

## Artifacts

- **Scripts:** `smoke_tests/gemma4_transformerlens/smoke_test_bridge.py`, `src/utils/model_loader.py`, `src/validate_gemma4_remaining.py`, `jobs/sge/gemma4_remaining_run.sh`
- **Inputs:** `config/gemma4_12b.yaml`
- **Outputs:** `results/logs/smoke_gemma4_12b_pinned.json`, `results/logs/gemma4_12b_smoke_20260718T140550Z_12b.{out,err}`, `results/logs/gemma4_remaining_submission_12b_smoke_20260718T140550Z_12b.json`, `results/logs/gemma4_remaining_smoke_outcome_20260718T141000Z.json`

## Results

Job `1145318` completed on one exact NVIDIA L40 with `failed=0` and `exit_status=0` in 78 seconds. The requested and resolved checkpoint revisions both equal `12ace6d648d72bd41519e140f1185f34d38c7e3d`.

| Gate | Result |
|---|---:|
| Layers / residual width | 48 / 3,840 |
| Probe hook | `blocks.31.hook_resid_post` |
| Activation shape | `[1, 9, 3840]` |
| Bridge–HF maximum logit difference | 0.0 |
| Baseline / steered Yes-No margin | 17.5569 / 17.5745 |
| Peak allocated / reserved VRAM | 22.50 / 22.79 GiB |

The nonzero margin change confirms that the intervention hook is active. Its magnitude and sign are not a scientific steering result because this smoke uses one prompt and a technical test direction.

## Decision

The pinned 12B checkpoint is accepted for the full SAE-free remaining-test pipeline on exact L40 hardware. The production jobs must retain the same revision, template hash, native-chat format, and single-device check.
