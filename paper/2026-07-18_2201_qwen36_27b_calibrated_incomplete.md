# Qwen3.6 27B Calibrated Steering Validation Failure

- **Produced:** 2026-07-18 22:01 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-27B`
- **Scope:** Diagnosis of the completed native-HF calibrated-steering pilot
- **Status:** Incomplete and not accepted as a full calibrated result

## Artifacts

- **Scripts:** `src/qwen36_calibrated_steering.py`, `src/validate_calibrated_steering.py`, `jobs/sge/calibrated_steering_run.sh`
- **Inputs:** `config/qwen36_27b.yaml`, `data/stimuli/concept_stories.jsonl`, `data/processed/concept_vectors_qwen36_27b/`
- **Outputs:** `results/tables/steering_dense_raw_qwen36_27b_calibrated.csv`, `results/tables/steering_dense_qwen36_27b_calibrated.csv`, `results/tables/steering_dense_null_qwen36_27b_calibrated.csv`, `results/logs/steering_dense_qwen36_27b_calibrated.json`, `results/logs/calibrated_steering_qwen36_27b_20260718T155600Z_qwen36_27b_retry1.{out,err}`

## Finding

Job `1145435` completed native-HF inference without TransformerLens and remained within the expected RTX PRO 6000 memory envelope. Its final validator correctly rejected the artifact because it contains 16,176 rather than 40,440 raw rows. Only 16 baseline rows were produced, which means four held-out stories per condition pair were evaluated instead of twenty.

The cause is a topic-identity bug in the Qwen-specific runner. It samples test indices from `0..49`, while the shared stimulus file contains 50 non-contiguous topic identifiers distributed across `0..97`. Several sampled integers therefore do not name any story. This is a code-level selection error, not a Qwen model, hook, memory, or library limitation.

The generated files are preserved for diagnosis but must not be cited as a completed calibrated Qwen result. No Qwen rerun is launched as part of the current CCU Gemma 4 queue.
