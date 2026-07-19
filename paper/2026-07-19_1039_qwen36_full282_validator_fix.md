# Qwen3.6 Full-282 Validator Recovery Fix

- **Produced:** 2026-07-19 10:39 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-35B-A3B`
- **Scope:** Backward-compatible validation and recovery after a completed full-name run lacked a success sentinel
- **Status:** Implemented and locally validated; empirical output preserved

## Artifacts

- **Scripts:** `src/validate_qwen36_hiring.py`, `jobs/ccu/run_qwen36_hiring.sh`
- **Inputs:** `results/tables/hiring_steering_raw_qwen36_35b_a3b_local_full282.csv`, `results/logs/hiring_steering_qwen36_35b_a3b_local_full282.json`
- **Outputs:** `tests/test_qwen36_hiring.py`

## Finding and Fix

The first 35B-A3B local full-282 run completed all 3,102 checkpoint work units and atomically published 2,820 rows, but the post-run validator still defaulted to the legacy 60-name expectation. It rejected the structurally correct 282-name artifact before the runner could create its success sentinel. No model inference, checkpoint, or output row failed.

The validator now infers the expected name count from the run metadata unless the caller explicitly supplies a count, and it cross-checks explicit values against metadata. Existing 60-name outputs retain the same validation contract. The runner also gains a recovery branch: when both final artifacts exist but the sentinel does not, it validates the published pair and atomically creates the sentinel without loading the model or recomputing completed work.

Four focused tests, Ruff, shell syntax, and whitespace checks passed. Recovery must still be executed from the pinned fixed commit before the result is accepted.
