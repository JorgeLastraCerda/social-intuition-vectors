# Qwen3.6 Resumable Parallel Pipeline

- **Produced:** 2026-07-19 09:44 Europe/Berlin
- **Models:** `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B`
- **Scope:** Resumable calibrated steering plus native-HF hiring audit and steering across two SCCKN RTX PRO 6000 GPUs and one CCU H100
- **Status:** Implementation validated; initial three independent GPU runs ready for submission

## Artifacts

- **Scripts:** `src/qwen36_calibrated_steering.py`, `src/qwen36_hiring.py`, `src/validate_qwen36_hiring.py`, `src/steering_checkpoint.py`, `jobs/sge/calibrated_steering_run.sh`, `jobs/ccu/run_qwen36_hiring.sh`
- **Inputs:** `config/qwen36_27b.yaml`, `config/qwen36_35b_a3b.yaml`, `data/stimuli/concept_stories.jsonl`, `data/processed/concept_vectors_qwen36_27b/`, `data/processed/concept_vectors_qwen36_35b_a3b/`, `data/raw/SocialPerceptions-Predict-Callback-main/0_data/ratings/names/df_all.csv`
- **Outputs:** `results/tables/steering_dense_{raw_,}qwen36_*_calibrated_*.csv`, `results/tables/steering_dense_null_qwen36_*_calibrated_*.csv`, `results/tables/hiring_audit_qwen36_*.csv`, `results/tables/hiring_steering_raw_qwen36_*.csv`, `results/logs/steering_dense_qwen36_*_calibrated_*.json`, `results/logs/hiring_probe_vs_human_qwen36_*.json`, `results/logs/hiring_steering_qwen36_*.json`

## Approach

The Qwen calibrated runner now uses the same immutable, fingerprinted checkpoint store as the validated Gemma dense-steering path. It writes one baseline shard per judgment axis and one shard per axis, intervention, direction, and strength combination. Resume requires an exact model revision, topic split, seed, argument set, and input-file hash match. Final CSV and JSON artifacts are atomically published only after all contiguous shards are present.

The new hiring path keeps Qwen isolated from TransformerLens. It uses the pinned Transformers 5.14.1 native model, raw explicit-BOS name passages for residual-stream projections, native-chat callback prompts, stable one-token Yes/No continuations, and direct PyTorch forward hooks. The audit checkpoints each of 282 names independently. Hiring steering checkpoints every baseline and every name-by-axis-by-strength unit while preserving the existing Gemma-compatible raw output schema.

## Execution design

The initial wave assigns corrected Qwen 27B calibrated steering and Qwen 35B-A3B calibrated steering to separate SCCKN RTX PRO 6000 devices. CCU H100 independently runs the 27B 282-name hiring audit. No output label is duplicated across machines, and no scheduler dependency chains the three jobs. Subsequent hiring steering, neutral extraction, denoising, post-hoc analysis, and conditional full-name expansion are released only after their own prerequisites validate.

Local validation passed 22 focused tests plus Ruff, Python compilation, shell syntax, and whitespace checks. These results establish implementation readiness, not empirical findings.
