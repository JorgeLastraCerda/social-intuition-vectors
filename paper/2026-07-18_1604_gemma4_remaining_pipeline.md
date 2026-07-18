# Gemma 4 Remaining-Test Pipeline

- **Produced:** 2026-07-18 16:04 Europe/Berlin
- **Models:** Gemma-4-12B-it, Gemma-4-26B-A4B-it, Gemma-4-31B-it
- **Scope:** SAE-free dense steering, neutral-PCA denoising, hiring audits, callback steering, and post-hoc disparity analyses
- **Status:** Implementation and local validation complete; SCCKN empirical runs pending

## Artifacts

- **Scripts:** `src/utils/model_loader.py`, `smoke_tests/gemma4_transformerlens/smoke_test_bridge.py`, `src/dense_steering.py`, `src/denoise_vectors.py`, `src/summarize_hiring_steering.py`, `src/validate_gemma4_remaining.py`, `jobs/sge/gemma4_remaining_run.sh`, `jobs/sge/submit_gemma4_remaining.sh`, `jobs/sync_outputs.sh`
- **Inputs:** `config/gemma4_12b.yaml`, `config/gemma4_26b_a4b.yaml`, `config/gemma4_31b.yaml`, `data/processed/concept_vectors_gemma4_12b/`, `data/processed/concept_vectors_gemma4_26b_a4b/`, `data/processed/concept_vectors_gemma4_31b/`, `data/stimuli/concept_stories.jsonl`, `data/raw/gallo_hausladen/`
- **Outputs:** `results/tables/steering_dense_null_gemma4_*.csv`, `results/tables/hiring_audit_gemma4_*.csv`, `results/tables/hiring_steering_{raw_,}gemma4_*.csv`, `results/tables/hiring_{disparity,group_r4,name_level}_gemma4_*.csv`, `results/logs/{steering_dense,hiring_steering,hiring_full282_gate,hiring_mediation,hiring_r4}_gemma4_*.json`, `results/logs/gemma4_remaining_submission_*.json`

## Summary

The pipeline closes the SAE-independent parts of the earlier-generation Gemma test matrix for all three Gemma 4 variants. It preserves the existing TransformerLens Bridge and bfloat16 callback-margin path for direct parity, while adding stronger cross-axis and empirical-null controls. No Gemma 4 SAE is assumed or substituted.

Each model is pinned to an exact Hugging Face revision:

| Model | Revision | Required GPU |
|---|---|---|
| Gemma-4-12B-it | `12ace6d648d72bd41519e140f1185f34d38c7e3d` | exact NVIDIA L40 |
| Gemma-4-26B-A4B-it | `01e5b3ee840d3a9e0b0b493c593e85398a30ef75` | NVIDIA RTX PRO 6000 |
| Gemma-4-31B-it | `b9ea41a2887d8607f594846523f94c6cc75ac8a4` | NVIDIA RTX PRO 6000 |

The loader records and verifies the resolved checkpoint revision and chat-template hash. It also records peak allocated and reserved VRAM. The scheduler runner rejects the wrong GPU family, insufficient free memory, changed critical code, existing output targets, or more than one visible GPU.

## Registered run set

The following jobs are independent and must not use `hold_jid`:

1. Technical smoke.
2. Neutral-corpus activation extraction (1,500 texts).
3. CPU PCA fitting and raw-direction denoising.
4. Raw dense steering with target-axis, other-axis, and 50 orthogonal random-direction controls.
5. Denoised dense steering with the same controls.
6. A 282-name unsteered hiring audit.
7. Local, broad, and denoised-local hiring steering on a seeded 60-name panel.
8. A pre-registered gate that expands all three hiring regimes to 282 names if any uncertainty, linearity, monotonicity, sign, or raw-versus-denoised stability criterion fails.
9. Post-hoc disparity, name-level R4, and bootstrap mediation analyses.

The enhanced dense design produces 10,440 rows per vector kind. For each judgment axis, 5,000 rows belong to 50 random directions. Endpoint and slope statistics for the target and cross-axis directions are evaluated against that empirical null.

Hiring summaries use a seeded paired bootstrap with 5,000 resamples. The legacy bfloat16 Yes/No callback margin remains intentionally unchanged for parity. Every summary therefore reports the number of unique margins, standard deviation, the fraction on a 0.125 grid, and a quantization warning. This design documents the known limitation rather than treating coarse margins as high-precision measurements.

## Local validation

- Full test suite: 74 passed.
- Targeted Ruff lint and formatting: passed.
- Shell syntax validation: passed.
- Independent submitter dry-runs: 33 of 33 model-by-run combinations passed.
- `git diff --check`: passed.

These checks establish implementation readiness only. Scientific conclusions require the SCCKN smoke and production outputs, which will receive separate model- and run-specific reports.
