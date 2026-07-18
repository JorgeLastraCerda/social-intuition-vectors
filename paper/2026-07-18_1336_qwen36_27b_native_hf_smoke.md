# Qwen3.6 27B Native Hugging Face Stage 1–3 Smoke

- **Produced:** 2026-07-18 13:36 CEST
- **Model:** `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- **Scope:** Single-GPU technical smoke for Stage 1 extraction, Stage 2 probe validation, and Stage 3 all-layer measurement on 40 concept stories
- **Status:** Complete; all technical gates passed on one RTX PRO 6000 without TransformerLens

## Artifacts

- **Scripts:** `src/qwen36_smoke.py`; `src/validate_qwen36_smoke.py`; `jobs/setup_qwen36_env.sh`; `jobs/sge/qwen36_smoke.sh`; `jobs/sge/qwen36_smoke_finalize.sh`; `jobs/sge/submit_qwen36_smoke.sh`
- **Inputs:** `config/qwen36_smoke.yaml`; `data/stimuli/concept_stories.jsonl`; `requirements-qwen36.txt`
- **Outputs:** `data/processed/concept_vectors_qwen36_27b_smoke/`; `results/tables/probe_metrics_qwen36_27b_smoke.csv`; `results/tables/layer_sweep_qwen36_27b_smoke.csv`; `results/tables/layer_sweep_qwen36_27b_smoke.meta.json`
- **Logs:** `results/logs/qwen36_smoke_submission_20260718T112311Z.json`; `results/logs/smoke_qwen36_27b.json`; `results/logs/validate_probes_qwen36_27b_smoke.json`

## Execution outcome

Grid Engine job `1145040` ran on one NVIDIA RTX PRO 6000 Blackwell Server Edition on `scc214`; dependent CPU finalizer `1145041` revalidated and synchronized the outputs. Both jobs finished with `failed=0` and `exit_status=0`. The GPU job took 545 seconds including the initial 51 GiB model download, and Grid Engine reported 52.164 GB maximum virtual memory. The runtime measured 51.227 GiB peak reserved VRAM, 53.9% of the visible 95.010 GiB device.

The isolated environment used PyTorch 2.13.0+cu130, Transformers 5.14.1, Accelerate 1.14.0, and native PyTorch forward hooks. TransformerLens was not installed or imported. The checkpoint resolves internally to `Qwen3_5ForConditionalGeneration` with a `Qwen3_5TextModel`; these are the Transformers implementation class names for the pinned `Qwen/Qwen3.6-27B` revision, not a substituted checkpoint.

## Technical gates

| Gate | Result |
|---|---:|
| Requested versus resolved revision | Exact match |
| TransformerLens installed/imported | No / no |
| Visible parameter devices | `cuda:0` only |
| Vision forward calls | 0 |
| Hook versus `hidden_states[43]` maximum difference | 0.0 |
| Passive hook versus baseline logits maximum difference | 0.0 |
| Token-length range after explicit BOS | 103–125 |
| Stage 2 versus Stage 3 probe-layer tolerance | Passed at `1e-6` |

These checks establish that native Hugging Face hooks expose the intended residual stream without altering logits and that the text-only path does not execute the vision tower.

## Stage 1 and Stage 2 smoke metrics

| Axis | Direction norm | High/low cosine | Cohen's d | Random 5-fold CV | Topic-holdout CV |
|---|---:|---:|---:|---:|---:|
| Warmth | 9.679317 | 0.990779 | 9.531037 | 1.00 | 1.00 |
| Competence | 11.162744 | 0.988024 | 10.469681 | 1.00 | 1.00 |

The probe layer is layer 42 of 64 (`frac=0.6667`). The smoke uses ten seeded topics and one story per condition per topic, for 40 stories total. The perfect scores therefore verify pipeline behavior and strong separability in this small subset; they are not substitutes for the planned full-data estimates.

## Stage 3 smoke profile

| Quantity | Layer | Fraction | Value |
|---|---:|---:|---:|
| Maximum warmth d | 28 | 0.4444 | 12.487637 |
| Maximum competence d | 17 | 0.2698 | 11.618631 |
| Maximum cos(W,C) | 32 | 0.5079 | 0.544980 |
| Probe layer | 42 | 0.6667 | d = 9.531037 / 10.469681; cos = 0.513844 |

Both axes first reach simultaneous perfect topic-holdout accuracy at layer 9. The complete 64-row table is finite and ordered, and its layer-42 metrics reproduce Stage 2 exactly at six-decimal precision.

## Decision and limitations

The smoke supports using native Hugging Face hooks for Qwen3.6 Stage 1–3 on RTX PRO 6000 hardware. It also leaves substantial VRAM headroom for the 27B checkpoint. Full runs should retain the pinned revision, explicit-BOS tokenization, text-only vision-call gate, hook/hidden-state parity gate, passive-logit parity gate, and per-run VRAM logging.

No scientific conclusion should be drawn from the smoke effect sizes or depth maxima because only 40 synthetic stories were used and Stage 3 reused the same activation buffer. The result authorizes full-run planning for the two selected Qwen3.6 models; it does not itself authorize launching those larger jobs.
