# Execution Plan: Warmth & Competence Probing

This is the master implementation plan for the active repository. Work through phases in order and report after each phase. Do not skip setup, source acquisition, or method-reading phases.

## Phase 0: Source Acquisition

Download and verify the two source papers:

- Anthropic emotion concepts paper: `papers/emotion_concepts_anthropic_2026.pdf`
- PLOS ONE warmth/competence callback paper: `papers/warmth_competence_callback_plos_2024.pdf`

Also fetch the PLOS supplementary data or repository contents into `data/raw/`.

Verification:

- `file papers/*.pdf` reports PDF documents.
- PDF sizes are non-trivial.
- Any download failure is reported with the exact error.

## Phase 1: Project Scaffolding

Create and maintain the root project structure:

```text
config/
papers/
data/raw/
data/stimuli/
data/processed/
src/
src/utils/
docs/
jobs/sge/
notebooks/
results/figures/
results/tables/
results/logs/
tests/
```

Repository rules:

- All active docs, code, comments, filenames, and scripts are English.
- `archive/status_quo_embedding_geometry/` is unrelated historical material.
- `archive/target_self_affect_leakage/` is the core prior source for workflow and SCCKN operational conventions.
- Heavy jobs target SCCKN Grid Engine first, via `jobs/sge/`.

## Phase 2: Method Notes

Read the downloaded papers before writing experimental logic. Write `docs/METHOD_NOTES.md` with:

- the exact activation extraction recipe,
- concept vector construction,
- neutral-corpus PCA denoising approach,
- probe validation approach,
- causal steering design,
- mapping from PLOS social signals to CV/hiring prompts.

## Phase 3: Stimulus Generation

Implement `src/generate_stimuli.py`.

Stimulus sets:

- Concept story corpus for high/low warmth and high/low competence.
- Hiring stimuli based on PLOS social signals and matched CV templates.

Outputs:

- `data/stimuli/concept_stories.jsonl`
- `data/stimuli/hiring_prompts.jsonl`
- validation summary in `results/tables/`

## Phase 4: Vector Extraction

Implement `src/extract_vectors.py`.

Core behavior:

- Load the configured open-weights model.
- Capture residual-stream activations at the layer selected by `probing.probe_layer_frac`.
- Average activations from `probing.start_token` onward.
- Build warmth and competence vectors.
- Save vectors and metadata under `data/processed/`.

## Phase 5: Probe Validation

Implement `src/validate_probes.py`.

Validation targets:

- held-out text classification / activation sanity checks,
- logit-lens or nearest-token sanity checks where feasible,
- correlation with human warmth/competence ratings from PLOS data.

Outputs:

- figures in `results/figures/`,
- tables in `results/tables/`,
- reproducible metadata in `results/logs/`.

## Phase 6: Causal Steering

Implement `src/steering.py`.

Core behavior:

- Apply warmth/competence steering at configured strengths.
- Measure callback logit/probability changes.
- Compare steered and unsteered conditions.
- Save raw results and summary tables.

## Phase 7: Hiring Benchmark

Implement `src/hiring_eval.py`.

Core behavior:

- Run baseline callback evaluation across social signals.
- Compare model callback disparities with human callback disparities.
- Relate callback rates to probe activations.

Headline outputs:

- Probe activation vs human warmth/competence rating.
- Steering strength vs callback-rate shift.
- Model callback disparity vs human callback disparity.

## Model Selection

Keep the model unset until compute is confirmed. Candidate defaults:

- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

Selection criteria:

- open weights,
- residual-stream access,
- reliable binary callback instruction following,
- fits in the available GPU allocation.

## SCCKN Execution

SCCKN uses Grid Engine. Use `qsub`, not `sbatch`.

Heavy phases should have scripts under `jobs/sge/`. Leave resource values as `# ADJUST` placeholders until the target queue, GPU node, module stack, and scratch path are confirmed.

Useful commands:

```bash
qsub jobs/sge/<job>.sh
qstat -u emrecan.ulu
qdel <job_id>
qacct -j <job_id>
```
