# Repository Context for AI Agents

Read this file at the start of every session. This repository is the active workspace for:

**Probing Warmth & Competence Representations in LLM Hiring Decisions**

The project studies whether open-weights language models encode **warmth** and **competence** as linearly probeable internal representations, and whether steering those representations causally changes hiring callback recommendations.

## Archive Map

- `archive/target_self_affect_leakage/`
  - Previous project titled "Who Feels the Fear?"
  - Core prior source for workflow, phase discipline, validation/logging style, and SCCKN operational notes.
  - Do not keep its research question active unless the user explicitly connects it to the new project.
  - Useful references:
    - `archive/target_self_affect_leakage/who_feels_the_fear_project_plan.md`
    - `archive/target_self_affect_leakage/scripts/`
    - `archive/target_self_affect_leakage/scckn/`

## What This Project Is

A mechanistic-interpretability study of hiring bias in LLMs. We extract internal warmth and competence representations, inspired by the Stereotype Content Model, from an open-weights model and test whether they causally influence callback recommendations.

The methodological template is Anthropic's "Emotion Concepts and their Function in a Large Language Model" (Sofroniew, Lindsey et al., 2026). The human benchmark is Gallo, Hausladen et al. (2024), "Perceived warmth and competence predict callback rates in meta-analyzed North American labor market experiments."

## Mental Model of the Pipeline

1. Generate synthetic stories expressing high/low warmth and high/low competence.
2. Extract residual-stream activations and build warmth/competence vectors.
3. Validate probes on held-out text and against human warmth/competence ratings.
4. Steer the vectors and measure causal shifts in callback decisions.
5. Benchmark model callback disparities against human correspondence-study data.

See `PLAN.md` for the phase-by-phase execution plan.

## Hard Constraints

- Use English for repo-facing documentation, filenames, code comments, identifiers, and job scripts.
- Use open-weights models for probing and steering. Closed API models cannot provide residual-stream activations.
- Keep the model name in `config/config.yaml` under `model.name`; never hardcode it elsewhere.
- Parameterize the probe layer as `probing.probe_layer_frac`, default `0.66`.
- Express steering strengths relative to the mean residual-stream norm at the steered layer.
- Seed all randomness and log the seed in result metadata.
- If a paper download, dataset download, or model load fails, surface the exact error. Do not silently substitute another source or fabricate data.

## SCCKN / Cluster Rules

This project runs primarily on the Universität Konstanz SCCKN cluster.

- SCCKN uses Grid Engine / `qsub`, not SLURM / `sbatch`.
- Heavy work runs through scripts in `jobs/sge/`.
- Leave queue names, module versions, scratch paths, and GPU resource details as `# ADJUST` placeholders unless the user provides exact values.
- Put large HuggingFace/model caches on scratch or work storage via `HF_HOME`, not in the home directory.
- Avoid heavy model loading or long jobs on login nodes.
- Keep scheduler support modular so a future `jobs/slurm/` backend can be added if another compute environment becomes available.
- If SCCKN is used in a paper, include this acknowledgement:
  "The authors acknowledge support by the local computing resources through the core facility SCCKN."

## Step Logging

Maintain an append-only research log at `step_logs/STEP_LOG.md`.

- At the start of each session, recover current research state before doing anything else:
  1. Read the latest 5–10 entries of `step_logs/STEP_LOG.md` for recent decisions and findings.
  2. Read the most recent report(s) in `paper/` (see `paper/README.md`) for the current empirical state.
- Append a new entry for **every meaningful step**: a decision, a finding, an experiment run, a library or model choice, a config change, a validation result. Not for every individual tool call.
- Entry format (one entry per heading):

  ```
  ## YYYY-MM-DD · Step N — <short title>
  - **Context:** which task/session this belongs to (1 sentence)
  - **Agent:** <model-id> (omit if step was taken by a human)
  - **Did:** what was done (files read/run/changed)
  - **Findings:** concrete results — numbers, file paths, pass/fail
  - **Decision / rationale:** decision taken and why (omit if none)
  - **Next:** immediate next action (omit if none)
  ```

- `Step N` resets to 1 each calendar date; the date in the heading disambiguates.
- Never edit or delete previous entries — only append. English only.
- This file is committed to git (shared with collaborators), so keep entries concise and factual.
- `ai-usage/steps.md` was retired on 2026-06-19; all AI agent actions are recorded here.

## Findings Reports

Any meaningful new finding, result, or methodological approach must be written up as a new dated Markdown file in `paper/`:

- Naming convention: `YYYY-MM-DD_<short-slug>.md` (use the date the result was produced).
- Figures go under `paper/figures/`; update `paper/figures/generate_figures.py` and regenerate as needed.
- The corresponding `step_logs/STEP_LOG.md` entry records *that* a report was created and links to it by path; the report carries the full detail (tables, figures, caveats, interpretation).
- See `paper/README.md` for the current list of reports and the figures inventory.

## Working Conventions

- Work phase by phase according to `PLAN.md`.
- Stop and summarize after each phase when running the research pipeline.
- Do not launch long GPU jobs before confirming `config/config.yaml` is filled in.
- Pipeline outputs (direction vectors, activation matrices, validation JSON logs, metric CSVs)
  are tracked in git under `data/processed/concept_vectors*/`, `results/logs/validate_probes_*.json`,
  and `results/tables/probe_metrics*.csv`. Commit and push them via `bash jobs/sync_outputs.sh`
  (also runs automatically at the end of each extraction SGE job). Model weights are never committed.
- Figures go to `results/figures/`; tables go to `results/tables/`; logs go to `results/logs/`.
- Use `transformer-lens` for hooks by default. `nnsight` can be added later for larger models.

## Done Criteria for the Core Result

The core paper result should produce three headline outputs:

1. Probe activation vs human warmth/competence rating.
2. Steering strength vs callback-rate shift.
3. Model callback disparity vs human callback disparity.
