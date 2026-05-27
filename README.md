# Probing Warmth & Competence Representations in LLM Hiring Decisions

Do large language models encode **warmth** and **competence**, the two dimensions of the Stereotype Content Model, as internal, linearly probeable representations? If so, do those representations causally shape who the model recommends for a callback?

This project adapts an interpretability pipeline originally built for emotion concepts and points it at hiring discrimination. The model behavior is then benchmarked against human correspondence-study data.

## Background

Two pieces of prior work motivate this project:

- **Sofroniew, Lindsey et al. (2026), _Emotion Concepts and their Function in a Large Language Model_.** This work shows that emotion concepts can exist inside a language model as linear vectors and that those vectors can causally influence behavior. This project adapts that method from emotion concepts to warmth and competence.
- **Gallo, Hausladen et al. (2024), _Perceived warmth and competence predict callback rates in meta-analyzed North American labor market experiments_.** This work meta-analyzes correspondence studies and links perceived warmth and competence to callback disparities. This project uses it as the human benchmark.

The bridge is simple: if emotions can be represented as causal internal directions, warmth and competence may be represented that way too, and they may help explain hiring bias in model outputs.

## Research Questions

1. **Existence:** Can we extract linear warmth and competence vectors from an open-weights model?
2. **Alignment:** Do those probes track human warmth/competence ratings of the same social signals?
3. **Causality:** Does steering the vectors shift model callback recommendations?
4. **Benchmark:** Do the model's callback disparities reproduce documented human hiring bias?

## Method Overview

1. Generate synthetic stories exhibiting high/low warmth and high/low competence.
2. Extract residual-stream activations and build concept vectors.
3. Validate probes against held-out text and human warmth/competence ratings.
4. Run steering experiments on a callback recommendation task.
5. Benchmark model callback disparities against the PLOS ONE human data.

See `PLAN.md` for the phased implementation plan and `CLAUDE.md` for working conventions.

## Requirements

- An open-weights model with residual-stream access.
- A GPU capable of running the selected model.
- Python dependencies from `requirements.txt`.
- For SCCKN: Grid Engine / `qsub` access and cluster-specific values filled into `jobs/sge/*.sh`.

## Repository Layout

```text
config/          Project configuration. Model selection lives in config/config.yaml.
data/raw/        Downloaded source datasets. Ignored by git except .gitkeep.
data/stimuli/    Generated concept stories and hiring prompts.
data/processed/  Activations, vectors, and derived arrays. Ignored by git.
docs/            Method and compute notes.
jobs/sge/        SCCKN Grid Engine job wrappers.
papers/          Downloaded source PDFs. Ignored by git except .gitkeep.
results/         Figures, tables, and logs.
src/             Python package and experiment entrypoints.
tests/           Lightweight structural tests for future development.
archive/         Previous projects and SCCKN notes.
```

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Choose a model only after compute is confirmed:

```yaml
model:
  name: "REPLACE_ME"
```

The model name must be set in `config/config.yaml`; scripts should not hardcode it.

## Data and Secrets

Local `.env` files, credentials, downloaded papers, raw datasets, model caches, activations, and cluster logs are ignored by git. Do not commit secrets or local SCCKN paths.

The active repo tracks structure and source code. Downloaded papers and benchmark data should be fetched locally or on SCCKN when needed.

## SCCKN

This project targets SCCKN Grid Engine by default:

```bash
qsub jobs/sge/<job>.sh
qstat -u emrecan.ulu
qdel <job_id>
qacct -j <job_id>
```

The job scripts contain `# ADJUST` placeholders for queue names, module versions, GPU resources, and scratch paths. Fill those in on the cluster before submitting heavy jobs.

## Status

Repository setup phase. Source papers and benchmark data are downloaded before experimental code is developed.

## Caveats

- Functional warmth/competence representations do not imply model subjective experience. This project studies representations and behavior.
- Results are model-specific. Cross-model generalization is a stretch goal.

## References

- Sofroniew, Kauvar, Saunders, Chen, et al. (2026). _Emotion Concepts and their Function in a Large Language Model._ arXiv:2604.07729.
- Gallo, Hausladen, Hsu, Jenkins, Ona, Camerer (2024). _Perceived warmth and competence predict callback rates in meta-analyzed North American labor market experiments._ PLOS ONE 19(7): e0304723. doi:10.1371/journal.pone.0304723.
