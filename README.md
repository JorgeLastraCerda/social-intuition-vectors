# Probing Warmth & Competence Representations in LLM Hiring Decisions

**Emrecan Ulu** and **Jorge Lastra Cerda**, University of Konstanz
`emrecan.ulu@uni-konstanz.de` · `jorge.lastra-cerda@uni-konstanz.de`

This manuscript has not been peer reviewed. The active source lives at
[`paper/paper/Ulu_Lastra.tex`](paper/paper/Ulu_Lastra.tex) (compiled PDF:
`paper/paper/Ulu_Lastra.pdf`).

## Abstract

Recent interpretability work shows that emotions are encoded as linear directions
in a language model's residual stream, and that steering activations along these
directions can alter model behavior. Adapting this emotion-vector method to how a
model socially perceives a person, we extract residual-stream directions for
warmth and competence, the two axes of the Stereotype Content Model, and examine
how these social-intuition vectors relate to callback recommendations across nine
open-weight models from the Gemma, Qwen, and Llama families. Warmth and competence
are linearly encoded with large effect sizes in every model and shift
concept-level judgments under steering, although steering increases callback
recommendations in some models and decreases them in others. Internal
representations and hiring outcomes show different patterns: both Gemma-3 models
track human warmth ratings, whereas the Llama family and earlier Qwen models
invert them, and this alignment varies across model generations. In hiring
outcomes, seven checkpoints favor names associated with Black applicants and
eight favor names associated with female applicants. The pooled human race gap
is near zero, so the model race results are not evidence of a consistent
reversal; the human gender benchmark favors male applicants, and Gemma-3-27B is
the only checkpoint that reproduces that direction. We
conclude that linear encoding and successful steering do not imply that these
representations reliably drive hiring decisions.

## Background

Four pieces of prior work position this project:

- **Sofroniew, Lindsey et al. (2026), _Emotion Concepts and their Function in a
  Large Language Model_.** Shows that emotion concepts exist inside a language
  model as linear directions, and that those directions causally influence
  behavior. This project adapts that method from emotion concepts to warmth and
  competence.
- **Gallo, Hausladen et al. (2024), _Perceived warmth and competence predict
  callback rates in meta-analyzed North American labor market experiments_.**
  Meta-analyzes correspondence studies and links perceived warmth and competence
  to callback disparities. This project uses it as the human benchmark.
- **Deas and McKeown (2025), _Artificial Impressions_.** Probes warmth and
  competence in open-weight model activations and connects those impressions
  to response properties. This is the nearest same-construct probe study.
- **Tripathy and Buckmann (2026), _Fair Outputs, Biased Internals_.** Uses
  activation steering to test latent demographic representations in mortgage
  underwriting. This is the nearest high-stakes representational audit.

The bridge: if emotions can be represented as causal internal directions, warmth
and competence may be represented that way too, and they may help explain hiring
bias in model outputs.

## Research Questions

1. **Existence:** Are warmth and competence extractable as linear directions from
   open-weights models?
2. **Alignment:** Do those directions track human warmth/competence ratings of the
   same social signals?
3. **Causality:** Does steering the vectors shift model callback recommendations?
4. **Benchmark:** Do the models' callback disparities reproduce documented human
   hiring bias?

## Models Studied

Nine open-weight checkpoints across three families, all probed and steered on the
same pipeline:

| Family | Checkpoint | Layers |
|---|---|---|
| Gemma-3 | `google/gemma-3-12b-it` | 48 |
| Gemma-3 | `google/gemma-3-27b-it` | 62 |
| Gemma-4 | Gemma-4-12B | 40 |
| Gemma-4 | Gemma-4-26B-A4B (MoE) | 48 |
| Gemma-4 | Gemma-4-31B | 30 |
| Qwen3 | Qwen3-14B | 40 |
| Qwen3.6 | Qwen3.6-27B | 64 |
| Qwen3.6 | Qwen3.6-35B-A3B (MoE) | 48 (approx.) |
| Llama-3.1 | Llama-3.1-8B-Instruct | 32 |

`google/gemma-3-12b-it` is the committed baseline in `config/config.yaml`
(`model.name`); the other eight checkpoints each have a dedicated config under
`config/` and are run through the same pipeline. The probe layer is fixed as a
fraction of model depth, `probing.probe_layer_frac = 0.66` by default.

## Pipeline

1. **Generate synthetic stories** exhibiting high/low warmth and high/low
   competence, topic-controlled — `src/generate_stimuli.py`.
2. **Extract residual-stream activations and build concept vectors** — mean
   contrast between conditions, PCA-denoised against a neutral corpus —
   `src/extract_vectors.py`, `src/denoise_vectors.py`.
3. **Validate probes** against held-out text and human warmth/competence ratings.
4. **Steer the vectors** at the concept level and on the hiring callback task,
   with strengths expressed relative to the mean residual-stream norm at the
   steered layer — `src/dense_steering.py`, `src/steering.py`,
   `src/hiring_steering.py`.
5. **Benchmark** model callback disparities against the Gallo–Hausladen human
   data, including bootstrap mediation — `src/hiring_disparity.py`,
   `src/hiring_r4.py`, `src/hiring_audit.py`.

All randomness is seeded and the seed is logged with each result. See `PLAN.md`
for the original phased plan and `AGENTS.md` for working conventions.

## Key Findings

- **Warmth and competence are linearly encoded with large effect sizes in all
  nine models**, geometrically stable across depth, and causally functional at
  the concept level; the directions are shared across architecturally distinct
  models.
- **Steering shifts hiring callbacks, but not consistently in one direction.**
  At Gemma-3-12B, warmth steering raises callback inclination roughly linearly;
  at Gemma-3-27B the same intervention is non-monotone and reverses sign at
  higher steering strength rather than saturating.
- **Encoding strength and hiring impact dissociate.** Gemma-3-12B has the
  largest normalized warmth steerability of the nine models yet shows no
  significant hiring mediation; Llama-3.1-8B is among the least steerable yet
  has the strongest indirect effect from name group to callback margin.
- **Human alignment varies by model generation.** Both Gemma-3 models track
  human warmth ratings; Llama-3.1-8B and Qwen3-14B invert them.
- **Race and gender gaps require different interpretations.** On the shared
  pooled within-group SD scale, seven models favor Black-signaling names, but
  the human race reference is only $d=+0.15$ and is better treated as near zero
  than as a stable directional penalty. Eight models favor female-signaling
  names against a human benchmark of $d=-0.47$, which favors male-signaling
  names; Gemma-3-27B is the only checkpoint with the human gender direction.
- **Bottom line:** linear encoding and successful steering at the concept level
  do not imply that these representations reliably drive hiring decisions.

Full detail, tables, and figures live in the dated findings reports under
[`paper/`](paper/README.md) and in the manuscript
[`paper/paper/Ulu_Lastra.tex`](paper/paper/Ulu_Lastra.tex). Key open
limitations (see the manuscript's Limitations section): the stimuli are
LLM-generated rather than drawn from real listings; probe-to-human correlations
are moderate rather than strong for several models; callback margins are
affected by a bf16 quantisation artifact that is flagged but not fully
resolved; and the mediation tests are not corrected for multiple comparisons
across all nine models.

## Repository Layout

```text
config/          Project configuration. Baseline model in config/config.yaml
                 (model.name); one YAML per additional checkpoint.
src/             Pipeline package: stimulus generation, vector extraction,
                 denoising, steering, hiring evaluation/steering/disparity/
                 mediation, per-family (Gemma Scope, Qwen3.6) helpers.
data/raw/        Vendored source data, including the Gallo–Hausladen
                 replication package. Tracked in git: no download needed.
data/stimuli/    Generated concept stories, neutral corpus, name roster.
data/processed/  Concept vectors and derived arrays per model, tracked in git
                 under data/processed/concept_vectors*/.
docs/            Method, compute, and cluster notes.
jobs/sge/        SCCKN Grid Engine job wrappers; jobs/sync_outputs.sh syncs
                 pipeline outputs back to git.
results/         Figures, tables, and logs from pipeline runs.
paper/           Dated findings reports (one per result/decision), figures,
                 idea notes, and the active manuscript in paper/paper/.
step_logs/       Append-only research log, step_logs/STEP_LOG.md.
notebooks/       Exploratory notebooks for hiring steering, audit, disparity.
literature/      Empty placeholder. The two source papers are cited by DOI
                 under References rather than redistributed here.
tests/           Pytest suite covering config, steering calibration, and
                 per-model pipeline stages.
scckn/           SCCKN cluster operational docs and job templates.
presentation/    Slide deck and presentation plan.
```

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Family-specific extras as needed:
pip install -r requirements-gemma4.txt
pip install -r requirements-qwen36.txt
```

The committed baseline model is set in `config/config.yaml`:

```yaml
model:
  name: "google/gemma-3-12b-it"
```

Runs against the other eight checkpoints use the matching config under
`config/` (e.g. `config/qwen36_35b_a3b.yaml`); scripts do not hardcode a model
name.

## Reproducing the Paper

Every table and figure in the manuscript rebuilds from files tracked in this
repository. **No GPU and no cluster access are required for this step**, because
the activations, probe directions, and per-model result files are all committed.
GPUs are needed only to re-extract activations from scratch.

```bash
git clone https://github.com/JorgeLastraCerda/social-intuition-vectors
cd social-intuition-vectors
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Regenerate the LaTeX tables the manuscript reads
python src/build_paper_probe_tables.py
python src/build_paper_mediation_table.py

# Confirm the regenerated tables match what is committed
git diff --stat results/tables/     # expect no output

# Build the manuscript (two passes resolve references)
cd paper/paper && pdflatex Ulu_Lastra && pdflatex Ulu_Lastra
```

The result is a 34-page PDF. `src/build_paper_mediation_table.py` reproduces
`results/tables/mediation_9model.tex` byte for byte from the stored bootstrap
logs (seed 20260527, 36 mediation tests, 14 with unadjusted 95% intervals
excluding zero), so the mediation numbers in the paper can be checked directly
against the artifacts.

Run the test suite with `pytest tests/`. Two tests in `tests/test_hiring_r4.py`
are known to fail against the current pandas version; every other test passes.

## Reading the Evidence

The paper is supported by dated findings reports under `paper/`, one per result
or methodological decision. Each report opens with an `## Artifacts` block
listing the exact scripts, inputs, outputs, and figures behind it, so any claim
in the manuscript can be traced to the files that produced it.
`paper/README.md` indexes all of them; `step_logs/STEP_LOG.md` is the
append-only chronological record of how the project got there.

## AI Assistance

AI coding assistants were used in this project to help write and refactor
pipeline code, generate LaTeX tables and figures, and draft and edit prose.

The authors designed the study, chose the methods, ran the experiments, and
verified every reported number against the stored result artifacts. All
scientific claims, interpretations, and errors are the authors' own
responsibility.

## Data and Secrets

Local `.env` files, credentials, reference PDFs, model caches, and cluster
logs are ignored by git. Raw benchmark data under `data/raw/` is an exception:
it is vendored so the analysis reproduces without any download. Do not commit
secrets or local SCCKN paths. Concept vectors, validation logs, and metric
tables produced by the pipeline are tracked in git and synced with
`bash jobs/sync_outputs.sh`; model weights are never committed.

## SCCKN

This project runs primarily on the Universität Konstanz SCCKN cluster
(Grid Engine, not SLURM):

```bash
qsub jobs/sge/<job>.sh
qstat -u emrecan.ulu
qdel <job_id>
qacct -j <job_id>
```

Job scripts contain `# ADJUST` placeholders for queue names, module versions,
GPU resources, and scratch paths; fill those in on the cluster before
submitting heavy jobs. HuggingFace/model caches are kept on scratch or work
storage via `HF_HOME`, not in the home directory.

The authors acknowledge support by the local computing resources through the
core facility SCCKN.

## Caveats

- Functional warmth/competence representations do not imply model subjective
  experience. This project studies representations and behavior.
- Results are model- and generation-specific; the extent of cross-model
  generalization is itself one of the paper's findings, not an assumption.

## References

- Sofroniew, Kauvar, Saunders, Chen, et al. (2026). _Emotion Concepts and their
  Function in a Large Language Model._ arXiv:2604.07729.
  <https://arxiv.org/abs/2604.07729>
- Gallo, Hausladen, Hsu, Jenkins, Ona, Camerer (2024). _Perceived warmth and
  competence predict callback rates in meta-analyzed North American labor
  market experiments._ PLOS ONE 19(7): e0304723.
  doi:10.1371/journal.pone.0304723.
  <https://doi.org/10.1371/journal.pone.0304723>
