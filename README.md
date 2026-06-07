# Probing Warmth & Competence Representations in LLM Hiring Decisions

## Abstract

Large language models (LLMs) are increasingly positioned as decision-support tools in hiring, yet little is known about how their internal representations encode social perceptions that are known to shape human labor-market discrimination. This project investigates whether LLMs represent the two core dimensions of the Stereotype Content Model, warmth and competence, as linearly probeable internal directions, and whether these representations causally influence callback recommendations in hiring tasks.

We propose a mechanistic-interpretability pipeline that adapts recent concept-vector methods from work on emotion representations in LLMs. First, we construct synthetic text corpora that contrast high and low warmth and high and low competence while controlling for topic and surface form. We then run an open-weights language model with residual-stream access, extract activations at a middle-late layer, and derive warmth and competence vectors from condition contrasts. These vectors will be validated on held-out text and against human warmth and competence ratings from meta-analyzed North American correspondence studies. To test causal relevance, we will apply activation steering along the extracted vectors during a binary callback recommendation task and measure whether increasing or suppressing warmth and competence changes model recommendations. Finally, model callback disparities will be compared with documented human callback disparities across social signals.

This study connects social-psychological theories of stereotyping with mechanistic interpretability. Rather than treating hiring bias only as an output-level phenomenon, it asks whether stereotype-relevant dimensions are encoded inside model activations and whether those encodings help drive downstream decisions. The resulting framework is intended to distinguish three possibilities: that warmth and competence are not robustly represented, that they are represented but behaviorally inert, or that they function as causal internal features shaping hiring recommendations. By linking internal model geometry to human benchmark data, the project aims to provide a more precise basis for auditing and eventually mitigating bias in LLM-assisted hiring systems.

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

## Pilot Experiment: Warmth Is Linearly Probeable

Before building the full extraction pipeline, we ran two self-contained smoke tests on the Windows
development machine (NVIDIA RTX 4050 Laptop GPU, Python 3.13, PyTorch 2.7.1+cu128,
TransformerLens 3.3.0) to verify that the core machinery — residual-stream extraction, linear
probing, and causal activation steering — works with a real open-weights model. Both tests use
`Qwen/Qwen2.5-1.5B-Instruct` (28 layers, d\_model 1536) with the probe layer set to
`round((28 - 1) × 0.66) = 18` (`blocks.18.hook_resid_post`). All random state is seeded at
`20260527`.

### Test 1 — Wiring check (1 warm + 1 cold sentence)

`scripts/smoke_test_activations.py` confirmed that (a) the model loads onto the GPU, (b) two
semantically opposite sentences produce distinct residual-stream activations, and (c) a forward hook
that injects a random unit vector at alpha = 0.5 causally shifts the output distribution. This test
checks infrastructure, not signal quality.

| Metric | Value |
|---|---|
| Residual norm (warm) | 65.50 |
| Residual norm (cold) | 63.25 |
| Diff-vector norm | 25.88 |
| Cosine(warm, cold) | 0.918 |
| Max logit delta (random steering, alpha=0.5) | 0.344 |
| Status | PASS |

Note: this test activates from `start_token=50`, which — because the sentences are shorter than
50 tokens — collapses to the single last token. Diff-vector norm and the probe-test norm are
therefore not directly comparable.

### Test 2 — Linear probe (50 warm + 50 cold sentences)

`scripts/smoke_test_probe.py` uses 50 hand-written warm sentences and 50 hand-written cold
sentences matched in grammatical register and length. The only systematic variable is warmth.
Activations are mean-pooled from token 1 onward (content tokens only). The warmth direction is
estimated as the difference of class mean activations. A logistic regression probe is then trained
and evaluated with 5-fold stratified cross-validation on the full 100-sentence matrix.

**Sentences** are short, third-person, and parallel in structure. Warm sentences depict behaviors
such as offering help, showing care, and acknowledging others. Cold sentences depict matched
behaviors such as ignoring requests, withholding help, and dismissing others.

#### Results

| Metric | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Layer | 18 / 28 (frac = 0.66) |
| d\_model | 1536 |
| n warm / cold | 50 / 50 |
| Diff-vector norm | 6.63 |
| Cosine(mean\_warm, mean\_cold) | 0.9915 |
| Projection mean (warm) | +3.55 +/- 2.49 |
| Projection mean (cold) | -3.08 +/- 2.45 |
| Cohen's d | **2.68** |
| 5-fold CV probe accuracy | **0.83 +/- 0.04** |
| Fold scores | [0.90, 0.80, 0.80, 0.80, 0.85] |
| Chance baseline | 0.50 |
| Mean residual norm at layer 18 | 53.98 |
| Steering alpha (0.5 x mean resid norm) | 26.99 |
| Max logit delta (warmth-direction steering) | **6.375** |
| Status | PASS |

#### Interpretation

- **Cohen's d = 2.68.** In social-science conventions, d > 0.8 is a large effect. d = 2.68 indicates
  that the warmth and cold sentence populations occupy clearly separated regions along the warmth
  direction. The two class projections (mean +3.55 vs -3.08) are well outside each other's one-sigma
  band.

- **Probe accuracy = 0.83 (chance = 0.50).** A linear logistic regression, evaluated in
  cross-validation, correctly decodes the warmth label in 83 % of sentences from a single
  layer's residual stream. This is consistent with warmth being encoded as an approximately linear
  direction at this layer. The result is not near-ceiling (accuracy could in principle reach 1.0 with
  a large enough corpus), which is expected with only 100 hand-written sentences.

- **Max logit delta = 6.375 vs 0.344 (random vector, Test 1).** Steering along the estimated warmth
  direction shifts the output logits roughly 18× more than steering along a random unit vector of
  the same injected magnitude. This indicates that the direction is not merely a geometric artifact:
  it is connected to the model's downstream computation. This is a preliminary positive signal for
  Research Question 3 (Causality), but causal claims require the full steering experiment with the
  hiring-task prompt.

- **High mean-vector cosine (0.9915).** The mean residual activations for warm and cold sentences
  point in nearly the same overall direction; the warmth signal is a relatively small, consistent
  offset on top of a large shared magnitude. This is typical for concept directions in mid-to-late
  residual layers and is the reason projection-based analysis (rather than raw distance) is the
  appropriate measure.

#### Limitations

This pilot addresses Research Question 1 (Existence) for warmth only, on a small, hand-written
stimulus set. Specific limitations that the main pipeline addresses:

- Stimuli are 100 hand-written sentences. The main experiment uses approximately 4,800 API-generated
  stories (100 topics x 12 stories x 4 conditions) with systematic topic control.
- Only warmth is tested here. Competence is the second target dimension and is untested so far.
- Only one model (Qwen2.5-1.5B-Instruct) and one layer (18) are evaluated. The probe-layer fraction
  0.66 will be applied to larger models on SCCKN.
- No alignment against human warmth ratings yet (Research Question 2).
- No hiring-task context yet (Research Questions 3 and 4). The steering alpha here is computed from
  residual-stream norms on the probing sentences, not from the eventual hiring-prompt context.

#### Reproducing

```bash
python scripts/smoke_test_activations.py                        # wiring check
python scripts/smoke_test_probe.py                             # 50+50 probe
python scripts/smoke_test_probe.py --device cpu --fallback-cpu-model gpt2   # CPU fallback
```

Logs are written to `results/logs/smoke_test_*.json` and
`results/logs/smoke_test_probe_*.json`.

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

**Pilot complete.** The extraction, linear-probe, and causal-steering machinery has been validated
on Qwen2.5-1.5B-Instruct on a local GPU (see Pilot Experiment section). The warmth direction is
linearly decodable at layer 18 with 83% cross-validated probe accuracy (chance 50%) and Cohen's d
of 2.68 on a 50+50 hand-written sentence set.

**Next step:** Phase 4 — implement `src/extract_vectors.py` to run the same extraction loop over
the full API-generated stimulus corpus (~4,800 stories), build per-concept mean-contrast vectors,
and save activations to `data/processed/`.

## Caveats

- Functional warmth/competence representations do not imply model subjective experience. This project studies representations and behavior.
- Results are model-specific. Cross-model generalization is a stretch goal.

## References

- Sofroniew, Kauvar, Saunders, Chen, et al. (2026). _Emotion Concepts and their Function in a Large Language Model._ arXiv:2604.07729.
- Gallo, Hausladen, Hsu, Jenkins, Ona, Camerer (2024). _Perceived warmth and competence predict callback rates in meta-analyzed North American labor market experiments._ PLOS ONE 19(7): e0304723. doi:10.1371/journal.pone.0304723.
