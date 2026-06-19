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

See `PLAN.md` for the phased implementation plan and `AGENTS.md` for working conventions.

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

- **Max logit delta = 6.375 (warmth) vs 0.344 (random, Test 1).** Steering along the estimated
  warmth direction shifts the output logits far more than steering along a random vector. Note:
  these two numbers are not directly comparable — Test 1 used alpha = 0.5 (absolute) while Test 2
  used alpha = 26.99 (0.5 × mean residual norm), a ~54× difference in injected magnitude. The
  restructured smoke tests (`smoke_tests/`) fix this by using equal-magnitude random controls in
  every test. Causal claims require the full steering experiment with the hiring-task prompt.

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

## Smoke Test Suite (`smoke_tests/`)

The pilot work above (Qwen2.5-1.5B) has been reorganised into a structured suite that
compares three model + tooling combinations on identical stimuli:

| Directory | Model | Tooling | Key addition |
|-----------|-------|---------|--------------|
| `smoke_tests/qwen_transformerlens/` | Qwen2.5-1.5B-Instruct | TransformerLens | Baseline (original pilot) |
| `smoke_tests/gemma3_transformerlens/` | **Gemma 3 12B-IT** | TransformerLens + GemmaScope 2 | SAE warmth-vs-tone decomposition |
| `smoke_tests/gemma4_nnsight/` | **Gemma 4 12B-IT** | nnsight | Exploratory; no SAE tooling yet |

All three tests use the same 100 sentences (`smoke_tests/stimuli.py`) and report
a `warmth_random_ratio` computed with **equal-magnitude** random and warmth vectors
for a fair comparison.  Results go to each test's own `results/` subdirectory.

**Model commitment and scale-up path.** The smoke tests run at 12B (fits on a single L40 48 GB
GPU on SCCKN) to enable a fair Gemma 3 vs Gemma 4 comparison. The committed model for the core
result is **Gemma 3 12B-IT** (`google/gemma-3-12b-it`) with GemmaScope 2 SAE decomposition to
address the warmth-vs-valence confound. The scale-up target — pending validation that 12B
results replicate — is **Gemma 3 27B-IT** (`google/gemma-3-27b-it`, ~54 GB bf16, runs on
scc214's RTX 6000 96 GB). This door is intentionally left open and will be entered once the
12B smoke results are satisfactory.

Run on SCCKN (GPU required for meaningful signal):

```bash
qsub jobs/sge/smoke_qwen.sh
qsub jobs/sge/smoke_gemma3.sh
qsub jobs/sge/smoke_gemma4.sh
```

See `smoke_tests/README.md` for full details and `docs/SCCKN_WINDOWS.md` for
Windows → cluster connection setup.

### Smoke Test Results (SCCKN, Gemma 3 12B-IT)

Run 2026-06-08 on `scc214` (NVIDIA RTX PRO 6000 Blackwell, 96 GB). Seed `20260527`.

| Metric | Value |
|---|---|
| Model | google/gemma-3-12b-it |
| Layer | 31 / 48 (frac = 0.66) |
| d\_model | 3840 |
| n warm / cold | 50 / 50 |
| Diff-vector norm | 1484.6 |
| Cosine(mean\_warm, mean\_cold) | 0.99975 |
| Cohen's d (in-sample) | 2.896 |
| 5-fold CV probe accuracy | **0.86 ± 0.08** |
| Fold scores | [0.95, 0.80, 0.95, 0.75, 0.85] |
| Chance baseline | 0.50 |
| Mean residual norm at layer 31 | 66184.6 |
| Steering alpha (0.5 × mean resid norm) | 33092.3 |
| Max logit delta (warmth-direction steering) | 40.0 |
| Max logit delta (equal-magnitude random) | 20.75 |
| Warmth / random ratio | **1.93×** |
| SAE CV accuracy (GemmaScope 2 layer\_31\_width\_16k\_l0\_medium) | 0.61 ± 0.07 |
| Status | **PASS** |

**Gemma 4 outcome (nnsight, 4B and 12B): zero results.** Both Gemma 4 sizes failed with the
same two errors: (1) Gemma 4 is registered as `AutoModelForImageTextToText` (multimodal) —
nnsight's `LanguageModel()` cannot load it; `VisionLanguageModel` is required but does not
resolve the processor (`Gemma4Processor`/`Gemma4UnifiedProcessor` import fails). (2)
`Gemma4Config` lacks the `num_hidden_layers` attribute that nnsight uses to map layer indices.
Root cause: nnsight 0.6 (released Feb 2026) predates Gemma 4 (Apr 2026). No results were
obtained from either Gemma 4 run. Gemma 4 is dropped from the pipeline until nnsight adds
native support.

#### Audit (Gemma 3 12B smoke)

The 0.86 CV accuracy clears the pre-specified >0.80 threshold and improves on the Qwen pilot
(0.83). The following caveats apply before treating this as a production-quality signal.

1. **Cohen's d is in-sample.** d = 2.90 is computed on the same 100 sentences used to fit the
   warmth direction — it measures separation, not generalization. The cross-validated probe
   accuracy (0.86) is the correct held-out figure.

2. **Causal signal is modest.** With an equal-magnitude random control, warmth steering
   achieves a logit delta of 40.0 vs 20.75 for random — a **1.93× ratio**, not the
   previously-reported (and since retracted) "18×." The ~2× ratio is consistent with warmth
   being one of many active directions at this layer. A meaningful causal claim requires the
   full hiring-task steering experiment, where the warmth vector is injected during a callback
   recommendation and measured against an equal-magnitude random baseline.

3. **Valence confound is unresolved.** The SAE probe (GemmaScope 2 `layer_31_width_16k_l0_medium`)
   achieves **sae\_cv\_mean = 0.61 ± 0.07** — barely above chance (0.50) and far below the
   raw-residual probe (0.86). The top warm-minus-cold SAE features are small in magnitude and
   mixed in sign. This gap has two possible interpretations: the warmth direction is spread
   across many features (no single warmth-specific feature), or the direction is dominated by
   general-valence features that the SAE correctly refuses to label as warmth-specific. Both
   are concerning. **Neuronpedia inspection of the top features is required before any
   warmth-vs-valence claim can be made.** This is the project's primary open methodological
   risk.

4. **Residual-stream geometry.** Cosine(warm, cold) = 0.99975 and diff_norm / mean_resid_norm
   ≈ 1484.6 / 66184.6 ≈ 2.2 %. Warmth is a tiny, consistent offset riding on a large shared
   activation magnitude. This geometry (typical for mid-late concept directions) justifies the
   projection-based analysis and confirms that raw cosine distance between individual sentence
   activations would be uninformative.

5. **Scope.** This run tests warmth only, on one model, at one layer, using 100 hand-written
   sentences that confound warmth with positive sentiment. Competence is untested. All of these
   limitations are carried forward to the main pipeline, which addresses them via API-generated
   topic-controlled stories (~4,800) and PCA-based tone denoising.

## Phase 4 Readiness & the Gallo–Hausladen Parallel

### What Phase 4 produces

`src/extract_vectors.py` runs the full 4,800-story concept corpus (100 topics × 4 conditions ×
12 stories, generated by `src/generate_stimuli.py`) through Gemma 3 12B-IT, captures the
residual stream at layer 31, mean-pools each story's activations, and builds **two
mean-contrast vectors** — warmth and competence — with global-mean centring and PCA-based
denoising against a neutral corpus. Both vectors are saved to `data/processed/` for downstream
probe validation (Phase 5), causal steering (Phase 6), and hiring benchmark (Phase 7).

### Readiness checklist

| Component | Status | Note |
|---|---|---|
| Method recipe (`docs/METHOD_NOTES.md` Part 1) | ✅ complete | Layer, hook, mean-pool, centring, PCA, logit-lens all specified |
| Model config (`config/config.yaml`) | ✅ committed | `google/gemma-3-12b-it`, layer frac 0.66, seed 20260527 |
| Stimulus generator (`src/generate_stimuli.py`) | ⚠️ written, not run | `ANTHROPIC_API_KEY` needed; no `data/stimuli/concept_stories.jsonl` yet |
| Extractor (`src/extract_vectors.py`) | ❌ stub | Raises `NotImplementedError` after model load |
| **Carina/Gallo benchmark data (`data/raw/`)** | ❌ not acquired | Phase 0 incomplete; repo must be cloned before Phase 5 correlation can be computed |
| Neutral corpus for PCA denoising | ❌ not chosen | METHOD_NOTES proposes Wikipedia leads; decision pending |
| `Yes`/`No` single-token check (Gemma 3 tokenizer) | ⚠️ open | Required before Phase 7 callback scoring |
| Layer formula reconciliation | ⚠️ open | `hooks.py` uses `round((n_layers-1)×frac)` → 31; METHOD_NOTES documents `round(frac×n_layers)` → 32; must be consistent |
| `start_token` decision | ⚠️ open | Config = 50 (preamble-skip for prompted text); smoke test used 1 (standalone sentences); 150-word concept stories are standalone — deliberate decision needed |

### Blocking steps (ordered)

1. **Clone Carina's data** — `git clone https://github.com/carinahausladen/SocialPerceptions-Predict-Callback data/raw/SocialPerceptions-Predict-Callback-main` (both locally and on SCCKN).
2. **Inspect Carina's analyses** — read `book/names.qmd` and `book/categories.qmd` and the callback-on-warmth-plus-competence regression tables **before** finalising Phase 5/7 design. We must replicate her test structure on the model, not invent our own.
3. **Run `generate_stimuli.py`** — needs `ANTHROPIC_API_KEY`; produces `data/stimuli/concept_stories.jsonl` (~4,800 stories). Safe to re-run; resumes if interrupted.
4. **Implement `extract_vectors.py`** — per METHOD_NOTES §1.2–1.4: batched extraction, centring, mean-contrast for **both** warmth and competence, PCA denoise, save to `data/processed/`. Check that the two vectors are reasonably orthogonal (Stereotype Content Model treats them as independent dimensions; high cosine similarity would indicate a confound).
5. **Resolve design decisions** — pick neutral corpus; fix layer-formula discrepancy; decide `start_token` for 150-word standalone stories.

### The Gallo–Hausladen parallel

Carina and colleagues tested whether **human-perceived warmth and competence** (rated 0–100 on
Prolific/MTurk for social signals such as names, race, disability, religion, gender) predict
**real-world callback rates** from meta-analyzed North American correspondence studies.

This project tests the same question at the level of **model-internal representations**. The
construct mapping is direct:

| Gallo–Hausladen | This project | Phase |
|---|---|---|
| Human warmth/competence ratings for social signals | Probe projection scores (signal → activation → project onto warmth/competence vector) | Phase 5 |
| Callback rates from correspondence studies | Model callback probabilities from hiring prompts | Phase 7 |
| — (not applicable) | Activation steering → causal shift in callback probability | Phase 6 |

The three [headline outputs](AGENTS.md) follow the same structure: (1) probe score vs human
rating, (2) steering strength vs callback shift, (3) model callback disparity vs human callback
disparity.

**Are we parallel yet? No.** The Gemma 3 12B smoke test established that a warmth direction
exists as a linear signal at layer 31 (RQ1, probe CV 0.86). It never touched Carina's social
signals or human ratings. True parallelism begins at **Phase 5**, once the warmth *and
competence* vectors are built from the full concept corpus and projected onto Carina's exact
signals for correlation. Additional gaps before a clean parallel is possible:

- **Competence vector is missing.** The SCM treats warmth and competence as two orthogonal
  dimensions. Only warmth has been tested so far. Phase 4 must produce a competence vector and
  verify warmth ⊥ competence before Phase 5 can reproduce Carina's two-factor structure.
- **Valence confound is unresolved.** SAE CV = 0.61 (barely above chance). The warmth vector
  may be tracking general positive sentiment rather than warmth specifically. Until Neuronpedia
  inspection and/or a within-topic control confirms otherwise, the warmth-vs-valence distinction
  — central to the SCM — is not confirmed in our representation.
- **Carina's data is not downloaded.** No correlation can be computed.

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

**Smoke tests complete. Committed model: Gemma 3 12B-IT.**

The Qwen pilot (local GPU) confirmed the machinery works. The SCCKN smoke test on
`google/gemma-3-12b-it` confirmed the signal scales to a research-grade model: probe CV
accuracy 0.86 (chance 0.50), Cohen's d 2.90, warmth_random_ratio 1.93× (equal-magnitude
control). Gemma 4 was attempted via nnsight but produced zero results (see Smoke Test Results
below); it is dropped for now. The committed model for the core result is **Gemma 3 12B-IT**
with GemmaScope 2 SAE decomposition. Scale-up to Gemma 3 27B-IT on scc214 (96 GB) remains
open pending 12B result replication.

**Next step:** (1) Inspect top SAE features on Neuronpedia to close the valence-confound
question. (2) Phase 4 — implement `src/extract_vectors.py` to run the extraction loop over
the full API-generated stimulus corpus (~4,800 stories), build per-concept mean-contrast
vectors, and save activations to `data/processed/`.

## Caveats

- Functional warmth/competence representations do not imply model subjective experience. This project studies representations and behavior.
- Results are model-specific. Cross-model generalization is a stretch goal.

## References

- Sofroniew, Kauvar, Saunders, Chen, et al. (2026). _Emotion Concepts and their Function in a Large Language Model._ arXiv:2604.07729.
- Gallo, Hausladen, Hsu, Jenkins, Ona, Camerer (2024). _Perceived warmth and competence predict callback rates in meta-analyzed North American labor market experiments._ PLOS ONE 19(7): e0304723. doi:10.1371/journal.pone.0304723.
