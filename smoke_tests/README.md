# Smoke Tests

This directory contains three self-contained smoke tests that validate the warmth
probing pipeline on different model + tooling combinations.  Each test uses the
same 100 sentences (`stimuli.py`) so results are directly comparable.

---

## Directory Layout

```
smoke_tests/
  stimuli.py                 # shared 100 sentences (50 warm + 50 cold) — do not duplicate
  qwen_transformerlens/      # Qwen2.5-1.5B + TransformerLens (original pilot)
    smoke_test_probe.py
    results/                 # JSON outputs (git-ignored)
  gemma3_transformerlens/    # Gemma 3 + TransformerLens + GemmaScope 2
    smoke_test_probe.py      # probe + steering + equal-magnitude random control
    sae_decompose.py         # warmth-vs-tone SAE decomposition (run after probe)
    results/                 # JSON outputs + saved vectors (git-ignored)
  gemma4_nnsight/            # Gemma 4 + nnsight (exploratory)
    smoke_test_probe.py
    results/                 # JSON outputs (git-ignored)
```

---

## What Each Test Measures

All three tests report the same core metrics:

| Metric | What it checks |
|--------|----------------|
| `diff_norm` | Warmth direction has non-zero magnitude |
| `cosine_mean_warm_cold` | Mean warm/cold activations are not identical |
| `cohens_d` | In-sample separation along the warmth direction |
| `probe_cv_mean` | 5-fold CV logistic regression accuracy (assert > 0.80) |
| `max_logit_delta_warmth` | Steering along warmth direction shifts output |
| `max_logit_delta_random` | Same for a random vector **of equal magnitude** |
| `warmth_random_ratio` | Warmth steering / random steering (fair comparison) |

The `warmth_random_ratio` field replaces the misleading "18×" claim in the
original README (where warmth and random vectors had ~54× different magnitudes).

---

## GemmaScope 2 SAE Decomposition (`gemma3_transformerlens/sae_decompose.py`)

This additional step addresses the primary scientific risk: warm sentences are
also more positive in tone, so the warmth direction might encode general
sentiment rather than warmth specifically.

GemmaScope 2 (Dec 2025) provides pretrained Sparse Autoencoders (SAEs) for
every Gemma 3 layer.  Each SAE maps a residual-stream activation to a small set
of human-interpretable features.  The decomposition encodes our warmth vector
through the SAE and reports which features activate most strongly.

- Features labelled "warmth", "care", "friendliness" → warmth-specific ✓
- Features labelled "positivity", "joy", "good" → valence confound ✗

Look up feature labels on [Neuronpedia](https://www.neuronpedia.org/).

---

## Running on SCCKN

SGE job scripts are in `jobs/sge/`:

```bash
# From the cluster frontend (after git clone + conda env setup):
qsub jobs/sge/smoke_qwen.sh       # Qwen baseline
qsub jobs/sge/smoke_gemma3.sh     # Gemma 3 + SAE
qsub jobs/sge/smoke_gemma4.sh     # Gemma 4 + nnsight

qstat -u emrecan.ulu              # monitor
```

All `# ADJUST` placeholders in the job scripts must be filled in before submitting
(queue name, GPU resource flag, VRAM).  See `docs/SCCKN_WINDOWS.md`.

---

## Running Locally (small models only)

For a quick syntax check on CPU (no meaningful probe signal):

```bash
python smoke_tests/qwen_transformerlens/smoke_test_probe.py \
    --model gpt2 --start-token 1
```

---

## Output Files

Each test writes to its own `results/` subdirectory:

| File | Content |
|------|---------|
| `smoke_probe_<timestamp>.json` | Full metrics for one run |
| `warmth_vector.npy` | Warmth direction vector (Gemma 3 only, for SAE step) |
| `X_warm.npy`, `X_cold.npy` | Per-sentence activation matrices (Gemma 3 only) |
| `sae_decompose_<timestamp>.json` | SAE feature report (Gemma 3 only) |

`results/` directories are git-ignored (large arrays).  JSON logs are small and
can optionally be committed for record-keeping.

---

## Conda Environments

| Environment | Packages | Used for |
|-------------|----------|----------|
| `wc-tl` | transformer-lens, sae-lens, torch | Qwen + Gemma 3 tests |
| `wc-nn` | nnsight, nnterp, torch | Gemma 4 test |

See `docs/SCCKN_WINDOWS.md` for setup commands.
