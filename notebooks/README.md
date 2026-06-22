# notebooks/ — interactive H100 notebooks

These run on the JupyterHub research space (H100), not in the lightweight sandbox. Pick the
**Full GPU (80 GB)** server option: Gemma-3-12B needs ~24 GB, but the headroom covers steering
activations and lets the same notebook load Gemma-3-27B (~54 GB).

Start the kernel in the repository root so `import src...` works.

| Notebook | Phase | What it does | Needs |
|----------|-------|--------------|-------|
| `06_hiring_steering_causality.ipynb` | 6 | Steers warmth/competence at the probe layer and measures the change in the hiring callback margin. The core causal result. | `data/processed/concept_vectors/` (Phase 4 vectors), transformer_lens, Gemma access |
| `07_hiring_audit.ipynb` | 7 | Probe-vs-human-rating validation (name-level Figure 1), baseline callbacks, and a flagged scaffold for the model-vs-human disparity. | same |

Both read the model name and probe layer from each run's `meta.json`, so set
`VECTORS_SUBDIR` to `concept_vectors` (12B) or `concept_vectors_gemma3_27b` (27B) and rerun.

Method note: the steering hook and the Yes/No callback margin mirror `src/gemma_scope_causality.py`
exactly, so the hiring numbers are directly comparable to Emre's concept-steering result.
