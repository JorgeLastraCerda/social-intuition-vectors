# Gemma 4 26B-A4B Neutral Extraction

- **Produced:** 2026-07-18 16:23 Europe/Berlin
- **Model:** `google/gemma-4-26B-A4B-it`
- **Scope:** Neutral-corpus residual extraction for PCA denoising
- **Status:** Complete; PCA input accepted

## Artifacts

- **Scripts:** `src/extract_neutral.py`, `src/validate_gemma4_remaining.py`, `jobs/sge/gemma4_remaining_run.sh`
- **Inputs:** `config/gemma4_26b_a4b.yaml`, `data/stimuli/neutral_corpus.jsonl`
- **Outputs:** `data/processed/concept_vectors_gemma4_26b_a4b/X_neutral.npy`, `data/processed/concept_vectors_gemma4_26b_a4b/neutral_meta.json`, `results/logs/gemma4_26b_a4b_neutral_20260718T141249Z_26b_a4b_neutral.{out,err}`

## Results

Job `1145339` completed on one NVIDIA RTX PRO 6000 Blackwell Server Edition with `failed=0`, `exit_status=0`, and 189 seconds wallclock. It produced a finite `1500 × 2816` residual matrix at `blocks.19.hook_resid_post` from the seeded neutral corpus.

The requested and resolved model revisions match exactly. Peak allocated and reserved VRAM were 49.04 GiB and 51.54 GiB, respectively. The extraction validator accepted the matrix dimensions, metadata, seed, layer, model identity, and revision provenance.

## Decision

The neutral matrix is accepted as the input to the independently submitted CPU PCA job `1145368`. No denoised steering job will be submitted until that PCA artifact passes its own validator.
