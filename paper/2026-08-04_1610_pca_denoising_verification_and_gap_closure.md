# PCA Denoising: Seven-Model Verification, Provenance, and Closing the Two-Model Gap

- **Produced:** 2026-08-04 16:10 Europe/Berlin
- **Model:** Nine Gemma, Qwen, and Llama checkpoints (seven verified; two jobs queued)
- **Scope:** Independently verify the correctness of the existing seven-model
  neutral-corpus PCA denoising outputs, trace why Llama-3.1-8B and Qwen3-14B
  were missing, and queue SGE jobs on SCCKN to close the gap
- **Status:** In progress — verification complete for seven models; SGE jobs
  `1204966` (Llama-3.1-8B) and `1204967` (Qwen3-14B) submitted and queued,
  not yet complete as of this report

## Artifacts

- **Scripts:** `src/extract_neutral.py`, `src/denoise_vectors.py` (both
  pre-existing, no new code); `jobs/sge/extract_neutral_llama31_8b.sh`,
  `jobs/sge/extract_neutral_qwen3_14b.sh` (new, mirror the existing
  `jobs/sge/extract_llama31_8b.sh` / `extract_qwen3_14b.sh` and
  `jobs/sge/gemma4_neutral.sh` conventions)
- **Inputs:** `data/processed/concept_vectors*/{X_neutral,warmth_vec,
  competence_vec,X_high_warmth,X_low_warmth,X_high_competence,
  X_low_competence}.npy`, `neutral_meta.json`, `denoise_summary.json` for the
  seven already-complete models; `results/logs/validate_probes_*.json`
  (independent cross-check source)
- **Outputs (pending):** `data/processed/concept_vectors_{llama31_8b,
  qwen3_14b}/{X_neutral.npy,concept_vectors_denoised.npz,
  denoise_summary.json,neutral_meta.json}`, to be produced by the queued
  jobs and synced back via `jobs/sync_outputs.sh`
- **Figures:** none

## Why this report exists

While rewriting the "Neutral-corpus PCA denoising" bullet in
`paper/paper/Ulu_Lastra.tex` for plain-language clarity, the user asked
whether this check covered all nine models, and separately raised doubt
about whether the existing seven models' output files were actually
correct. Both questions needed a real answer, not an assumption: the first
because it turned out two models are missing entirely, the second because
this session had already found and closed several coverage gaps in other
checks (split-half cosine, cross-axis zero-shot, cross-model Spearman
agreement) and the user wanted the existing seven models actively
distrusted and checked, not taken on faith.

## Part 1: Verifying the seven existing models

Three independent cross-checks were run for each of Gemma-3-12B,
Gemma-3-27B, Gemma-4-12B, Gemma-4-26B-A4B, Gemma-4-31B, Qwen3.6-27B, and
Qwen3.6-35B-A3B:

1. **Metadata consistency.** `neutral_meta.json`'s `probe_layer` and
   `start_token` were compared against the same model's own `meta.json`
   (the file written during the original concept-story extraction). All
   seven matched exactly, confirming the neutral corpus was extracted at
   the same layer and token offset as the concept vectors it is meant to
   denoise.
2. **Cross-source cosine agreement.** `denoise_summary.json`'s
   `cosine_before` (the warmth-competence cosine computed inside
   `denoise_vectors.py`, before any denoising) was compared against
   `axis_cosine` in `results/logs/validate_probes_*.json` (computed by a
   completely separate script, `src/validate_probes.py`, on a different
   run). All seven matched to 6 decimal places, e.g. Gemma-3-12B: 0.748953
   vs. 0.7489530444145203.
3. **Array integrity and independent recomputation.** Loaded every
   `X_neutral.npy`, `warmth_vec.npy`, and `concept_vectors_denoised.npz`
   directly: all shapes matched each model's own `d_model`, `X_neutral` had
   1,500 rows (the full neutral corpus) in every case, and all values were
   finite. For Gemma-3-12B, independently re-ran the exact PCA and
   project-out procedure from `src/denoise_vectors.py` from scratch and
   compared the resulting `cosine_after` to the stored value: 0.529611349
   (recomputed) vs. 0.529611287205324 (stored), a difference attributable
   to float32/float64 rounding, not a discrepancy.

**Conclusion: all seven models' denoising outputs are genuine, internally
consistent with the independently-computed rest of the pipeline, and not
corrupted or fabricated.**

## Part 2: Why Llama-3.1-8B and Qwen3-14B were missing

`git log --diff-filter=A -- 'data/processed/*/X_neutral.npy'` shows the
first PCA-denoising rollout was commit `d1773c1` ("pca denoising",
2026-06-29, authored by collaborator Jorge, no accompanying
`step_logs/STEP_LOG.md` entry). That commit added `X_neutral.npy` for
exactly two models: Gemma-3-12B (`concept_vectors`) and Gemma-3-27B
(`concept_vectors_gemma3_27b`). Grepping the notebook that commit modified,
`notebooks/08_valence_denoising.ipynb`, for model names found only
"Gemma-3-12B" and "gemma-3-27b" ever mentioned. The subsequent
2026-07-18/19 "gemma4_remaining" pipeline wave added neutral-corpus
extraction and denoising as a standard step for every newly-onboarded model
(Gemma-4-12B, Gemma-4-26B-A4B, Gemma-4-31B, Qwen3.6-27B, Qwen3.6-35B-A3B)
but never revisited the two original models it did not already cover.

This reads as a scope limitation of an exploratory notebook pass focused on
the Gemma-3 family specifically (contemporaneous with the "Gemma scale
paradox" investigation into that family's elevated cos(W,C)), not a
deliberate exclusion or a later regression. Confirmed via SSH that no
`X_neutral.npy` exists for either model anywhere on SCCKN either (checked
both `/work/emrecan.ulu/normalcy-axis` and `/work/emrecan.ulu/normalcy-axis-parity`),
ruling out a sync gap.

## Part 3: Closing the gap

`src/extract_neutral.py` requires loading actual model weights
(`load_hooked_model`), unlike the other gaps closed this session (split-half,
cross-axis zero-shot, cross-model agreement), which only needed numpy/scipy
on already-extracted vectors. This cannot run on the local machine (no GPU).

Wrote two new SGE scripts mirroring existing conventions exactly (same
`conda activate wc-tl` environment used by the original Llama/Qwen3-14B
extraction jobs, same `--config config/config.yaml --model <name>
--vectors-subdir <dir>` invocation pattern, same `jobs/sync_outputs.sh`
finishing step):

- `jobs/sge/extract_neutral_llama31_8b.sh`
- `jobs/sge/extract_neutral_qwen3_14b.sh`

Before submitting, checked live SGE state (`qstat -f -q 'gpu@*'`) rather
than trusting a GPU-status dashboard screenshot the user had provided: the
dashboard reported `scc192` (L40) and `spiderman` (A100) as idle/free, but
the live scheduler showed both in state `d` (disabled) — jobs submitted
there would queue indefinitely without running. Attempted to find the
disable reason via MOTD, `qstat -explain`, `qconf -sq`, and the cluster's
public status page, without success (the status page is JS-rendered and
not readable via `curl`); this is flagged for the user to check directly or
raise with the cluster admin. `scc213` and `scc214` were busy (0/8 free)
but not disabled, so both jobs were retargeted to
`-q gpu@scc213,gpu@scc214` and submitted:

- Job `1204966` — Llama-3.1-8B-Instruct
- Job `1204967` — Qwen3-14B

Both confirmed in `qw` (queued/waiting) state via `qstat -u emrecan.ulu` at
submission time. Neither model's weights need downloading (both already
cached under `HF_HOME` on SCCKN).

## Results so far (seven models)

| Model | k (PCs removed) | Variance kept | cos(W,C) before | cos(W,C) after |
|---|---:|---:|---:|---:|
| Gemma-3-12B-it | 1 | 56.1% | 0.749 | 0.530 |
| Gemma-3-27B-it | 43 | 50.2% | 0.708 | 0.487 |
| Gemma-4-12B | 11 | 51.2% | 0.494 | 0.473 |
| Gemma-4-26B-A4B | 11 | 50.3% | 0.587 | 0.564 |
| Gemma-4-31B | 17 | 50.6% | 0.526 | 0.469 |
| Qwen3.6-27B | 27 | 50.2% | 0.580 | 0.560 |
| Qwen3.6-35B-A3B | 17 | 50.3% | 0.619 | 0.595 |

Denoising reduces cos(W,C) in every model but never eliminates it. The
reduction is largest in the two Gemma-3 models (0.749→0.530, 0.708→0.487)
and comparatively small in Gemma-4/Qwen3.6 (typically −0.02 to −0.06),
consistent with the manuscript's existing shared-valence narrative: some of
the warmth-competence overlap is removable general tone, and some appears
to be a more structural property that a single linear neutral-tone
subtraction does not remove.

## Next Steps

- Poll `qstat -u emrecan.ulu` on SCCKN for jobs `1204966` / `1204967`.
- On completion, run the same three-way verification used for the other
  seven models (metadata consistency, cross-source cosine agreement,
  independent recomputation) before trusting the new outputs.
- Write a short follow-up report (or an update to this one) with the
  completed nine-model table.
- Separately, the cause of `scc192` / `spiderman` showing SGE state `d`
  remains unresolved; worth a direct check of the cluster status page in a
  browser or a question to the cluster admin if it recurs.
