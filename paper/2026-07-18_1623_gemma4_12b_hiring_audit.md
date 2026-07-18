# Gemma 4 12B Unsteered Hiring Audit

- **Produced:** 2026-07-18 16:23 Europe/Berlin
- **Model:** `google/gemma-4-12B-it`
- **Scope:** Full 282-name probe-versus-human and unsteered callback association audit
- **Status:** Complete

## Artifacts

- **Scripts:** `src/hiring_audit.py`, `src/validate_gemma4_remaining.py`, `jobs/sge/gemma4_remaining_run.sh`
- **Inputs:** `config/gemma4_12b.yaml`, `data/processed/concept_vectors_gemma4_12b/`, `data/raw/gallo_hausladen/`
- **Outputs:** `results/tables/hiring_audit_gemma4_12b.csv`, `results/logs/hiring_probe_vs_human_gemma4_12b.json`, `results/logs/gemma4_12b_audit_20260718T141249Z_12b_audit.{out,err}`

## Results

Job `1145333` completed on one exact NVIDIA L40 with `failed=0`, `exit_status=0`, and 155 seconds wallclock. All 282 rated names were evaluated with the pinned native-chat checkpoint.

| Association | Spearman rho | p |
|---|---:|---:|
| Model warmth probe vs human warmth | 0.020 | 0.742 |
| Model competence probe vs human competence | 0.222 | <0.001 |
| Callback margin vs model warmth probe | -0.110 | 0.066 |
| Callback margin vs model competence probe | -0.124 | 0.038 |
| Callback margin vs human warmth | -0.001 | 0.986 |
| Callback margin vs human competence | 0.110 | 0.065 |

The warmth direction does not align with the human warmth ranking in this name-only hiring setting. Competence shows a statistically detectable but modest positive alignment with human competence ratings. Meanwhile, the model's unsteered callback margin is negatively associated with its own competence projection, which motivates the planned causal steering test rather than supporting a simple positive mediation account.

## Caveat

These are observational name-level associations. The bfloat16 Yes/No margin limitation remains active, and causal interpretation is deferred to the local, broad, and denoised steering sweeps.
