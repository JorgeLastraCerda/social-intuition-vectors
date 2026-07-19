# Qwen3.6 27B Full-282 Balanced Comparison Synthesis

- **Produced:** 2026-07-19 12:15 Europe/Berlin
- **Model:** `Qwen/Qwen3.6-27B`
- **Scope:** Comparison of the original 60-name steering panels with three post-hoc 282-name replications
- **Status:** Complete; all six axis-by-regime endpoint comparisons are stable

## Artifacts

- **Scripts:** `src/qwen36_hiring.py`, `src/validate_qwen36_hiring.py`, `src/summarize_hiring_steering.py`, `jobs/ccu/run_qwen36_hiring.sh`
- **Inputs:** `config/qwen36_27b.yaml`, `results/logs/hiring_steering_summary_qwen36_27b_local.json`, `results/logs/hiring_steering_summary_qwen36_27b_broad.json`, `results/logs/hiring_steering_summary_qwen36_27b_denoised_local.json`
- **Outputs:** `results/logs/hiring_steering_summary_qwen36_27b_local_full282.json`, `results/logs/hiring_steering_summary_qwen36_27b_broad_full282.json`, `results/logs/hiring_steering_summary_qwen36_27b_denoised_local_full282.json`

## Comparison

| Regime | Axis | 60-name endpoint | 282-name endpoint | Relative change | 282-name monotone |
|---|---|---:|---:|---:|:---:|
| Raw local (+0.10) | Warmth | +1.196 | +1.193 | -0.2% | Yes |
| Raw local (+0.10) | Competence | +0.533 | +0.519 | -2.8% | Yes |
| Raw broad (+0.50) | Warmth | +2.240 | +2.237 | -0.1% | Yes |
| Raw broad (+0.50) | Competence | +1.069 | +1.055 | -1.3% | Yes |
| Denoised local (+0.10) | Warmth | +1.140 | +1.133 | -0.5% | Yes |
| Denoised local (+0.10) | Competence | +0.408 | +0.409 | +0.1% | Yes |

## Interpretation

Panel expansion changes every endpoint by less than 3% and preserves positive signs, bootstrap-resolved intervals, and monotonicity for both axes in all three regimes. The original 60-name estimates were therefore representative of the full rated-name panel for Qwen3.6-27B.

The original predeclared gate remains a valid `run_full_282=false` result because none of its instability criteria fired. These later runs answer a different, user-approved balanced-comparison question and are labeled post-hoc. They add sampling robustness without rewriting the confirmatory protocol.

The full-name broad result also sharpens the cross-model contrast: Qwen3.6-27B retains positive monotone competence steering at +0.50, whereas Qwen3.6-35B-A3B reproduces a negative broad endpoint across all 282 names. Name-panel sampling does not explain that model difference.
