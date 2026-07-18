# Gemma 4 26B-A4B and 31B layer-sweep results

- **Produced:** 2026-07-18 12:08 Europe/Berlin
- **Models:** `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it`
- **Scope:** Stage 3 all-layer warmth and competence probe sweep on 200 synthetic concept stories
- **Status:** Complete for 26B-A4B and 31B; the separate 12B Stage 3 OOM remains unresolved

## Artifacts

- **Scripts:** `src/layer_sweep.py`, `src/validate_gemma4_stage.py`, `jobs/sge/gemma4_stage3_retry.sh`, `jobs/sge/gemma4_stage3_finalize.sh`, `jobs/sge/submit_gemma4_stage3_retry.sh`
- **Inputs:** `config/config.yaml`, `data/stimuli/concept_stories.jsonl`
- **Outputs:** `results/tables/layer_sweep_gemma4_26b_a4b.csv`, `results/tables/layer_sweep_gemma4_26b_a4b.meta.json`, `results/tables/layer_sweep_gemma4_31b.csv`, `results/tables/layer_sweep_gemma4_31b.meta.json`, `results/logs/gemma4_stage3_retry_submission_20260718T100211Z.json`, `results/logs/gemma4_stage3_retry_20260718T100211Z_{26b,31b,final}.{out,err}`

## Summary

Both retries passed their structural, metadata, completeness, and finite-value gates. The configured probe-layer rows exactly reproduce the Stage 2 Cohen's d values for both axes and models. Warmth and competence separation is strong across almost the entire depth, but it is not a late-onset representation: topic-holdout accuracy is already 0.80 or higher at layer 0 and reaches 1.00 in the middle layers. Effect sizes peak before the configured 0.66-depth probe layer and then decline toward the final layer.

## Execution outcome

| Model | Grid Engine job | GPU | Wall time | Exit status | Maximum virtual memory |
|---|---:|---|---:|---:|---:|
| Gemma 4 26B-A4B | `1144931` | RTX PRO 6000, physical device 2 | 59 s | 0 | 45.325 GB |
| Gemma 4 31B | `1144932` | RTX PRO 6000, physical device 6 | 78 s | 0 | 52.678 GB |
| CPU finalizer | `1144933` | None | 58 s | 0 | 108.539 MB |

The two model jobs started concurrently on distinct devices. Neither job performed Git operations. The dependent CPU finalizer validated both outputs and pushed them in one commit (`8512efa`).

## Probe-layer replication

| Model | Probe layer | Warmth topic CV | Competence topic CV | Warmth d | Competence d | cos(W,C) | Mean residual norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B-A4B | 19 / 30 | 1.00 | 1.00 | 8.3571 | 8.7539 | 0.5867 | 68.4368 |
| Gemma 4 31B | 39 / 60 | 1.00 | 1.00 | 7.5624 | 6.0319 | 0.5262 | 188.5253 |

The warmth and competence Cohen's d values match the Stage 2 tables exactly, with a numeric difference of 0.0 for all four comparisons. This confirms that the all-layer extraction reproduces the fixed-layer result at the configured probe layer.

## Depth profiles

| Model | Peak layer | Peak fraction | Peak warmth d | Peak competence d | cos(W,C) at peak | Warmth topic-CV range | Competence topic-CV range |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B-A4B | 16 | 0.5517 | 9.1377 | 9.7773 | 0.7102 | 0.80–1.00 | 0.94–1.00 |
| Gemma 4 31B | 24 | 0.4068 | 11.4946 | 9.6113 | 0.5072 | 0.81–1.00 | 0.95–1.00 |

For 26B-A4B, the axis cosine ranges from -0.1878 at layer 0 to a maximum of 0.7362 at layer 12. For 31B, it ranges from -0.1830 at layer 0 to 0.7046 at layer 28. Thus, the warmth and competence directions become most aligned in the middle network rather than remaining at one fixed angle throughout depth.

At the final layer, separation drops to d = 0.8001/1.4756 for 26B-A4B and d = 0.7254/1.9932 for 31B (warmth/competence). The configured 0.66-depth probe therefore samples a strong but not maximal representation and avoids the late-layer collapse.

## Interpretation and caveats

The result supports depth-wide linear probeability and mid-layer amplification of warmth and competence signals. It does not establish that distinct SCM dimensions emerge from an initially neutral representation, because classification is already strong at the first residual layer. Lexical and general evaluative structure in the synthetic stories can contribute to this early signal.

High middle-layer cosines also reinforce the Stage 2 caveat: large within-axis effect sizes do not imply independent warmth and competence representations. This sweep measures geometry and same-axis topic holdout at each layer; it does not add new cross-axis classification, human-rating validation, or hiring-decision evidence. Those external and discriminant validations remain necessary.

The failed 12B Stage 3 job is outside this retry result. Its CUDA OOM should be diagnosed separately rather than inferred from the successful 26B-A4B and 31B executions.
