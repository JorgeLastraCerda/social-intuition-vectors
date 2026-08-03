# Nine-model warmth–competence representation geometry

- **Produced:** 2026-07-19 14:56 Europe/Berlin
- **Models:** Gemma-3-12B · Gemma-3-27B · Llama-3.1-8B · Gemma-4-12B · Gemma-4-26B-A4B · Gemma-4-31B · Qwen3-14B · Qwen3.6-27B · Qwen3.6-35B-A3B
- **Scope:** Cross-model synthesis of raw probe-layer warmth/competence direction angles and 200-story oblique projection geometry
- **Status:** Complete; integrated as a main-text empirical figure

## Artifacts

- **Scripts:** `paper/figures/generate_figures.py`
- **Inputs:** `data/stimuli/concept_stories.jsonl`; `data/processed/concept_vectors/`; `data/processed/concept_vectors_{gemma3_27b,llama31_8b,gemma4_12b,gemma4_26b_a4b,gemma4_31b,qwen3_14b,qwen36_27b,qwen36_35b_a3b}/`
- **Figures:** `paper/figures/paper_figure1_axis_arrows.{png,pdf}`

## Summary

All nine raw probe-layer warmth and competence directions have positive cosine similarity. Cosines range from 0.494 to 0.749, equivalent to inter-axis angles from 60.4° to 41.5°. The directions are therefore neither redundant nor orthogonal in any tested model. Gemma 3 has the strongest geometric alignment, while Gemma 4, Qwen, and Llama occupy a more oblique range.

The expanded figure uses the same 200 synthetic stories per model, model-internal z-scoring, shared plot limits, and equal x/y data scaling. The plotted arrow angle therefore matches the numerical angle derived from the direction-vector dot product rather than serving as a schematic approximation.

## Results

| Figure row | Model | cos(W,C) | Angle |
|---|---|---:|---:|
| Earlier baselines | Gemma-3-12B | 0.748953 | 41.500° |
| Earlier baselines | Gemma-3-27B | 0.707798 | 44.944° |
| Earlier baselines | Llama-3.1-8B | 0.505418 | 59.641° |
| Gemma 4 | Gemma-4-12B | 0.493539 | 60.427° |
| Gemma 4 | Gemma-4-26B-A4B | 0.586665 | 54.079° |
| Gemma 4 | Gemma-4-31B | 0.526157 | 58.254° |
| Qwen lineage | Qwen3-14B | 0.535891 | 57.596° |
| Qwen lineage | Qwen3.6-27B | 0.579814 | 54.563° |
| Qwen lineage | Qwen3.6-35B-A3B | 0.619180 | 51.744° |

Every model contributed four finite 50-story matrices and two finite, dimension-matched direction vectors. Recomputed cosines and angles agree with the figure annotations at their displayed precision; `cos(angle)` agrees with the direct vector cosine within `1e-12`.

## Interpretation

The cross-model comparison strengthens the shared-evaluative-component interpretation without establishing that warmth and competence are interchangeable. Gemma-3-12B and Gemma-3-27B have visibly narrower angles than the other seven checkpoints. Within Gemma 4, the MoE 26B-A4B variant is more aligned than either dense variant. The Qwen sequence moves toward greater overlap from Qwen3-14B to Qwen3.6-35B-A3B, but architecture and active parameter count change across these checkpoints, so this is not a controlled scale trend.

The story clouds also show why perfect target-axis classification does not imply construct purity. High warmth and high competence stories occupy related positive regions, while both low conditions occupy related negative regions. Direction-specific topic validation and strict cross-axis controls remain necessary for separating target information from shared evaluative framing.

## Display contract and caveats

Each model is z-scored internally before the oblique transform
`x = z_w + z_c cos(theta)`, `y = z_c sin(theta)`. Common limits of `x = [-4.6, 4.4]` and `y = [-2.0, 2.6]`, together with equal x/y scaling, permit direct visual comparison and preserve the displayed arrow angle. The figure uses raw dense directions only; PCA-denoised vectors are not mixed into the panel grid.

The publication layout omits an internal figure title, places the shared condition legend immediately above the panel grid, and uses single shared warmth- and competence-axis labels close to the grid. Model titles color only the family prefix: Gemma generations use a high-separation navy-to-blue progression, Qwen generations use distinct violet-to-raspberry purples, and Llama uses a separate vivid orange. All family colors exceed a 4.5:1 contrast ratio on white; size and architecture suffixes remain neutral. Direction labels use regular font weight. The manuscript caption carries the figure-level explanation.

These geometries come from one balanced synthetic story distribution at the fixed probe layer. They do not establish natural-text construct validity, independence of the two social dimensions, or a monotonic relationship with nominal model size.
