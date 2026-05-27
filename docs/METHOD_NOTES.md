# Method Notes

This file is the working bridge between the source papers and the implementation. It should be expanded after the PDFs and benchmark data are verified.

## Sources

- Sofroniew, Kauvar, Saunders, Chen, et al. (2026). _Emotion Concepts and their Function in a Large Language Model._
- Gallo, Hausladen, Hsu, Jenkins, Ona, Camerer (2024). _Perceived warmth and competence predict callback rates in meta-analyzed North American labor market experiments._

## Current Implementation Recipe

1. Generate high/low warmth and high/low competence text corpora.
2. Run the configured open-weights model and capture residual-stream activations.
3. Select the probe layer using `probing.probe_layer_frac`.
4. Average activations from `probing.start_token` onward.
5. Build warmth and competence vectors from condition contrasts.
6. Validate against held-out text and PLOS warmth/competence ratings.
7. Steer vectors at configured strengths and measure callback changes.

## Open Items After Paper Reading

- Exact neutral-corpus PCA denoising threshold and implementation details.
- Exact activation aggregation choices for short texts.
- Exact PLOS supplementary file and column mapping for social signals, ratings, and callback rates.
- Exact binary callback scoring method for the selected model tokenizer.
