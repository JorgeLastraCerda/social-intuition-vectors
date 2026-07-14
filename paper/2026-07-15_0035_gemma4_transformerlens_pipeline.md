# Gemma 4 TransformerLens Replication Pipeline

- **Produced:** 2026-07-15 00:35 Europe/Berlin
- **Model:** Gemma 4 31B-IT and Gemma 4 26B-A4B-IT
- **Scope:** TransformerLens Bridge implementation and SCCKN execution design
- **Status:** Implementation complete; SCCKN smoke and production runs pending

## Artifacts

- **Scripts:** `src/utils/model_loader.py`, `src/utils/prompting.py`, `src/extract_vectors.py`, `src/layer_sweep.py`, `src/extract_neutral.py`, `src/denoise_vectors.py`, `src/dense_steering.py`, `src/hiring_steering.py`, `src/hiring_audit.py`, `src/hiring_disparity.py`, `src/hiring_r4.py`, `src/validate_gemma4_run.py`, `smoke_tests/gemma4_transformerlens/smoke_test_bridge.py`, `jobs/setup_gemma4_env.sh`, `jobs/sge/gemma4_*.sh`, `jobs/sge/submit_gemma4.sh`
- **Inputs:** `config/config.yaml`, `requirements-gemma4.txt`, `data/stimuli/concept_stories.jsonl`, `data/stimuli/neutral_corpus.jsonl`, `data/raw/SocialPerceptions-Predict-Callback-main/`

## Summary

The production probing pipeline now loads open-weight models through TransformerLens 3's
`TransformerBridge`, preserving raw Hugging Face weights. This path supports the Gemma 4
31B dense model and 26B-A4B mixture-of-experts model while retaining the project's existing
residual-stream hook interface. The implementation does not enable legacy weight folding.

Passive story, neutral-text, and name-activation inputs remain raw text with BOS, preserving
the existing probe design. Yes/No judgment and hiring prompts use the model's native chat
template with thinking disabled. Candidate token IDs are resolved as exact one-token
continuations of the rendered prompt instead of assuming that `" Yes"` and `" No"` are
single standalone tokens.

## Replication coverage

Both models are configured to receive the existing tests without architecture-specific
extensions: concept extraction, held-out validation, all-layer sweep, raw dense steering,
neutral-corpus PCA denoising, broad and local hiring steering, denoised local hiring
steering, the 282-name audit, disparity, bootstrap mediation, and the 149-name R4 join and
OLS analysis. Router/expert analyses, Gemma Scope, and SAE experiments are excluded.

## SCCKN gates

The dedicated `wc-tl-g4` environment pins TransformerLens 3.5.1 and Transformers 5.13.0
while preserving the cluster's CUDA-compatible PyTorch build. Smoke jobs verify the model
dimensions, residual hook, activation shape, finite values, native-chat continuation tokens,
Bridge-versus-underlying-HF logits, causal steering response, and peak allocated VRAM.

Production jobs use Grid Engine dependencies. The dense 31B chain completes before the
26B-A4B chain begins, preventing output synchronization races on the shared 96 GB GPU node.
Any model-load, OOM, tokenization, hook, or non-finite-value failure stops dependent work;
there is no quantized or smaller-model fallback.

## Verification status

Local tests pass (`16 passed`). Python compilation, shell syntax checks, stimulus dry runs,
neutral-corpus dry runs, and Git whitespace validation pass. GPU/model-load acceptance remains
pending because the models must run on SCCKN.

## Sources

- [TransformerLens repository and Bridge quick start](https://github.com/TransformerLensOrg/TransformerLens)
- [Gemma 4 31B model card](https://huggingface.co/google/gemma-4-31B-it)
- [Gemma 4 26B-A4B model card](https://huggingface.co/google/gemma-4-26B-A4B-it)
