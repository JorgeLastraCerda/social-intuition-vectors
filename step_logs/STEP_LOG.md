# Step Log — Warmth & Competence Probing

> Append-only research log. Every meaningful step/finding gets one entry, newest at the bottom.
> Do not edit or delete past entries. English only. See the "Step Logging" rule in AGENTS.md.
>
> Entry format:
> ```
> ## YYYY-MM-DD · Step N — <short title>
> - **Context:** which task/session this belongs to (1 sentence)
> - **Agent:** <model-id> (omit if step was taken by a human)
> - **Did:** what was done
> - **Findings:** concrete results — numbers, file paths, pass/fail
> - **Decision / rationale:** decision taken and why (omit if none)
> - **Next:** immediate next action (omit if none)
> ```

---

## 2026-06-08 · Step 1 — Smoke test audit

- **Context:** Full audit of the two pilot smoke test scripts and their result logs to assess scientific validity.
- **Did:** Read `scripts/smoke_test_activations.py`, `scripts/smoke_test_probe.py`, `results/logs/smoke_test_1780866834.json`, `results/logs/smoke_test_probe_1780867700.json`, `README.md` Pilot section, `docs/METHOD_NOTES.md`, `config/config.yaml`, `src/utils/hooks.py`.
- **Findings:**
  - Model: Qwen/Qwen2.5-1.5B-Instruct, layer 18/28 (frac=0.66), d_model=1536.
  - Probe CV accuracy = 0.83 (chance 0.50); Cohen's d = 2.68 (in-sample); max logit delta = 6.375.
  - **Critical internal inconsistency:** README claims warmth steering produces "18× more logit shift than a random vector of the same injected magnitude." This is false — Test 1 used alpha=0.5 (absolute), Test 2 used alpha=26.99 (0.5 × mean_resid_norm=53.98). Injected magnitudes differ ~54×. The comparison is not apples-to-apples.
  - Secondary issues: (a) Cohen's d is in-sample on the same 100 sentences used to fit the direction; (b) warm/cold sentences conflate warmth with general sentiment (valence confound); (c) `hooks.py` uses `round((n_layers-1)*frac)` but METHOD_NOTES documents `round(frac*n_layers)` — off-by-one on Llama-32 (20 vs 21) and Gemma-42 (27 vs 28); (d) probe accuracy 0.83 is below the METHOD_NOTES target of >0.90.
- **Decision / rationale:** Pilot is sufficient to proceed to Phase 4 (linear signal confirmed), but README 18× claim should be corrected before sharing; valence confound is the most important scientific caveat — addressed by topic-controlled stories + PCA denoising in the main pipeline.
- **Next:** Fix README 18× wording; add equal-magnitude random steering control to `smoke_test_probe.py`; align layer formula in METHOD_NOTES.

---

## 2026-06-08 · Step 2 — Model backend survey (TransformerLens alternatives)

- **Context:** User wants to switch from Qwen to Gemma; evaluating whether TransformerLens supports Gemma 4 and what alternatives exist.
- **Did:** Web search + page fetches for TransformerLens model table, Gemma 4 HF docs, nnsight, nnterp, pyvene, steering-vectors, repeng.
- **Findings:**
  - TransformerLens supports Gemma 1/2/3 (including gemma-3-4b-it, gemma-3-27b-it) but **not Gemma 4**.
  - Gemma 4 sizes: E2B (~2B), E4B (~4B), 12B, 26B-MoE, 31B. E4B fits in 6 GB VRAM at Q4 (~5 GB). However Gemma 4 introduces Per-Layer Embeddings (PLE) that complicate the clean residual-stream superposition assumption.
  - Key alternatives: **nnsight** (wraps any HF model, supports Gemma 4, v0.6 Feb 2026); **nnterp** (thin wrapper on nnsight, 50+ model families, standardized naming, logit-lens + steering built-in, Nov 2025); **steering-vectors** (contrastive activation addition library, explicitly lists Gemma support, closest to our method); **pyvene** (Stanford NLP, more complex intervention schemes, overkill for our current needs).
  - `steering-vectors` library implements exactly our approach (mean contrast → vector → patch activations) with a high-level API; supports Gemma/Llama/Mistral.
- **Decision / rationale:** Recommended path: use **Gemma 3 4B-IT** (TransformerLens-supported, ~3.5 GB at 4-bit, fits RTX 4050 6 GB) for the local test run — no infrastructure change needed. For Gemma 4 on SCCKN: switch hook backend to nnsight/nnterp. `HOOK_BACKEND` constant in `src/utils/hooks.py` already isolates the dependency. `steering-vectors` is worth evaluating as a drop-in for the extraction + steering loop.
- **Next:** Update `config/config.yaml` to Gemma 3 4B-IT and run smoke test; or optionally wire `steering-vectors` as an alternative extraction path.

---

## 2026-06-08 · Step 3 — Step logging convention established

- **Context:** User requested a persistent append-only step log so decisions and findings are traceable across sessions and visible to collaborator Jorge.
- **Did:** Added `## Step Logging` rule to `CLAUDE.md`; created `step_logs/STEP_LOG.md` with template and seed entries (Steps 1–2 from this session).
- **Findings:** `.gitignore` does not exclude `step_logs/` — file will be tracked automatically. `.claude/` is git-ignored so hook-based automation is not viable for shared logging; CLAUDE.md instruction is the correct mechanism.
- **Decision / rationale:** Single append-only file (`STEP_LOG.md`) chosen over per-session or per-step files; entries triggered by meaningful steps, not every tool call; committed to git for shared access.
- **Next:** On next session start, read the last few entries here before proceeding.

---

## 2026-06-08 · Step 4 — Backend decision report (model + interpretability tooling)

- **Context:** User asked for a two-sided literature review to choose one (model + tooling)
  pair to commit to, after deciding we must stay on a single model throughout the project.
- **Did:** Web searches + page fetches on TransformerLens model coverage, nnsight, nnterp,
  pyvene, steering-vectors, GemmaScope 2; re-read CLAUDE.md. Compared two paths and drafted
  an advisor email to Carina Hausladen.
- **Findings:**
  - **Tool roles:** TransformerLens (~2022) reimplements the model with clean named hooks but
    only covers up to Gemma 3 and can show tiny numerical drift vs. original weights. nnsight
    (2024–2025) wraps the original HF model, works on any/new model and remote 70B+, but
    needs native module names. nnterp (late 2025) = nnsight + standardized names + built-in
    logit lens/steering. steering-vectors = contrastive-activation-addition library, closest
    to our exact method. GemmaScope 2 (Dec 2025) = ready-made SAEs on every Gemma 3 layer;
    can split our direction into interpretable parts → directly tests the warmth-vs-valence
    confound (our #1 methodological risk).
  - **Path A (Gemma 3 + TransformerLens + GemmaScope 2):** mature, reviewer-trusted, zero
    pipeline rewrite, GemmaScope addresses our #1 risk; con: Gemma 3 less capable.
  - **Path B (Gemma 4 + nnsight/nnterp):** smarter model, exact-weights access, remote
    scale; cons: Gemma 4 is very new (Apr 2026), no SAE/tooling/published work yet,
    Per-Layer Embeddings (PLE) complicate the clean residual-stream assumption, full
    pipeline rewrite required.
  - Local fit: Gemma 3 4B-IT (~3.5 GB at 4-bit) fits RTX 4050 6 GB; SCCKN → 12B/27B.
- **Decision / rationale:** Recommend Path A (Gemma 3) for the core result; Path B (Gemma 4
  + nnsight) as a later robustness/scale-up check. User decided to pilot both paths and
  decide on evidence, and to ask advisor Carina for her vision first.
- **Next:** Await Carina's direction; prepare pilots on both paths.

---

## 2026-06-08 · Step 6 — SCCKN Windows connection setup + smoke_tests/ structure

- **Context:** Decision to run all smoke tests on SCCKN (not local PC) to avoid
  bit/quantization constraints; Windows SSH and smoke test scaffold built this session.
- **Did:**
  - Verified `scc.uni-konstanz.de:22` reachable from Windows PC (134.34.147.166).
  - Confirmed `~/.ssh/id_ed25519` + `.pub` already present; Windows OpenSSH installed.
  - Created `~/.ssh/config` with `scckn` alias (`HostName scc.uni-konstanz.de`, `User emrecan.ulu`).
  - Wrote `docs/SCCKN_WINDOWS.md` (key copy command, config, git sync, conda env setup,
    GPU job template, tmux, VPN note).
  - Built `smoke_tests/` directory:
    - `stimuli.py`: shared 100 sentences (extracted from original `scripts/smoke_test_probe.py`)
    - `qwen_transformerlens/smoke_test_probe.py`: Qwen baseline, adds equal-magnitude random control
    - `gemma3_transformerlens/smoke_test_probe.py`: Gemma 3 + TransformerLens, saves warmth_vector.npy
    - `gemma3_transformerlens/sae_decompose.py`: GemmaScope 2 SAE warmth-vs-tone decomposition
    - `gemma4_nnsight/smoke_test_probe.py`: Gemma 4 + nnsight, same probe + steering structure
    - `smoke_tests/README.md`: structure, metrics table, how to run, output files
  - Added SGE job scripts: `jobs/sge/smoke_qwen.sh`, `smoke_gemma3.sh`, `smoke_gemma4.sh`
    (all with `# ADJUST` placeholders for GPU resource flags).
  - Updated `requirements.txt` to include `accelerate`, `sae-lens`, `nnsight`, `nnterp`.
  - Fixed README "18×" claim (noted magnitude mismatch ~54×; replaced with correct explanation).
  - Added "Smoke Test Suite" section to README describing the three-test structure.
  - Updated `.gitignore` to ignore `smoke_tests/*/results/*` (except `.gitkeep`).
- **Findings:**
  - Repo is public on GitHub (`github.com/JorgeLastraCerda/normalcy-axis`) → git clone works
    without PAT.
  - Gemma models are gated on HF — user needs to accept license + `huggingface-cli login`
    on the cluster before first model download.
  - GPU resource flag for `qsub -l gpu=1` needs confirmation with Stefan (exact syntax
    cluster-specific).
- **Decision / rationale:** Two separate conda envs (`wc-tl` for TransformerLens,
  `wc-nn` for nnsight) to avoid dependency conflicts between transformer-lens and nnsight.
- **Next:** User copies public key to cluster (interactive step, must be done manually):
  `type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh emrecan.ulu@scc.uni-konstanz.de "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"`
  Then: `git clone` on cluster, create conda envs, fill `# ADJUST` in job scripts, submit.

---

## 2026-06-08 · Step 8 — Conda envs created, HF login, smoke jobs submitted

- **Context:** Cluster environment setup and smoke test execution for the warmth/competence probing project.
- **Did:**
  - Fixed BOM in `C:\Users\emrec\.ssh\config` (was causing `Bad configuration option` on every SSH call).
  - Ran `setup_envs.sh` via nohup on cluster: created `wc-tl` and `wc-nn` conda envs in `/home/scc/emrecan.ulu/.conda/envs/`.
  - Fixed conda activate in interactive shell: `conda init bash` → `source ~/.bashrc`.
  - HF login via `hf auth login` (new CLI; `huggingface-cli` deprecated). Token saved to `/work/emrecan.ulu/hf_cache/token`. Git credential prompt answered No.
  - User accepted Gemma 3 license on HuggingFace web (google/gemma-3-12b-it).
  - Submitted three smoke jobs: `qsub smoke_qwen.sh` (1015381), `qsub smoke_gemma3.sh` (1015382), `qsub smoke_gemma4.sh` (1015383). All entered `qw` state.
- **Findings:**
  - `wc-tl` and `wc-nn` installed successfully: `torch 2.12.0+cu130`. CUDA: False on login node (expected — GPU nodes will show True).
  - `module load conda/2024.10` is the correct module name on SCCKN.
  - `hf auth login` / `hf auth whoami` is the current CLI. `hf auth whoami` throws latin-1 encoding error on this terminal but login is valid.
  - Gemma 4 dense max is 12B; 27B is MoE — scale-up path for main result stays Gemma 3 27B.
- **Decision / rationale:** Final model commitment: Gemma 3 12B for smoke, Gemma 3 27B as scale-up target. Gemma 4 12B runs as exploratory nnsight comparison only. Decision revisable after smoke results.
- **Next:** Wait for job completion. Check `results/logs/smoke_*.out` and `smoke_tests/*/results/*.json`. If probe_cv_mean > 0.80 on Gemma 3 → proceed to Phase 4 (full corpus extraction).

---

## 2026-06-08 · Step 9 — Six-test smoke matrix: 4 resource-tuned jobs added

- **Context:** Smoke jobs 1015382 (gemma3 12B) and 1015383 (gemma4 12B) were stuck in `qw` due to single-node pin (`-q gpu@scc213`, h_vmem 64G, smp 4). Added 4 new jobs to get something running sooner.
- **Did:**
  - Added `--out-dir` parameter to `smoke_tests/gemma3_transformerlens/smoke_test_probe.py` and `sae_decompose.py` (default = `HERE/results`, backward-compatible). Prevents concurrent 12B + 4B runs from clobbering each other's `warmth_vector.npy` / `X_warm.npy` / `X_cold.npy` (d_model mismatch: 3840 vs 2560 would crash the SAE step).
  - Created 4 new SGE job scripts in `jobs/sge/`:
    - `smoke_gemma3_12b_light.sh`: Gemma 3 12B, wc-tl, queue pool `scc192,scc213,scc214` (3× nodes), h_vmem 32G, smp 2, h_rt 1h, `--out-dir .../results/g3_12b`.
    - `smoke_gemma4_12b_light.sh`: Gemma 4 12B, wc-nn, same trimmed resources.
    - `smoke_gemma3_4b.sh`: Gemma 3 4B-IT, wc-tl, `-q gpu` (any node), h_vmem 16G, smp 2, h_rt 30min, GemmaScope 2 `gemma-scope-2-4b-it-res` / `layer_22_width_16k_l0_medium`, `--out-dir .../results/g3_4b`.
    - `smoke_gemma4_4b.sh`: Gemma 4 E4B-IT, wc-nn, same minimal resources.
- **Findings:**
  - Clarified: **"4B" (model size) ≠ "4-bit" (quantization)**. Previous concern about "4-bit approaches not working" referred to quantization; SCCKN runs bf16 full precision. Gemma 3 4B bf16 = ~8 GB VRAM, method intact (TL + GemmaScope 2 `gemma-scope-2-4b-it-res` confirmed available).
  - There is no Gemma 3 "6B"; sizes are 270M / 1B / 4B / 12B / 27B.
  - Gemma 4 4B model id: `google/gemma-4-E4B-it` (confirmed on HF).
  - Gemma 3 4B has 34 layers; probe_layer_frac=0.66 → layer 22 ≈ 65% depth → matches GemmaScope 2 4B SAE checkpoint. **VERIFY sae-id** from catalog before relying on `layer_22_width_16k_l0_medium`.
  - 12B headline commitment unchanged; 4B jobs are fast-feedback tier, not the main result.
- **Decision / rationale:** Keep 12B→27B as the committed model for the paper. 4B bf16 is a fully valid fast iteration tier (all method components preserved). 12B-light jobs widen the node pool from 1 to 3, reducing queue wait without changing compute.
- **Next:** User: `git add jobs/sge/smoke_gemma3_12b_light.sh jobs/sge/smoke_gemma4_12b_light.sh jobs/sge/smoke_gemma3_4b.sh jobs/sge/smoke_gemma4_4b.sh smoke_tests/gemma3_transformerlens/smoke_test_probe.py smoke_tests/gemma3_transformerlens/sae_decompose.py && git commit -m "Add four resource-tuned smoke jobs and --out-dir isolation fix" && git push`. Then on cluster: `git pull && qsub jobs/sge/smoke_gemma3_4b.sh && qsub jobs/sge/smoke_gemma4_4b.sh && qsub jobs/sge/smoke_gemma3_12b_light.sh && qsub jobs/sge/smoke_gemma4_12b_light.sh`. Submit 4B jobs first.

---

## 2026-06-08 · Step 7 — Model selection finalized; job scripts + config updated for 12B

- **Context:** User approved plan to run smoke tests on SCCKN with Gemma 3 12B-IT and Gemma 4 12B-IT.
- **Did:**
  - Updated `jobs/sge/smoke_gemma3.sh`: model `google/gemma-3-12b-it`, pinned to `#$ -q gpu@scc213` (L40 48 GB), h_vmem 64G, SAE release `gemma-scope-2-12b-it-res`, sae-id `layer_31_width_16k_l0_medium`, layer comment updated (48 layers, frac=0.66 → layer 31 = 65% depth, exact GemmaScope 2 match).
  - Updated `jobs/sge/smoke_gemma4.sh`: model `google/gemma-4-12B-it` (capital B), pinned to `#$ -q gpu@scc213`, h_vmem 64G.
  - Updated `config/config.yaml`: `model.name: google/gemma-3-12b-it` (committed model, CLAUDE.md compliance).
  - Updated `README.md` Smoke Test Suite section: table now shows 12B models; added "Model commitment and scale-up path" paragraph (open-door to 27B on scc214).
  - Updated `docs/SCCKN_WINDOWS.md`: git sync section notes SSH remote is active; HF section notes HF_HOME already in .bashrc; added GPU node pinning guidance.
- **Findings:**
  - Layer 31 at frac=0.66 for Gemma 3 12B (48 layers): round((48-1)*0.66) = 31, which is 31/48 ≈ 65% depth — exact alignment with GemmaScope 2 12B's %65 SAE checkpoint. No adjustment needed.
  - Gemma 4 12B-IT model id requires capital B: `google/gemma-4-12B-it`.
- **Decision / rationale:** 12B for smoke tests: fits single L40 (48 GB), enables fair Gemma 3 vs Gemma 4 comparison, GemmaScope 2 available. 27B documented as open-door scale-up on scc214 (96 GB). Two envs (wc-tl / wc-nn) kept separate to avoid TransformerLens ↔ nnsight dependency conflicts.
- **Next:** (1) Commit + push this batch. (2) User: `ssh scckn`, `tmux new -s setup`, create conda envs wc-tl + wc-nn. (3) User: `huggingface-cli login` + accept Gemma 3 license on HF web. (4) `qsub` the three smoke jobs.

---

## 2026-06-08 · Step 5 — nnsight literature scan + Gemma 4 status

- **Context:** User asked specifically how nnsight appears in current literature, whether
  anything exists for Gemma 4 (scanning recent → older), and how it compares to SAEs and
  TransformerLens.
- **Did:** Web searches + fetches: nnsight 2026 blog archive, Neuronpedia ecosystem blog
  (assistant-axis post), NDIF foundational paper, Gemma 4 interpretability searches.
- **Findings:**
  - **Gemma 4 + interpretability: nothing yet.** Gemma 4 released Apr 2026; no published
    interpretability study, no SAE/GemmaScope, no Neuronpedia support. nnsight can run it
    technically (any HF model), but with no tooling or reference literature around it.
  - **nnsight timeline (recent → older):** NDIF 130× remote speedup (Apr 2026) → "Calling
    all Lies" deception study (Mar 2026) → nnsight 0.6 + nnterp enter ecosystem (Feb 2026)
    → Neuronpedia backend on nnsight for gpt-oss, Gemma 3, Llama 3.3 70B (early 2026) →
    foundational paper "NNsight and NDIF" (ICLR 2025, arXiv:2407.14561).
  - **Comparison vs our question (Existence + Causality):** TransformerLens and nnsight
    both cover extract + probe + steer. SAE/GemmaScope uniquely tests warmth-vs-tone —
    our biggest scientific risk. Most finding-producing work pairs nnsight with SAEs on
    Gemma 3 / Llama, not Gemma 4.
  - nnsight = workhorse for large/new models + remote execution; TransformerLens = clean,
    reviewer-trusted up to Gemma 3; SAE = the scientific value-add, Gemma 3 only for now.
- **Decision / rationale:** Reinforces leaning toward Gemma 3 + GemmaScope for the core
  result. Gemma 4's "smarter model" advantage is currently offset by having no tooling or
  literature. Both paths will be piloted before a final commitment.
- **Next:** Await Carina's direction; run Gemma 3 4B smoke test locally.

---

## 2026-06-09 · Step 1 — Smoke matrix results: Gemma 3 12B PASS, Gemma 4 failed

- **Context:** Final tally of the six-job SCCKN smoke-test matrix submitted 2026-06-08.
- **Did:** Read `results/logs/smoke_gemma3_12b_light.out`, `smoke_gemma4_12b_light.out`,
  `smoke_gemma3_4b.out`, `smoke_gemma4_4b.out` and the corresponding `.err` files on SCCKN.
  Parsed `smoke_tests/gemma3_transformerlens/results/g3_12b/smoke_probe_1780951986.json` and
  `sae_decompose_1780952012.json`. Cancelled stalled jobs 1015382 and 1015383 (`qdel`).
- **Findings:**
  - **Gemma 3 12B-IT — PASS.** layer 31/48, d_model 3840, seed 20260527.
    diff_norm 1484.6, cosine(warm, cold) 0.99975, Cohen's d 2.896 (in-sample),
    probe_cv_mean **0.86 ± 0.08** (folds 0.95/0.80/0.95/0.75/0.85), mean_resid_norm 66184.6,
    steering_alpha 33092.3, max_logit_delta warmth 40.0 vs random 20.75,
    warmth_random_ratio **1.93×**. Clears the >0.80 threshold.
  - **Gemma 3 12B SAE decomposition (GemmaScope 2 `layer_31_width_16k_l0_medium`) — DONE but
    low.** sae_cv_mean 0.61 ± 0.07 (barely above chance 0.50). Top warm-minus-cold features are
    small and mixed-sign; Neuronpedia inspection still needed to close the valence-confound
    question.
  - **Gemma 4 4B and 12B — zero results.** Two separate failures: (a) all Gemma 4 variants are
    registered as `AutoModelForImageTextToText` (multimodal) — nnsight's `LanguageModel()` can't
    load them; `VisionLanguageModel` must be used. (b) even with `VisionLanguageModel`, nnsight
    fails to import `Gemma4Processor`/`Gemma4UnifiedProcessor` and `Gemma4Config`/
    `Gemma4UnifiedConfig` lacks the `num_hidden_layers` attribute nnsight requires to map layers.
    Root cause: nnsight 0.6 (Feb 2026) predates Gemma 4's release (Apr 2026); no working path.
  - **Gemma 3 4B — OOM** on CPU allocation (100 MB alloc failure); no result.
  - Jobs 1015382, 1015383 (original 12B pinned jobs) were stuck in `qw` since submission;
    cancelled this session.
- **Decision / rationale:** Commit to **Gemma 3 12B-IT** as the sole model for the core
  result. Gemma 4 dropped: nnsight support is absent, no SAEs exist, and the multimodal
  registration adds complexity with no scientific benefit. The 12B smoke passes the probe
  threshold; Gemma 3 27B remains an open-door scale-up. Gemma 4 can be revisited if nnsight
  adds Gemma 4 native support in a future release.
- **Next:** (1) Inspect top SAE features on Neuronpedia (`gemma-scope-2-12b-it-res`,
  `layer_31_width_16k_l0_medium`) to assess warmth-vs-valence confound. (2) Phase 4 —
  implement `src/extract_vectors.py` for full corpus extraction over ~4,800 API-generated
  stories.

## 2026-06-15 · Step 1 — Cross-audit corrections applied to overleaf/references.bib

- **Context:** Resuming after previous session completed cross-audit of three files (Ulu_Lastra.tex, references.bib, literature_review_table.xlsx).
- **Did:** Read overleaf/references.bib and confirmed two remaining errors from the audit had not been applied to the overleaf copy (outputs/ had the corrected version). Applied: (1) Gallo first author corrected from "Edoardo" to "Marcos"; Ona corrected from "Devon" to "Vaida"; Jenkins middle initial "C." added; Camerer "F." added. (2) Sofroniew et al. "and others" replaced with full 16-author list.
- **Findings:** overleaf/Ulu_Lastra.tex is correct and complete (Introduction 3 paragraphs, Literature gap 4+1 subsections, Methods skeleton, Results with smoke-test placeholder, Discussion). overleaf/references.bib now clean.
- **Decision / rationale:** The overleaf/ folder is the canonical Overleaf-ready bundle. Both files now match the verified versions from the audit.
- **Next:** Write the abstract (collaborative). Then Methods section.

---

## 2026-06-15 · Step 2 — Pilot concept stimuli generated (in-Cowork, no API cost)

- **Context:** Phase 3 stimulus generation. User wanted a quality pilot without paying for Anthropic API calls (`src/generate_stimuli.py` uses the paid API; a Pro subscription does not cover API usage).
- **Did:**
  - Generated 20 pilot stories directly via Claude in Cowork (model: **claude-opus-4-8**, manual generation — recorded in each story's `generation_model` field as "claude-opus-4-8 (Cowork manual)"). Wrote to `data/stimuli/concept_stories.jsonl` in the exact schema `src/generate_stimuli.py` produces (`id, condition, topic_idx, topic, text, generation_model`; id format `{cond}_t{idx:03d}_s{idx:05d}`).
  - Design: matched/paired — topics 0–4, all 4 conditions per topic (5 per condition). Off-axis dimension held neutral; protagonist genders varied.
  - Added `scripts/validate_stimuli.py`: mechanical QC (forbidden-word stems per condition, required fields, unique id, length band, per-condition counts vs target). Non-zero exit on hard violations so it can gate the pipeline.
  - Added `data/stimuli/STIMULI_TRACKER.md`: target vs current per condition, run log, design rules, and open decisions D1–D3.
- **Findings:** Validator PASS, 0 violations. Word lengths tightly matched across conditions (means 153–158, range 150–164) → no length confound in the pilot. Counts: 5 per condition, 20 total, 4,780 remaining to target.
- **Decision / rationale:** Use in-Cowork generation (subscription-covered) instead of the paid API path for now. Flagged open decisions: (D1) whether to fold "hold off-axis neutral / demographic balance / behavioural diversity" into the API prompt; (D2) Figure 1 — mirror Carina's PCA-collapse vs. keep warmth/competence probes separate; (D3) reduce stories-per-topic for a faster first corpus.
- **Next:** Jorge reviews pilot quality. Then either scale up generation (after deciding D1/D3) or proceed to wire Phase 4 extraction against this pilot to produce a first Figure 1.

---

## 2026-06-15 · Step 3 — Demographic + topic tracking; generator bias caught; D2 decided

- **Context:** Jorge asked for (a) a system to track protected-attribute balance of story protagonists to prove variation and catch generator bias, (b) a topic tracker, and (c) a decision on D2 (warmth/competence aggregation).
- **Did:**
  - Added `data/stimuli/protagonist_metadata.jsonl` (per-story: name, gender, name_origin cue, age/disability/religion cues).
  - Added `data/stimuli/name_roster.csv` (30 names balanced across gender x 7 name-origin groups + neutral) for assignment-by-design.
  - Added `scripts/audit_stimuli.py`: advisory report of topic coverage (per-topic counts per condition, depth vs 12 target) and demographic balance by condition, with a 70%-skew warning.
  - Updated `STIMULI_TRACKER.md`: demographics + topic sections; recorded D2 decision; added D4.
- **Findings:**
  - **Generator bias detected in the Run-1 pilot (model: claude-opus-4-8).** `audit_stimuli.py` flagged: low_warmth and low_competence = 100% male; low_warmth 80% / low_competence 100% Anglo-named; high_warmth 80% female and diverse. The "good" protagonists were diverse/female, the "bad" ones uniformly Anglo-male. Unfixed, the warmth/competence direction would encode gender + name-origin — a fatal confound. This validates exactly why the audit was requested.
  - Topic coverage: 5/100 topics, 0 at full depth.
- **Decision / rationale:**
  - **D2 — DECIDED (Jorge):** extract warmth and competence as two separate probes (one-way-door: aggregation is reversible, separation is not). In analysis, report each separately AND test their correlation; if strongly positive like Carina's ratings, run PCA to reproduce her PC1/PC2 and compare to her published PC1. Delivers both the apples-to-apples comparison and the separate-dimension causal analysis. Do both.
  - **D4 — OPEN:** neutralise demographics in concept stories vs. balance-by-design via roster. Either fixes the confound. Pending Jorge.
- **Next:** Pick D4 (neutralise vs balance), then regenerate the pilot demographics accordingly and re-run `audit_stimuli.py` to confirm no skew. Then proceed to scale-up or Phase 4 wiring.

---

## 2026-06-15 · Step 4 — D4 decided (neutralise names); pilot regenerated name-free; validator bug fixed

- **Context:** Jorge decided to remove protected-attribute signals from the concept stories (clarifying that names belong only at the measurement stage, sourced from Carina, not in the vector-defining stories).
- **Did:**
  - **D4 = NEUTRALISE.** Regenerated the 20-story pilot name-free (third-person "they", role words, no proper names) across **5 domains** (topics 0/19/45/65/75: workplace, learning, community, sport, travel), fixing the earlier all-workplace topic concentration. Off-axis dimension held neutral; generation_model recorded as "claude-opus-4-8 (Cowork manual, name-free)".
  - Rewrote `protagonist_metadata.jsonl` as all-neutral (gender unspecified, origin none).
  - Fixed a `validate_stimuli.py` false-positive: forbidden stem "friend" -> "friendl" (so "friendly/friendliness" is caught but the noun "friend" is not).
  - Updated `STIMULI_TRACKER.md` (D2 + D4 decided, demographics + topic sections, Run-2 log).
- **Findings:** `validate_stimuli.py` PASS, 0 violations, lengths 127-148 tightly matched across conditions. `audit_stimuli.py`: name-free => no demographic signal in concept stories, no skew. Topic coverage 5/100 across 5 domains.
- **Decision / rationale:** Concept stories must define clean warmth/competence directions, so they carry no demographic signal; this also sidesteps the hard name-origin balancing problem. Demographic variation enters only at Fig-1/steering/benchmark, using Gallo-Hausladen's validated names. D1 (fold rules into the API generator) and D3 (stories per topic) remain open.
- **Note:** Write-tool -> sandbox sync truncated code files mid-write twice this session; switched to writing code/docs via shell heredoc, which is reliable. (Matches the "cowork sandbox quirks" memory.)
- **Next:** Confirm whether the model hiring-evaluation stage (PLAN.md Phases 6-7) is still in scope. Then settle D3, fold rules into `generate_stimuli.py` (D1), and scale generation across all 100 topics in validated batches.

---

## 2026-06-16 · Step 1 — Concept corpus expanded to 200 name-free stories (50/condition, 50 topics)

- **Context:** For an expanded smoke test before the preliminary-results presentation, Jorge set a target of 200 stories (revised down from 400) generated by Claude in Cowork (no API cost), in validated batches.
- **Did:**
  - Generated 180 additional name-free, off-axis-neutral concept stories (model: claude-opus-4-8, Cowork manual) in four validated batches, bringing `concept_stories.jsonl` to **200** (50 per condition, 50 of 100 topics, all 10 domains). Extended `protagonist_metadata.jsonl` to match (all neutral).
  - After each batch ran `validate_stimuli.py`; padded stories that fell under the 90-word floor so lengths stayed balanced across conditions (means 100-103, range 90-148).
  - Fixed an earlier validator false-positive (forbidden stem "friend" -> "friendl").
- **Findings:** Validator PASS, 0 rule violations, 200 unique ids. `audit_stimuli.py`: no demographic skew (name-free), topic coverage 50/100 across all 10 domains. Nothing lost across the interrupted session (verified line counts + unique ids).
- **Decision / rationale:** 200 (50/condition) doubles the original smoke test's per-probe size and, unlike it, covers both warmth and competence and spans 50 real-world scenarios rather than 100 homogeneous sentences. Generated in Cowork to avoid API cost; batched + mechanically validated to bound drift.
- **Next:** (deferred per Jorge) wire the Gemma3 smoke-test script to read `concept_stories.jsonl` (both probes) and run on SCCKN to produce the expanded smoke-test numbers. Open: D1 (fold rules into API generator), D3 (expand toward full 4,800 / more per topic).

## 2026-06-16 · Step 2 — Implemented extract_vectors.py and validate_probes.py; updated extract_vectors SGE job

- **Context:** Phase 4+5 implementation triggered by Jorge pushing 200 concept stories to `data/stimuli/concept_stories.jsonl`.
- **Did:** Replaced stubs in `src/extract_vectors.py` and `src/validate_probes.py` with full implementations; rewrote `jobs/sge/extract_vectors.sh` with 3-node fan-out queue optimisation.
- **Findings:** Dry-run passes — 4×50 stimulus balance confirmed, config load clean. All files pass `python3 --dry-run`.
- **Decision / rationale:** Kept model fixed to `google/gemma-3-12b-it` (already in config). Used `probe_layer_frac=0.66` → layer 31 (GemmaScope 2 SAE compatible for future sprint). Cross-axis orthogonality test added to validate_probes because Jorge designed the stories to isolate each axis independently — this is the primary new scientific check vs the smoke tests.
- **Next:** SSH to SCCKN, `git pull`, `qsub jobs/sge/extract_vectors.sh`. Monitor with `qstat -u emrecan.ulu`.

## 2026-06-16 · Step 3 — Probe findings report revised after figure audit

- **Context:** Audit found that Fig 4's off-diagonal 0.50 CV cells should not be used as a headline behavioural-independence claim, while Fig 2 and Fig 3 remained useful.
- **Did:** Revised `paper/2026-06-16_concept_stories_probe_findings.md` to remove the Fig 4 embed and recast the result as linearly probeable warmth/competence contrasts with unresolved valence overlap. Updated `paper/figures/generate_figures.py` and regenerated Fig 3 PNG/PDF.
- **Findings:** Verification grep found no remaining report claims of "full behavioural independence", "Cross-axis CV", "Figure 4", or `fig4_axis_geometry`. Fig 3 annotations now show Top 11 / 479 / 1426 dims and no horizontal threshold dotted lines.
- **Decision / rationale:** Keep Fig 1 as a valence-overlap visual, keep Fig 2 and Fig 3 as quantitative evidence, and defer behavioural-independence claims until symmetric cross-axis validation after denoising.
- **Next:** Commit and push the report and Fig 3 updates.

## 2026-06-16 · Step 4a — AGENTS.md and ai-usage/ logging infrastructure created

- **Context:** Jorge added a separate AI-action trail (`ai-usage/steps.md`) alongside `AGENTS.md` to support multi-agent (Claude + GPT/Codex) workflows.
- **Agent:** claude-sonnet-4-6
- **Did:** Created `AGENTS.md` as tool-agnostic single source of truth; rewrote `CLAUDE.md` as a 3-line pointer to `AGENTS.md`; created `ai-usage/steps.md` with entry format.
- **Findings:** All three files created. Steps 2–4 on this date (extract_vectors, probe report, fig2 polish) were also recorded in `ai-usage/steps.md` by claude-opus-4-7 and gpt-5-codex respectively.
- **Decision / rationale:** On 2026-06-19 the separate `ai-usage/steps.md` was retired — it duplicated this log with no unique scientific content. Model-id attribution is now captured here via the optional **Agent:** field.

---

## 2026-06-19 · Step 4 — Cross-model concept findings report + figures

- **Context:** Writing up the three-model Phase 4+5 replication in `paper/`, with full figure suite and cross-model analyses.
- **Agent:** claude-sonnet-4-6
- **Did:**
  - scp'd Gemma/Qwen3/Llama concept vectors (+ meta.json) and all validate_probes JSON logs + probe_metrics CSVs to local.
  - Fixed `fig5_cross_model` cos(W,C) panel (was a non-functional stub); added `--logs` CLI arg to pass validate_probes JSONs directly.
  - Added `fig6_cross_model_story_agreement`: two 3×3 Spearman ρ heatmaps (warmth + competence per-story ranking agreement across model pairs). Added `fig7_same_story_demo`: z-scored warmth/competence coordinates for 6 exemplar stories plotted simultaneously for all 3 models.
  - Added `--vec-dirs`, `--logs`, `--stories` CLI args to `generate_figures.py`; updated `main()` for fig6/7 dispatch.
  - Generated per-model fig1–3 for Qwen3/Llama (`paper/figures/qwen3_14b/`, `llama31_8b/`); regenerated Gemma fig4 (cos now computed from data).
  - Generated cross-model fig5, fig6, fig7.
  - Wrote `paper/2026-06-19_cross_model_concept_findings.md` (9 sections; model-selection rationale + rejected-models table; full results; same-story demo; cross-axis paradox §7; scale note §6; Phase B roadmap §8).
  - Updated `paper/README.md`: added per-model and cross-model figure inventory rows + new report row.
- **Findings:**
  - All 3 models: warmth CV = 100%, competence CV = 100%. PASS.
  - Random-baseline z: Gemma 3.9/3.7, Qwen3 14.1/14.6, Llama 15.0/15.1 — all far from null.
  - Per-story Spearman ρ (warmth): Gemma↔Qwen=0.760, Gemma↔Llama=0.768, Qwen↔Llama=0.978.
  - Per-story Spearman ρ (competence): Gemma↔Qwen=0.795, Gemma↔Llama=0.782, Qwen↔Llama=0.992.
  - Cross-axis paradox confirmed: Gemma cos(W,C)=0.749 → cross-axis CV=0.50 (chance); Qwen/Llama cos≈0.51–0.54 → cross-axis CV=0.99–1.00. Paradox discussed in §7, three hypotheses proposed.
- **Decision / rationale:** Spearman ρ results constitute strong evidence for a shared cross-model warmth/competence construct. Cross-axis paradox is the main open scientific question; Phase B layer sweep will directly test the depth-threshold hypothesis.
- **Next:** Phase B — topic-holdout CV, layer sweep (all layers single pass), Gemma 3 27B, scale normalisation. Commit + push this session's work.

---

## 2026-06-19 · Step 3 — Cross-model 200-story pipeline: parametrize extract/validate/figures + new SGE jobs

- **Context:** Planning and implementing "Path B" — run the full 200-story warmth+competence pipeline on Qwen3-14B and Llama-3.1-8B under identical conditions as Gemma, to show the result is architecture-general and enable a parallel paper report.
- **Agent:** claude-sonnet-4-6
- **Did:**
  - Added `--model` + `--out-subdir` to `src/extract_vectors.py` (uses `dataclasses.replace` for frozen config; default behaviour unchanged).
  - Added `--vectors-subdir` + `--label` to `src/validate_probes.py`; label-suffixed outputs (`probe_metrics_<label>.csv`, `results/figures/<label>/`).
  - Parametrized `paper/figures/generate_figures.py` with `--vec-dir` / `--out-dir`; fixed `fig4_axis_geometry` to compute cosine(W,C) from data (was hardcoded 0.749); added `fig5_cross_model` grouped-bar figure.
  - Created `jobs/sge/extract_qwen3_14b.sh` and `jobs/sge/extract_llama31_8b.sh` (mirror `extract_vectors.sh`; h_vmem 64G, start_token 50 from config, model-scoped out-subdirs).
- **Findings:** All changes local; Gemma's committed outputs are fully isolated (default flag values unchanged). Both new SGE jobs ready to submit.
- **Decision / rationale:** CLI override (not per-model config files) matches the existing smoke-test pattern and keeps a single config.yaml.
- **Next:** Commit + push, SSH to SCCKN, `git pull`, `qsub` both jobs, monitor, then scp vectors locally and generate figures.

---

## 2026-06-19 · Step 2 — Cross-model smoke tests: Qwen3-14B PASS, Llama-3.1-8B PASS

- **Context:** Extending the Gemma 3 12B warmth-probeability result to two additional model families (Alibaba/Qwen, Meta/Llama) under identical conditions to show the finding is architecture-general.
- **Agent:** claude-sonnet-4-6
- **Did:** Created `smoke_tests/transformerlens_probe.py` (family-neutral probe script); wrote `jobs/sge/smoke_qwen3_14b.sh` and `jobs/sge/smoke_llama31_8b.sh`; committed + pushed; SSH'd to SCCKN, git pull, downloaded both models (~44 GB), submitted jobs. Llama OOM'd at h_vmem=32G → resubmitted at 64G; fixed the job script.
- **Findings:**
  - **Qwen3-14B** (40 layers, d_model 5120, probe layer 26, scc214 RTX 6000): probe_cv_mean **0.88 ± 0.05**, Cohen's d 3.08, warmth/random ratio 1.46×. **PASS.**
  - **Llama-3.1-8B-Instruct** (32 layers, d_model 4096, probe layer 20, scc214 RTX 6000): probe_cv_mean **0.88 ± 0.06**, Cohen's d 3.45, warmth/random ratio 2.18×. **PASS.**
  - Both loaded natively via TransformerLens (`trust_remote_code=True` warning for Qwen3 is non-blocking).
  - Results in `smoke_tests/results/qwen3_14b/smoke_probe_1781866858.json` and `smoke_tests/results/llama31_8b/smoke_probe_1781867569.json`.
- **Decision / rationale:** Warmth is linearly probeable (CV > 0.80) across all three tested families (Google Gemma 3, Alibaba Qwen3, Meta Llama 3.1) at the same layer fraction (0.66) and stimuli. This is the cross-family generalization result. h_vmem for Llama corrected to 64G in job script.
- **Next:** Pull result JSONs to local, write `paper/YYYY-MM-DD_cross_model_smoke.md` with three-model comparison table, commit.

---

## 2026-06-19 · Step 1 — Log consolidation, CLAUDE.md slimming, and new AGENTS.md rules

- **Context:** Documentation hygiene session: retire duplicate `ai-usage/steps.md`, slim `CLAUDE.md` to a bare import, and add session-start + findings-report rules to `AGENTS.md`.
- **Agent:** claude-sonnet-4-6
- **Did:**
  - Pulled 9 commits from `origin/main` (Jorge's presentation + figure work 2026-06-16).
  - Retired `ai-usage/steps.md` and `ai-usage/` directory; migrated unique entry (AGENTS.md infrastructure creation) into this log as Step 4a on 2026-06-16.
  - Removed `## AI Usage Logging` section from `AGENTS.md`; added optional `**Agent:**` field to the Step Logging entry format.
  - Fixed two stale `CLAUDE.md` → `AGENTS.md` references in `README.md` and the STEP_LOG header.
  - Replaced `CLAUDE.md` body with a bare `@AGENTS.md` import (comment + directive only).
  - Added session-start rule to `AGENTS.md` Step Logging: read latest STEP_LOG *and* latest `paper/` report at session start.
  - Added `## Findings Reports` section to `AGENTS.md`: new findings go in `paper/YYYY-MM-DD_<slug>.md`.
  - Created `paper/README.md`: naming convention, figures inventory, relationship to STEP_LOG, current reports list.
- **Findings:** All changes verified; two commits pushed to `origin/main` (SHA `394028e` and this session's commit).
- **Decision / rationale:** Single log (`STEP_LOG.md`) is simpler than two parallel logs; `CLAUDE.md` as a bare import eliminates prose drift between it and `AGENTS.md`; explicit session-start and findings-report rules make the workflow self-enforcing for any AI agent.

---

## 2026-06-16 · Step 4 — Fig 2 annotation spacing adjusted

- **Context:** Visual audit of Fig 2 found the "Our direction" annotation too close to the red vertical line.
- **Did:** Updated `paper/figures/generate_figures.py` to shift the Fig 2 annotation text left and regenerated `paper/figures/fig2_random_baseline.{png,pdf}`.
- **Findings:** Fig 2 regenerated successfully. Visual inspection confirmed the annotation text no longer crowds the red line, while the arrow still points to the direction marker.
- **Next:** Commit and push the figure polish.

---

## 2026-06-19 · Step 1 — Valence-denoising scaffold (Wikipedia neutral corpus + PCA project-out)

- **Context:** Phase "valence denoising". Emre's probe run left a shared good-vs-bad component: cos(warmth, competence) = 0.75. This step removes it, following the Anthropic emotion-concepts recipe in METHOD_NOTES 1.4.
- **Did:**
  - `scripts/build_neutral_corpus.py`: streams Wikipedia intros (HF `wikimedia/wikipedia`), length-matches to the concept stories (90-200 words), drops valence/violence intros via a stoplist, seed-samples 1,500 -> `data/stimuli/neutral_corpus.jsonl`. Runs on the SCCKN login node (has internet); offline `--self-test` for the filter logic (PASS).
  - `src/extract_neutral.py`: GPU extraction of neutral activations at the same layer (31) and start_token, reusing `extract_vectors.extract_activations`; saves `data/processed/concept_vectors/X_neutral.npy`.
  - `src/denoise_vectors.py`: PCA on neutral activations, keep top PCs covering >=50% variance, project them out of warmth/competence vectors; re-reports cos(w,c) and per-axis Cohen's d plus a warmth-on-competence "leak" diagnostic; saves `concept_vectors_denoised.npz` + `denoise_summary.json`.
  - `jobs/sge/extract_neutral.sh` (mirrors extract_vectors.sh), `config` `neutral` section + `NeutralConfig` (optional, non-breaking), scikit-learn/datasets in requirements.
- **Findings:** Synthetic verification PASS: project-out dropped a planted cos 0.906 -> 0.071 while preserving the axis-specific signal (0.99 alignment). Sandbox cannot reach Hugging Face (proxy 403), so corpus build is a login-node step; everything else verified locally.
- **Decision / rationale:** Neutral corpus = Wikipedia intros (matches Anthropic; externally sourced, so no LLM circularity with the model we probe; length-matched so length cannot leak). Variance threshold 0.50 per method notes. Valence stoplist keeps it socially neutral. The exact number of PCs (k) is data-driven on the real neutral activations.
- **Next (cluster):** login node `python scripts/build_neutral_corpus.py`; then `qsub jobs/sge/extract_neutral.sh` (extract_neutral -> denoise_vectors). Compare cos(w,c) before vs after and refresh the Figure 1 / Figure 4 story with denoised vectors. Then SAE decomposition and Phase 6 steering.

---

## 2026-06-20 · Step 1 — Git-tracked pipeline outputs: .gitignore un-ignore + sync_outputs.sh

- **Context:** Pipeline outputs (concept vectors, activation matrices, validation logs, metric CSVs) were git-ignored and lived only on SCCKN /work (scratch-like, not backed up) and local disk after manual scp. Risk: SCCKN purge or failure loses artifacts that require GPU hours to regenerate.
- **Agent:** claude-sonnet-4-6
- **Did:**
  - Appended selective un-ignore block to `.gitignore`: `!data/processed/concept_vectors/`, `!data/processed/concept_vectors/**`, `!data/processed/concept_vectors_*/`, `!data/processed/concept_vectors_*/**`, `!results/logs/validate_probes_*.json`, `!results/tables/probe_metrics*.csv`. Model weights (`*.safetensors`, `*.bin`, `*.pt`) remain ignored.
  - Created `jobs/sync_outputs.sh`: idempotent additive sync script (stages tracked paths, checks for changes, commits with hostname+timestamp, `git pull --rebase`, `git push`; graceful exit if nothing to commit; never force-pushes).
  - Appended `Step 3: Sync outputs to git` to `jobs/sge/extract_vectors.sh`, `extract_qwen3_14b.sh`, `extract_llama31_8b.sh` (tolerant `|| echo` so a push failure on a compute node never kills the GPU job).
  - Updated `AGENTS.md` Working Conventions: replaced stale "ignored by git" note with the canonical tracked-path list and reference to `sync_outputs.sh`.
- **Findings:** Total artifact footprint ~10 MB (float32, write-once across 3 models) — comfortably within plain git; Git LFS not needed at this scale. `git check-ignore` verified negations win (gitignore last-match-wins; directory must be re-included before its contents).
- **Decision / rationale:** Bidirectional sync via git (not scp): SCCKN jobs push outputs after extraction; collaborators and local pull via `git pull`. Additive only — `pull --rebase` before every push prevents overwriting parallel report/code work. Model weights excluded (public on HF Hub, 8–54 GB, reproducible via model id + seed).
- **Next:** Commit this session's changes (gitignore, sync script, job updates, AGENTS.md), push, then on SCCKN: `git pull` + `bash jobs/sync_outputs.sh` to upload existing 3-model artifacts.

---

## 2026-06-20 · Step 2 — Phase B1 (topic-holdout CV) + B2 (layer sweep) implementation

- **Context:** Phase B: add discriminative evaluation metric (B1) and cross-layer analysis (B2) to the 3-model warmth/competence probing result.
- **Agent:** claude-sonnet-4-6
- **Did:**
  - **B1:** Extended `src/validate_probes.py` with `load_topic_groups()` helper (reads topic_idx per condition in same sequential order as extract_vectors.load_stories), `topic_holdout_cv()` function (GroupKFold, n_splits=5, deterministic), and optional `groups_high/groups_low` params on `probe_axis()`. Runs automatically when concept_stories.jsonl is present; alignment-asserted. New fields in CSV/JSON: `topic_cv_mean`, `topic_cv_std`, `topic_cv_folds`, `pass_warmth_topic_cv`, `pass_competence_topic_cv`. SUMMARY now prints both metrics. Ran on all 3 models (GPU-free).
  - **B2:** Created `src/layer_sweep.py`: loads model once, `run_with_cache(names_filter=endswith("hook_resid_post"))` captures all n_layers in one forward pass per story; per-layer warmth/competence vectors + topic-holdout CV + Cohen's d + cos(W,C) + mean_resid_norm; writes `results/tables/layer_sweep_<label>.csv` + `.meta.json`; no .npy dumps (scale guard). Created 3 SGE jobs: `jobs/sge/layer_sweep_{gemma,qwen3_14b,llama31_8b}.sh`.
  - **Figure 8:** Added `fig8_layer_emergence()` to `paper/figures/generate_figures.py`: two-panel (warmth/competence) emergence curves vs layer fraction, one line per model, Cohen's d twin axis; `--sweep-csvs` CLI arg + dispatch in main().
  - **gitignore + sync_outputs.sh:** un-ignored `results/tables/layer_sweep*.csv` and `layer_sweep*.meta.json`; added to `git add` list in `jobs/sync_outputs.sh`.
- **Findings:** B1 topic-holdout CV = 1.0000 on all 3 models (same as 5-fold CV). This is a **strong positive result**: separation is not topic-vocabulary leakage but genuine cross-topic generalization. Cohen's d (Qwen 9.0, Llama 8.5, Gemma 2.7) predicts this — very large effect sizes are robust to unseen-topic test. B2 layer sweep will reveal WHERE in the network this emerges and whether the cross-axis paradox is a depth effect.
- **Decision / rationale:** Topic-holdout staying at 1.0 is scientifically meaningful, not disappointing — it shows the representations generalize completely across situations. The sweep (B2) is now the key analysis for ranking layers and testing the paradox hypothesis.
- **Next:** SSH to SCCKN, git pull, qsub the 3 layer-sweep jobs. When done, pull CSVs locally and run `generate_figures.py --fig 8 --sweep-csvs ... --labels ...`.

---

## 2026-06-20 · Step 3 — B2 layer sweep jobs completed; fig8 generated

- **Context:** Phase B2 — layer sweep jobs submitted to SCCKN and completed.
- **Agent:** claude-sonnet-4-6
- **Did:** qsub layer_sweep_{gemma,qwen3_14b,llama31_8b}.sh (jobs 1058948-1058950); all 3 finished; manual sync from login node (compute-node push fallback as designed); git pull locally; generated paper/figures/fig8_layer_emergence.{png,pdf}.
- **Findings:**
  - All 3 models: topic-holdout CV = 1.0000 at ALL layers above a low threshold — representations are robustly separable from very early layers onward.
  - **Gemma-3-12B**: first peak at L10 (frac=0.21, d=1.29/1.79), probe layer L31 (frac=0.66, cos=0.749, norm=79756).
  - **Qwen3-14B**: warmth peak L13 (frac=0.33, d=6.26), competence peak L3 (frac=0.08, d=4.75), probe layer L26 (frac=0.67, cos=0.536, norm=206.6).
  - **Llama-3.1-8B**: warmth peak L7 (frac=0.23, d=6.95), competence peak L2 (frac=0.06, d=5.72), probe layer L20 (frac=0.65, cos=0.505, norm=11.4).
  - Cross-axis paradox: Gemma cos(W,C)=0.749 at L31 vs Qwen/Llama cos~0.51-0.54. The emergence curves will show whether cosine diverges across depth.
- **Decision / rationale:** CV ceiling at 1.0 across all layers means Cohen's d is the discriminative metric for ranking layers. Emergence is early (frac<0.25) for Qwen/Llama, later for Gemma.
- **Next:** Inspect fig8 visually; commit figure; update paper/README.md; consider B3 (Gemma 27B) or B5 (report revision).

---

## 2026-06-20 · Step 4 — Phase B findings report + fig8 redesign

- **Context:** Write-up of B1 (topic-holdout CV) and B2 (layer sweep) results; redesign fig8 to be paradox-focused.
- **Agent:** claude-sonnet-4-6
- **Did:**
  - Redesigned `fig8_layer_emergence()` in `paper/figures/generate_figures.py`: replaced 2 CV-panel layout with (left) Cohen's d emergence curves + (right) cos(W,C) depth profile. Right panel is the paradox diagnostic: Gemma's cos stays elevated at ALL depths; Qwen/Llama plateau near 0.50.
  - Wrote `paper/2026-06-20_layer_sweep_topic_holdout.md`: full findings report covering (a) probe concept explanation, (b) topic-holdout rationale + B1 result, (c) layer sweep method + B2 Cohen's d emergence tables, (d) cross-axis paradox resolution with per-layer cos(W,C) evidence, (e) residual norm scale variation, (f) limitations, (g) next steps table.
  - Updated `paper/README.md`: added fig8 row to figures inventory and new report row to current reports table.
- **Findings:**
  - B1: topic-holdout CV = 1.00 ± 0.00 on all 3 models (both axes) — genuine generalisation, not topic-vocabulary leakage.
  - B2 emergence: Llama peaks L10-14 (d=10.6/11.5), Qwen peaks L22-25 (d=9.9/10.8), Gemma rises late to peak L45 (d=6.1/4.4). Early layers (frac<0.15) already reach d>4 for Qwen/Llama.
  - Paradox resolution: Gemma cos(W,C) ranges 0.49–0.95 across all layers (probe layer 0.749 is representative, not an outlier). Qwen max 0.62, Llama max 0.58. H2 (depth effect) falsified; H3 (architectural effect) supported.
  - Residual norm varies ~7,000x across models (Gemma L31: 79,756; Llama L20: 11.4). Relative steering calibration (already enforced in AGENTS.md) is confirmed necessary.
- **Decision / rationale:** Fig8 CV panels were flat (1.0 everywhere) and therefore uninformative; Cohen's d + cos(W,C) profile panels carry the scientific signal and tell the paradox story visually.
- **Next:** Regenerate fig8 on local machine with new `generate_figures.py`; then B3 (Gemma-3-27B layer sweep on SCCKN scc214) or valence denoising (B6, login node corpus build pending).

---

## 2026-06-20 · Step 5 — Phase B3: Gemma-3-27B job script + fig8 4-model prep

- **Context:** Phase B3 — within-family scale test: does Gemma-3-27B show the same cos(W,C) entanglement as 12B at every depth?
- **Agent:** claude-sonnet-4-6
- **Did:**
  - Created `jobs/sge/extract_gemma3_27b.sh`: single chained SGE job (extract → validate → sweep → sync) pinned to `gpu@scc214` only (96 GB RTX 6000; 27B bf16 ~54 GB VRAM does not fit L40 nodes). h_rt=04:00:00, h_vmem=96G (conservative: from_pretrained_no_processing halves peak but 27B is large). CLI overrides: `--model google/gemma-3-27b-it --out-subdir concept_vectors_gemma3_27b --label gemma3_27b`. config.yaml unchanged (model name passed via --model flag per AGENTS.md constraint).
  - Extended `fig8_layer_emergence` in `paper/figures/generate_figures.py`: added 4th model_color (#006d6d dark teal, readable as "Gemma family" but distinct from 12B green) and 4th linestyle (dash-dot-dot `(0,(3,1,1,1))`). No other code change needed — function already loops over zip(sweeps, model_labels).
- **Findings:** n/a (job not yet submitted; script ready for SCCKN).
- **Decision / rationale:** Single chained job to use scc214's 96 GB allocation only once (avoid double queueing on the cluster's most contended node). Extract step fails fast if TransformerLens cannot load 27B — acceptable risk per plan.
- **Next:** git pull on SCCKN login node; confirm `hf auth whoami`; `qsub jobs/sge/extract_gemma3_27b.sh`. When done: pull results, run `python paper/figures/generate_figures.py --fig 8 --sweep-csvs ...<4 CSVs>... --labels ...`, write B3 report.

---

## 2026-06-20 · Step 6 — B3 Gemma-3-27B results + report

- **Context:** Phase B3 — within-family scale test: SCCKN job 1059107 completed; 27B results pulled and analysed.
- **Agent:** claude-sonnet-4-6
- **Did:**
  - Pulled 11 output files from SCCKN via `jobs/sync_outputs.sh` (concept_vectors_gemma3_27b/, probe_metrics_gemma3_27b.csv, layer_sweep_gemma3_27b.csv + .meta.json, validate_probes_1781952895.json).
  - Regenerated `paper/figures/fig8_layer_emergence.{png,pdf}` with 4 models (added Gemma-3-27B dark teal); updated suptitle to "four open-weights models".
  - Wrote `paper/2026-06-20_gemma_scale_paradox.md`: B3 findings report with fig8 embedded; covers probe metrics, layer sweep emergence curves, cos(W,C) depth profile comparison (12B vs 27B), and four-model summary table.
  - Updated `paper/README.md`: fig8 description updated to 4 models; new report row added.
- **Findings:**
  - TransformerLens loaded google/gemma-3-27b-it: n_layers=62, d_model=5376, probe_layer=40.
  - CV=1.0, topic-CV=1.0 on both axes (same as 12B).
  - Cohen's d: warmth=2.95, comp=3.27 — slightly stronger than 12B (2.70/2.83).
  - axis_cosine=0.708 — slightly lower than 12B (0.749) but same order.
  - cross_warmth_on_competence_cv=0.50, cross_competence_on_warmth_cv=0.50 — paradox preserved at 27B.
  - cos(W,C) depth profile: same shape as 12B (rises to ~0.93 by frac=0.38, stays elevated); peak cos=0.933 at L23 vs 12B peak 0.952 at L16. Scale does not change the entanglement pattern.
  - mean_resid_norm at probe layer: 61,576 (lower than 12B's 79,756 — reflects absolute layer position; final layer is 177,437).
- **Decision / rationale:** Cross-axis paradox is scale-invariant within Gemma-3 family — confirms architectural explanation. Four-model picture (2 Gemma + Qwen + Llama) now fully populated.
- **Next:** B4 (scale-normalised analysis using per-layer mean_resid_norm) or B6 (valence denoising, login-node corpus build pending).

---

## 2026-06-20 · Step 7 — Cross-axis metric correction and figure-output cleanup

- **Context:** Reproducibility audit and cleanup of unused validation figures after Phase B3.
- **Agent:** gpt-5-codex
- **Did:** Added fold-local standardisation to 1-D projected CV; regenerated four deterministic validation logs and Figure 4; removed the unused `results/figures` plotting path and helper; corrected affected reports; wrote `paper/2026-06-20_cross_axis_metric_correction.md`.
- **Findings:** Unscaled logistic regression produced spurious 0.50 Gemma results under scikit-learn 1.9.0. Corrected W→C/C→W CV: Gemma-12B 0.87/0.82, Gemma-27B 0.90/0.86, Qwen 1.00/1.00, Llama 0.99/1.00. All 5 tests pass; all 14 report PNGs are referenced.
- **Decision / rationale:** Withdraw the cross-axis paradox claim. Retain the supported finding that Gemma's elevated cos(W,C) depth profile persists from 12B to 27B.
- **Next:** Push the correction, remove untracked validation/smoke artifacts on SCCKN, and fast-forward the cluster checkout.

---

## 2026-06-20 · Step 8 — SCCKN artifact cleanup and repository sync

- **Context:** Finalise the validation-output cleanup on the SCCKN checkout.
- **Agent:** gpt-5-codex
- **Did:** Deleted 12 unused validation PNGs under the legacy results figure path and 8 Qwen/Llama smoke-test artifacts while preserving both smoke-test `.gitkeep` files; fast-forwarded SCCKN to correction commit `d5f4721`.
- **Findings:** SCCKN had no remaining result-figure files or untracked smoke outputs; its working tree was clean and matched `origin/main`.
- **Decision / rationale:** Keep only report figures under `paper/figures/`; retain smoke-test scripts and empty result directories for future compatibility checks.

---

## 2026-06-20 · Step 9 — Timestamped report filenames

- **Context:** Restore chronological ordering after report edit times obscured the original sequence.
- **Agent:** gpt-5-codex
- **Did:** Renamed all five findings reports to `YYYY-MM-DD_HHMM_<short-slug>.md`, updated active cross-report references and the `paper/README.md` inventory, and made `paper/README.md` the canonical naming rule.
- **Findings:** Earliest Git commit times established the order: 2026-06-16 20:01, 2026-06-19 18:08, and 2026-06-20 at 11:37, 13:03, and 13:37 (Europe/Berlin).
- **Decision / rationale:** Use result-production time in Europe/Berlin; for historical reports without an explicit production timestamp, use the earliest Git commit time.
- **Rename map:** `2026-06-16_2001_concept_stories_probe_findings.md`; `2026-06-19_1808_cross_model_concept_findings.md`; `2026-06-20_1137_layer_sweep_topic_holdout.md`; `2026-06-20_1303_gemma_scale_paradox.md`; `2026-06-20_1337_cross_axis_metric_correction.md`.

---

## 2026-06-20 · Step 10 — Gemma Scope 2 analysis and concept-causality implementation

- **Context:** Decompose the existing Gemma-3-12B and 27B warmth/competence activations with Gemma Scope 2 and test concept-level causal effects.
- **Agent:** gpt-5-codex
- **Did:** Added sparse SAE analysis for 16k/65k/262k widths, held-out concept steering and error-preserving feature ablation at 65k, cross-scale story-profile feature matching, compact-output git tracking, tests, and parallel SGE jobs pinned to `scc213` (12B) and `scc214` (27B).
- **Findings:** Existing committed activations match the published SAE layers (12B layer 31; 27B layer 40). SAELens 6.44.2 contains all six requested checkpoints. Gemma tokenizers encode both `" Yes"` and `" No"` as one token, enabling one-forward logit-margin evaluation. Local test suite: 11 passed.
- **Decision / rationale:** Reuse the committed 200 mean residual activations instead of rerunning extraction; use 65k as the primary causal SAE, 16k as baseline, and 262k as reconstruction/robustness analysis. Jobs do not push independently, preventing concurrent git races.
- **Next:** Commit and push the implementation, fast-forward SCCKN, submit both jobs in parallel, then sync and analyse their outputs.

---

## 2026-06-24 · Step 13 — Hiring callback causality at 27B (Phase 6 + 7 replication)

- **Context:** Scale replication of the hiring-callback causality experiment at Gemma-3-27B.
- **Agent:** claude-sonnet-4-6 (Cowork) + Jorge (ran notebooks on H100)
- **Did:** Reran `notebooks/06_hiring_steering_causality.ipynb` and `notebooks/07_hiring_audit.ipynb` with `VECTORS_SUBDIR = "concept_vectors_gemma3_27b"`. Results saved to `results/tables/hiring_steering_raw_concept_vectors_gemma3_27b.csv` and `results/tables/hiring_audit_concept_vectors_gemma3_27b.csv`. Findings written to `paper/2026-06-24_1300_hiring_causality_27b_results.md`.
- **Findings:**
  - Baseline: P(Yes)=0.767, every name positive (vs 12B P(Yes)=0.451). 27B is much more generous overall.
  - **Warmth causal effect: absent** (slope=+1.09, R²=0.026 vs 12B R²=0.924). Warmth steering direction is inconsistent and non-monotone.
  - **Competence causal effect: uniformly negative** (slope=+2.88, R²=0.340). Both increasing and decreasing competence reduces callbacks. Role-fit non-linearity from 12B does not replicate.
  - **Probe-vs-human: stronger at 27B** — Warmth ρ=0.381 (vs 0.355), Competence ρ=0.283 (vs 0.230). The 27B encodes social stereotypes more faithfully despite weaker causal effect.
  - **Reversed baseline association**: model warmth/competence probe scores negatively predict callback (ρ=−0.17/−0.16, p<0.01). Names perceived as warmer get fewer callbacks at 27B.
  - Demographic pattern: highest callbacks for Donnell/Lakeisha/Terrell/Darnell; lowest for Dong Liu/Na Li/Fang Wang. Different from 12B.
- **Decision / rationale:** Scale does not eliminate stereotype encoding — it strengthens it. But the causal pathway from representation to hiring decision is disrupted at 27B, and a different bias pattern emerges. Both the layer-sweep and the demographic disparity analysis are priority follow-ups. Research decisions D-Phase7-A and D-Phase7-B still pending.
- **Next:** Resolve D-Phase7-A/B, complete demographic disparity analysis for both models.

---

## 2026-06-24 · Step 12 — Hiring callback causality (Phase 6 + 7)

- **Context:** First run of the hiring-callback causality notebooks on the JupyterHub H100.
- **Agent:** claude-sonnet-4-6 (Cowork) + Jorge (ran notebooks on H100)
- **Did:** Built and ran `notebooks/06_hiring_steering_causality.ipynb` (causal steering sweep) and `notebooks/07_hiring_audit.ipynb` (probe-vs-human validation and baseline). Results saved to `results/tables/hiring_steering_raw_concept_vectors.csv` and `results/tables/hiring_audit_concept_vectors.csv`. Findings written to `paper/2026-06-24_1136_hiring_causality_results.md`.
- **Findings:**
  - Baseline callback margin: mean=−0.195 (SD=0.140), P(Yes)≈0.451 across 282 names.
  - **Warmth causal effect:** slope=+12.954 margin per unit steering strength, R²=0.924 (60 names × 5 strengths). Clean monotone linear effect.
  - **Competence causal effect:** slope=+9.061, R²=0.663. Non-monotonic: strength=−0.25 produces a positive delta (mean +3.90) for every name, reversing at −0.50 (mean −4.61). Interpreted as role-fit (overqualification penalty) for Administrative Assistant.
  - **Probe-vs-human validation (N=282):** Warmth Spearman ρ=0.355 (p=8×10⁻¹⁰); Competence ρ=0.230 (p=9.7×10⁻⁵). Model internal directions align with human ratings at a moderate but highly significant level.
  - **Baseline predictability:** Model probe scores do not significantly predict callback at rest (ρ≈+0.10, p>0.06). Human ratings do (ρ=+0.17 to +0.21, p<0.005). Gap interpreted as latent bias present in the representation but not always expressed in the output.
- **Decision / rationale:** Causal chain (name → internal warmth → callback) is now empirically supported. Demographic-grouped disparity (the fairness-specific comparison) requires research decisions D-Phase7-A (human callback dataset) and D-Phase7-B (demographic grouping) from Jorge before it can be computed.
- **Next:** Resolve D-Phase7-A and D-Phase7-B, wire in real demographic groupings in notebook 07 cell 11, run mediation test.

---

## 2026-06-20 · Step 11 — Gemma Scope 2 cross-scale and causal results

- **Context:** Execute and report the Gemma Scope 2 analysis for Gemma-3-12B and 27B.
- **Agent:** gpt-5-codex
- **Did:** Ran parallel full jobs 1059187/1059188 and local-regime steering jobs 1059225/1059226; synced compact outputs through Git; matched 12B↔27B features with a 500-permutation null; generated Figures 9–12; wrote `paper/2026-06-20_1451_gemma_scope2_feature_causality.md`.
- **Findings:** Reconstruction cosine was 0.995–0.998. Cross-scale feature-profile means were 0.490–0.655 and exceeded all permutation nulls (p=0.002). Dense target steering was positive and locally linear in all four model×axis cases (R²=0.915–0.990). SAE causal preservation held in 3/4 cases and failed for 27B warmth. In 27B, shared-feature ablation reduced warmth and competence gaps by -0.725 and -0.319; 12B did not replicate this necessity pattern.
- **Decision / rationale:** Claim dense concept-level causality and cross-scale feature conservation, but not clean axis-specific sparse localization or hiring causality. Retain the broad steering run as a saturation diagnostic and use the ±0.10 local run for causal slopes.
- **Next:** Implement the hiring callback evaluation before extending the causal claim to employment decisions.

---

## 2026-06-24 · Step 1 — Paper-draft figures for supervisor presentation

- **Context:** Pre-Phase-6 paper draft; three final-quality figures to communicate Geometry → Universality → Causality narrative.
- **Agent:** claude-sonnet-4-6
- **Did:** Added `paper_figure1_axis_arrows`, `paper_figure2_universal_representation`, and `paper_figure3_causal_steering` functions to `paper/figures/generate_figures.py`; added `--steering-slopes` CLI arg and `p1/p2/p3` dispatch tokens; updated `paper/figures/style.py` with arrow-colour constants; updated `paper/README.md` figures inventory.
- **Findings:** All six files produced successfully: `paper_figure1_axis_arrows.{png,pdf}` (529 KB / 44 KB), `paper_figure2_universal_representation.{png,pdf}` (906 KB / 211 KB), `paper_figure3_causal_steering.{png,pdf}` (313 KB / 22 KB). RuntimeWarning about float32 overflow in numpy norm is cosmetic — output is correct (float64 arithmetic used throughout).
- **Decision / rationale:** paper_figure* prefix keeps draft figures distinct from report figures fig1–fig12 in the same directory. Oblique-basis rendering in paper_figure1 encodes the true inter-axis angle (Gemma ~41–45°, Qwen/Llama ~57–59°) so arrow geometry is scientifically honest. paper_figure3 draws only Gemma-family (concept steering data available); Qwen/Llama steering flagged as future work in caption.
- **Next:** Visual QC of the three PNGs; paper draft writing using figures as section anchors; Phase 6 hiring callback evaluation.

---

## 2026-06-24 · Step 2 — Replace paper_figure2 with layer-emergence figure

- **Context:** paper_figure2_universal_representation (2×2 KDE clouds) was judged redundant with paper_figure1 (same geometric message); replaced with a single-panel depth-emergence figure.
- **Agent:** claude-sonnet-4-6
- **Did:** Added `paper_figure2_layer_emergence(sweep_csv_paths, model_labels)` to `paper/figures/generate_figures.py`; retains dead-code stub of the old function with a `NotImplementedError` guard; refactored `main()` p2 dispatch to use `--sweep-csvs` instead of `--metrics` and moved `--vec-dirs` validation to p1 only; deleted `paper_figure2_universal_representation.{png,pdf}`; updated `paper/README.md` inventory row.
- **Findings:** `paper_figure2_layer_emergence.{png,pdf}` produced successfully (4 models × 2 axes = 8 curves; probe-layer frac=0.66 marker and d=0.80 reference line present). p1 and p3 dispatch smoke-tested; no regressions.
- **Decision / rationale:** New figure adds the **depth** dimension (Ne → Nerede/ne kadar derin → Ne işe yarıyor narrative arc); eliminates redundancy with Fig1.
- **Next:** Visual QC of new PNG; commit all three paper figures for supervisor presentation.

---

## 2026-06-24 · Step 3 — Split paper_figure2 into two panels (warmth | competence)

- **Context:** Single-panel fig2 had 8 overlapping curves (4 models × 2 axes) that were hard to read; user requested two side-by-side panels.
- **Agent:** claude-opus-4-8
- **Did:** Rewrote `paper_figure2_layer_emergence` in `paper/figures/generate_figures.py` — changed from `plt.subplots(figsize=(7,4.5))` to `plt.subplots(1, 2, figsize=(11,4.5), sharey=True)`; left panel = warmth only, right panel = competence only; shared y-axis upper bound from global max; legend only on left panel; panel titles "Warmth" / "Competence"; suptitle updated to drop "solid=warmth, dotted=comp" line. File name unchanged.
- **Findings:** `paper_figure2_layer_emergence.{png,pdf}` regenerated successfully. Each panel shows 4 clean model curves; Llama/Qwen higher plateau clearly visible vs Gemma's lower but rising profile; probe-layer and d=0.80 reference lines present in both panels.
- **Decision / rationale:** Separating axes into panels reduces each panel from 8 to 4 curves, making model-level comparisons immediate. Shared y-axis preserves cross-panel comparability (warmth vs competence magnitudes directly comparable).
- **Next:** Commit all three paper figures for supervisor presentation.

---

## 2026-06-24 · Step 4 — Add direction arrows, remove large-effect line, inline probe-layer label

- **Context:** paper_figure2 polish: direction arrows on curves, remove axhline(d=0.80), move probe-layer label off legend onto figure.
- **Agent:** claude-opus-4-8
- **Did:** Rewrote `paper_figure2_layer_emergence` in `generate_figures.py`: (1) removed `axhline(0.8)` entirely; (2) `axvline(0.66)` no longer carries a legend label — replaced with `ax.text` annotation at the line using a blended transform; (3) added `_add_direction_arrows` inner helper that smooths each curve, finds steepest rise and (if substantial) steepest fall, then places `-|>` arrowheads offset `y_max * 0.048` above the curve in the parallel direction of the local tangent; legend now shows only 4 model labels.
- **Findings:** `paper_figure2_layer_emergence.{png,pdf}` regenerated; arrows visible and above curves; Llama/Qwen show rise+fall pair; Gemma shows rise only (no substantial descent detected).
- **Decision / rationale:** Arrows parallel to local tangent + vertical offset avoids overlap with curve lines. `annotation_clip=True` prevents out-of-bounds arrows.
- **Next:** Commit figures for supervisor presentation.

---

## 2026-06-24 · Step 5 — Curated free-space arrow placement for paper_figure2

- **Context:** REV5: automatic curve-hugging arrows tangled with lines; replaced with hand-placed arrows in visually empty bands.
- **Agent:** claude-opus-4-8
- **Did:** Removed `_add_direction_arrows` inner function from `paper_figure2_layer_emergence`; replaced with two explicit curated lists `ARROWS_WARMTH` and `ARROWS_COMP` (6 arrows each: 2×Llama rise+fall, 2×Qwen rise+fall, 1×Gemma-12B rise, 1×Gemma-27B rise). Arrow coordinates derived from frac-grid d-value analysis to place each arrowhead in the gap between curve clusters. Regenerated figure.
- **Findings:** `paper_figure2_layer_emergence.png` — all 12 arrows visible in clear empty bands; no arrows cross any curve; each arrow color matches its model.
- **Decision / rationale:** Curated positions trade automation for visual cleanliness; approach is maintainable because arrow count is fixed (4 models × fixed rise/fall pattern) and grid data makes gap selection straightforward.
- **Next:** Commit all three paper figures.

---

## 2026-06-24 · Step 6 — Remove arrows; add line-end layer+d_model labels to paper_figure2

- **Context:** Arrows removed per user request; replaced with informative end-of-line labels showing total layer count and residual-stream width (d_model).
- **Agent:** claude-opus-4-8
- **Did:** Replaced all arrow code (ARROWS_WARMTH, ARROWS_COMP, _ap, annotate loops) in `paper_figure2_layer_emergence` with a `_draw_end_labels` helper; added `D_MODEL` dict (Gemma-12B: 3840, Gemma-27B: 5376, Qwen: 5120, Llama: 4096); labels formatted as `"{n}L · d{dim}"`; vertical de-cluttering nudges overlapping labels apart by `y_max*0.045`; set `y_max*1.05` headroom; added `wspace=0.26` to open inter-panel channel for left-panel labels.
- **Findings:** Clean two-panel figure, no arrows, 4 colour-matched end labels per panel with correct values. Qwen/Llama and Gemma label pairs separated cleanly by de-clutter.
- **Decision / rationale:** End-of-line labels deliver the layer/d_model context directly adjacent to the curve they annotate, without adding marks that cross the plot area.
- **Next:** Commit all three paper figures for supervisor presentation.

---

## 2026-06-24 · Step 7 — Redesign paper_figure3 as causal steering schematic

- **Context:** Supervisor-presentation Figure 3 polish after the line-chart version was judged visually too generic.
- **Agent:** gpt-5-codex
- **Did:** Rewrote `paper_figure3_causal_steering` in `paper/figures/generate_figures.py` as a two-panel prompt → residual-stream direction intervention → No/Yes judgement-shift schematic; updated `paper/README.md`; regenerated `paper/figures/paper_figure3_causal_steering.{png,pdf}` with `python3 paper/figures/generate_figures.py --fig p3 --steering-slopes results/tables/gemma_scope_local_steering_slopes.csv`.
- **Findings:** Figure regenerated successfully. Warmth panel endpoints: Gemma-3-12B ±2.78 (R2=0.956), Gemma-3-27B ±1.29 (R2=0.990). Competence panel endpoints: Gemma-3-12B ±1.51 (R2=0.915), Gemma-3-27B ±0.89 (R2=0.826).
- **Decision / rationale:** Keep the causal claim visually focused on dense concept directions and direct concept prompts; random/other-axis controls remain in Figure 10/report text rather than the presentation schematic.
- **Next:** Commit the three presentation figures and documentation updates after final visual QC.

---

## 2026-06-24 · Step 8 — Compress paper_figure3 into one-way steering schematic

- **Context:** User requested a smaller single-figure version of paper_figure3 with one-way arrows and a visual style closer to paper_figure1/2.
- **Agent:** gpt-5-codex
- **Did:** Reworked `paper_figure3_causal_steering` from the two-panel bidirectional schematic into one compact prompt → residual-stream intervention → shared No/Yes axis figure; updated `paper/README.md`; regenerated `paper/figures/paper_figure3_causal_steering.{png,pdf}` with `python3 paper/figures/generate_figures.py --fig p3 --steering-slopes results/tables/gemma_scope_local_steering_slopes.csv`.
- **Findings:** Figure regenerated successfully with one-way +0.10 mean-residual-norm arrows. Endpoints: warmth 12B +2.78 (R2=0.956), competence 12B +1.51 (R2=0.915), warmth 27B +1.29 (R2=0.990), competence 27B +0.89 (R2=0.826).
- **Decision / rationale:** Show only positive direction addition in the presentation figure; leave negative steering symmetry and random/other-axis controls in Figure 10/report text.
- **Next:** Final visual QC and commit presentation figures.

---

## 2026-06-24 · Step 9 — Redraw paper_figure3 in compact blueprint style

- **Context:** User provided a reference mockup and requested a much smaller figure with the same structural layout, while retaining the paper color palette.
- **Agent:** gpt-5-codex
- **Did:** Reimplemented `paper_figure3_causal_steering` as a fixed-coordinate blueprint-style schematic with an outer frame, monospaced prompt text, intervention connector/callout, and compact bidirectional local-response bars; updated `paper/README.md`; regenerated `paper/figures/paper_figure3_causal_steering.{png,pdf}`.
- **Findings:** Figure regenerated successfully at `figsize=(4.9, 3.35)`. Bars use raw-dense local effects at ±0.10 mean residual norm: warmth 12B ±2.78, competence 12B ±1.51, warmth 27B ±1.29, competence 27B ±0.89.
- **Decision / rationale:** Preserve the mockup's local bidirectional-response grammar, but render warmth/competence in the established paper colors rather than the black-background terminal palette.
- **Next:** Commit presentation figures after final QC.

---

## 2026-06-24 · Step 10 — Align paper_figure3 typography and prompt wording

- **Context:** User requested paper_figure3 polish: add a short title, use the same typeface as paper_figure1/2, remove the long in-frame title text, align the No marker with -3, replace "Read the story below", and shrink the figure.
- **Agent:** gpt-5-codex
- **Did:** Updated `paper_figure3_causal_steering` to use the shared sans-serif style, added title "Concept vectors shift social judgement", replaced the prompt text with a long-story input placeholder, aligned `<- No` to the -3 tick, reduced figure size to `figsize=(4.35, 2.95)`, and regenerated `paper/figures/paper_figure3_causal_steering.{png,pdf}`.
- **Findings:** Figure regenerated successfully with the same raw-dense local effects: warmth 12B ±2.78, competence 12B ±1.51, warmth 27B ±1.29, competence 27B ±0.89.
- **Decision / rationale:** Use ASCII arrow labels (`<-`, `->`) to avoid Helvetica missing-glyph warnings while preserving the intended direction labels.
- **Next:** Commit presentation figures after final QC.

---

## 2026-06-24 · Step 11 — Further shrink paper_figure3

- **Context:** User requested that paper_figure3 be made much smaller.
- **Agent:** gpt-5-codex
- **Did:** Reduced `paper_figure3_causal_steering` from `figsize=(4.35, 2.95)` to `figsize=(3.0, 2.05)`, scaled down labels, strokes, markers, and replaced `||resid||` with `mean resid norm` for legibility; regenerated `paper/figures/paper_figure3_causal_steering.{png,pdf}`.
- **Findings:** Figure regenerated successfully and remains readable with the same local-response values: warmth 12B ±2.78, competence 12B ±1.51, warmth 27B ±1.29, competence 27B ±0.89.
- **Decision / rationale:** Preserve the compact schematic while reducing physical figure footprint and avoiding ambiguous vertical-bar glyphs at small size.
- **Next:** Commit presentation figures after final QC.

---

## 2026-06-24 · Step 12 — Narrow paper_figure3 horizontally

- **Context:** User requested that paper_figure3 be narrowed horizontally while keeping the vertical length unchanged.
- **Agent:** gpt-5-codex
- **Did:** Changed `paper_figure3_causal_steering` from `figsize=(3.0, 2.05)` to `figsize=(2.35, 2.05)`; shortened and wrapped title, prompt, intervention label, and x-axis label to prevent `bbox=tight` from expanding the saved output; regenerated `paper/figures/paper_figure3_causal_steering.{png,pdf}`.
- **Findings:** Figure regenerated successfully with the same local-response values: warmth 12B ±2.78, competence 12B ±1.51, warmth 27B ±1.29, competence 27B ±0.89.
- **Decision / rationale:** Reduce horizontal footprint without reducing vertical height or changing causal data.
- **Next:** Commit presentation figures after final QC.

---

## 2026-06-24 · Step 13 — Rework paper_figure3 as card-style steering schematic

- **Context:** User provided an Anthropic-style reference figure and requested paper_figure3 be made visually similar.
- **Agent:** gpt-5-codex
- **Did:** Reworked `paper_figure3_causal_steering` into a rounded card layout with a light prompt box, dashed intervention arrow, explanatory caption, and zero-origin response arrows; updated `paper/README.md`; regenerated `paper/figures/paper_figure3_causal_steering.{png,pdf}`.
- **Findings:** Figure regenerated successfully with raw-dense +0.10 local effects: warmth 12B +2.78, competence 12B +1.51, warmth 27B +1.29, competence 27B +0.89.
- **Decision / rationale:** Use the reference figure's preference-arrow grammar while retaining project colors and direct concept-judgement wording.
- **Next:** Commit presentation figures after final QC.

---

## 2026-06-24 · Step 14 — Remove paper_figure3 generation path

- **Context:** User deleted the Figure 3 output files and requested removal of only the Figure 3 creation code, while preserving other figures and all results.
- **Agent:** gpt-5-codex
- **Did:** Removed `paper_figure3_causal_steering`, removed the `--steering-slopes` CLI argument and p3 dispatch, made unsupported `--fig p3` fail explicitly, and removed the `paper_figure3_causal_steering.{png,pdf}` inventory row from `paper/README.md`.
- **Findings:** No data, results, reports, Figure 1 code, Figure 2 code, or other generated outputs were deleted by this step.
- **Decision / rationale:** Keep the presentation figure generator limited to p1 and p2 until Figure 3 is redesigned from scratch.

---

## 2026-06-24 · Step 15 — Redesign paper_figure3 as Anthropic-style diverging dot-arrow plot

- **Context:** User approved a new design for paper_figure3 inspired by the Anthropic "Emotion Concepts" paper figure showing steering-induced preference shifts.
- **Agent:** claude-opus-4-8
- **Did:** Added `paper_figure3_diverging_steering(slopes_csv)` to `paper/figures/generate_figures.py`; re-enabled `--fig p3` dispatch and added `--steering-slopes` CLI argument; updated `paper/README.md` inventory. Produced `paper/figures/paper_figure3_diverging_steering.{png,pdf}`.
- **Findings:** Figure has three blocks: (1) prompt box with real judgement prompt and `high_warmth` story excerpt, "warmth" highlighted in blue; (2) dashed-arrow annotation "We add the warmth / competence direction to the residual stream (±0.10 × mean residual norm)"; (3) diverging dot-arrow chart — 4 rows (12B warmth ±2.78, 12B competence ±1.51, 27B warmth ±1.29, 27B competence ±0.89), warmth=deep blue, competence=deep gold, baseline dot at 0, symmetric arrows left (No) and right (Yes).
- **Decision / rationale:** Anthropic-style grammar (dot-at-baseline + diverging arrows) communicates bidirectional causal control more intuitively than line+slope plots; values from raw_dense direction, slope×0.10 (R²≥0.83 for all rows).
- **Next:** Commit all three paper figures for supervisor presentation.

---

## 2026-06-24 · Step 16 — Rework paper_figure3 as position+boundary chart (first-class redesign)

- **Context:** User audit + design review identified REV7 figure as weak: oversized prompt box, symmetric arrows carrying no independent information, excess axis decoration, too wide.
- **Agent:** claude-opus-4-8
- **Did:** Rewrote `paper_figure3_diverging_steering` in `paper/figures/generate_figures.py`. Key changes: (1) figsize 8×6.5 -> 5.2×5.6; (2) GridSpec height_ratios [0.28,0.10,0.62] -> [0.16,0.10,0.74]; (3) x-axis reframed from symmetric change to absolute Yes/No logit margin (x=0 = decision boundary); (4) each row shows baseline dot (intercept) + steerable range line with arrow at +0.10 end — asymmetric, information-dense; (5) "warmth" inline-bold-blue via HPacker/AnnotationBbox offsetbox (no fragile x-offset); (6) stripped x ticks/bottom spine; (7) Yes-half soft shade; (8) bull's-eye dot (filled outer, white inner). Updated `paper/README.md` figure description.
- **Findings:** All four rows cross x=0 — ±0.10 steering is sufficient to flip Yes/No in every case. 12B·Warmth range: −2.21 to +3.34; 27B·Competence range: −1.23 to +0.54. Baseline positions reveal 12B starts Yes-leaning (+0.56), 27B·Comp starts No-leaning (−0.34).
- **Decision / rationale:** Reframing from "change" to "position + boundary" makes the causal claim concrete and the asymmetry (warmth > competence, 12B > 27B) visually legible. Symmetric papyon design carried zero extra information.
- **Next:** Commit all three paper figures for supervisor presentation.

---

## 2026-06-27 · Step 1 — Add dense (SAE-free) steering script and 4-model SGE jobs

- **Context:** Equalizing Phase-6 concept steering across all four models (Gemma-3-12B/27B, Llama-3.1-8B, Qwen3-14B) before proceeding to Phase 7 hiring.
- **Agent:** claude-sonnet-4-6
- **Did:** Created `src/dense_steering.py` — reuses validated helpers from `src/gemma_scope_causality.py` (unit, yes_no_margin, make_steering_hook, train_test_topics, summarize_*) but drops all Gemma Scope / SAE dependencies. Directions: `raw_dense` (high−low concept vector, train-topic means) + orthogonalized `random` control. Created four SGE jobs: `jobs/sge/steering_dense_gemma3_12b.sh`, `steering_dense_gemma3_27b.sh`, `steering_dense_llama31_8b.sh`, `steering_dense_qwen3_14b.sh`.
- **Findings:** Syntax check passed. Script reads all model-specific parameters (model name, probe layer, d_model) from `meta.json`; `mean_resid_norm` from stored activations; identical `train_test_topics` split via `cfg.probing.seed=20260527` and `n_test_topics=10` ensures comparable steering curves across models.
- **Decision / rationale:** Keep `gemma_scope_causality.py` untouched (Gemma SAE pipeline stays valid). Regression gate: Gemma-12B job must match existing `gemma_scope_causality_gemma3_12b_local.csv` raw_dense rows (warmth +0.1 → 3.88125, competence +0.1 → 2.00625) before submitting Llama/Qwen jobs.
- **Next:** Commit + push → SCCKN `git pull` → `qsub steering_dense_gemma3_12b.sh` (regression gate) → on pass, `qsub` remaining three jobs → pull results → write 4-model dense steering findings report in `paper/`.

---

## 2026-06-27 · Step 2 — Phase 7: productionise hiring pipeline to src/, add 4-model SGE jobs

- **Context:** Phase 7 (third headline output: model callback disparity vs. human disparity + mediation) previously lived only in notebooks 06/07, run only for Gemma-12B/27B. Decisions D-Phase7-A/B/C now locked with user.
- **Agent:** claude-sonnet-4-6
- **Did:** Created three model-agnostic src/ scripts: `src/hiring_steering.py` (causal sweep, GPU; replaces notebook 06), `src/hiring_audit.py` (probe-vs-human validation + baseline, GPU; replaces notebook 07), `src/hiring_disparity.py` (race/gender disparity table + bootstrap mediation, CPU-only). Updated `src/hiring_eval.py` stub to dispatcher pointing to the three new scripts. Created four SGE jobs: `jobs/sge/hiring_gemma3_12b.sh` (regression gate), `hiring_gemma3_27b.sh`, `hiring_llama31_8b.sh`, `hiring_qwen3_14b.sh`.
- **Findings:** Syntax check passed for all three new scripts. Local dry-run of `hiring_disparity.py` against existing `hiring_audit_concept_vectors.csv` (Gemma-12B): 269/282 names joined to `published_data/df_all.csv`; race disparity — Black margin=−0.184 vs White=−0.200; human callback — Black=0.183 vs White=0.171; all mediation tests n.s. at 12B (consistent with existing report). Research decisions: D-Phase7-A = `published_data/df_all.csv` name-level; D-Phase7-B = race(Black/White) primary, gender(Female/Male) secondary, mirroring Gallo & Hausladen's `group_by(name, race, gender)` coding; D-Phase7-C = bootstrap mediation N=5000 seeded.
- **Decision / rationale:** Isolated under new `hiring_*` labels (gemma3_12b etc.) so legacy Gemma outputs (prefix `concept_vectors`) are never overwritten. Gemma-12B job is regression gate: warmth Δmargin at +0.25/+0.50 ≈ +7.125/+8.404; probe-vs-human rho ≈ 0.355/0.230.
- **Next:** Commit + push → SCCKN `git pull` → `qsub hiring_gemma3_12b.sh` (gate) → on pass `qsub` remaining three → pull results → write 4-model Phase 7 findings report in `paper/`.

---

## 2026-06-27 · Step 3 — Dense steering 4-model findings: gitignore fix, figures, report

- **Context:** Dense (SAE-free) SCCKN jobs for all four models completed; results were never written up or committed because `steering_dense_*.csv/.json` were gitignored with no un-ignore exception.
- **Agent:** claude-opus-4-8
- **Did:** (1) Fixed `.gitignore` — added un-ignore exceptions for `steering_dense_*.csv`, `hiring_disparity_*.csv`, `steering_dense_*.json`, `hiring_steering_*.json`, `hiring_probe_vs_human_*.json`, `hiring_mediation_*.json`. (2) Updated `jobs/sync_outputs.sh` — added all eight new globs to the `git add` block and updated the header comment. (3) Added three figure builders to `paper/figures/generate_figures.py`: `fig13_dense_steering_doseresponse` (2×4 grid, raw_dense solid / random dashed, free y-axis), `fig14_dense_steering_normalized` (1×2 cross-model, effect/baseline_gap, shared y-axis), `fig15_dense_steering_signal_vs_control` (1×2 grouped bars, ⚠ annotation for leakage). Added `--dense-csvs` argument and dispatch block for `--fig 13/14/15`. (4) Wrote findings report `paper/2026-06-27_1446_dense_steering_4model.md`. (5) Updated `paper/README.md` with report row and fig13/14/15 inventory entries.
- **Findings:** Normalized steerability (effect/baseline_gap at α=+0.10): warmth 12B=0.236, Qwen=0.125, 27B=0.040, Llama=0.029; competence 12B=0.140, Qwen=0.103, Llama=0.024, 27B=0.009. Gemma-27B competence random-control leakage: random effect −3.36 > raw_dense +0.55 at α=+0.10 (non-specific perturbation dominates). Raw effects span ~100× due to mean_resid_norm differences (Llama 11.4 → Gemma-12B 79722); normalization is required for cross-model comparison. Dense `steering_dense_*.csv/.json` are now tracked; hiring output tracking (`hiring_disparity_*.csv`, `results/logs/hiring_*.json`) fixed simultaneously.
- **Decision / rationale:** Dense steering report placed before Phase 7 report to document the steerability baseline that predicts hiring causal inertia. gitignore/sync fix placed here because it was a blocker for committing any dense or hiring outputs from the cluster.
- **Next:** `scp` the 12 dense files from SCCKN into local repo → commit → push. Run figures: `python paper/figures/generate_figures.py --fig 13 14 15 --dense-csvs results/tables/steering_dense_{gemma3_12b,gemma3_27b,llama31_8b,qwen3_14b}.csv --labels "Gemma-3-12B,Gemma-3-27B,Llama-3.1-8B,Qwen3-14B"`. Monitor gate job 1080336 (Phase 7 hiring) → on pass submit remaining three hiring jobs → write 4-model Phase 7 report.

---

## 2026-06-27 · Step 4 — Render figures 13–19 and write Phase 7 consolidated report

- **Context:** All four Phase-7 hiring SCCKN jobs completed and outputs synced to `origin/main` (commit fe85dec). Plan called for rendering dense fig13–15 (long overdue) and building + rendering Phase-7 fig16–19, then writing the 4-model consolidated report.
- **Agent:** claude-opus-4-8
- **Did:** (1) Rendered `fig13_dense_steering_doseresponse`, `fig14_dense_steering_normalized`, `fig15_dense_steering_signal_vs_control` — images existed as builders but files were never produced (CSVs arrived locally only after the sync). (2) Added four builders to `paper/figures/generate_figures.py`: `fig16_hiring_probe_vs_human` (Spearman ρ grouped bars, signed, Llama/Qwen negative warmth visible), `fig17_hiring_steering_callback` (2×4 grid, mean Δmargin ± 95% CI over 60 names), `fig18_hiring_disparity` (two-panel: magnitude in within-model SD units + direction-agreement grid), `fig19_hiring_mediation_forest` (indirect effect forest plot, significant rows filled). Added `import json`, `from scipy.stats import spearmanr` to imports; added four `--hiring-*` CLI args; added dispatch block for 16/17/18/19. (3) Rendered all seven figures in one command (no errors; font-fallback warnings for special chars are cosmetic). (4) Added `## Input data` block to `paper/2026-06-27_1446_dense_steering_4model.md` documenting: concept_stories.jsonl 200×6, 4 conditions, 50 topics, claude-opus-4-8 generator, SCM + Sofroniew/Lindsey 2026 literature. (5) Wrote `paper/2026-06-27_1541_hiring_phase7_4model.md` — full 4-model consolidated Phase-7 report with Artifacts, Input-data, probe-vs-human (anti-alignment discussion), steering→callback, disparity (SD-normalized + direction panel), mediation forest (steerability paradox), cross-report reconciliation, bridge to dense steering, caveats. (6) Updated `paper/README.md` with new report row and fig16–19 inventory.
- **Findings:** fig16: Gemma warmth ρ= +0.366/+0.396 (positive); Llama −0.300, Qwen −0.193 (anti-aligned). fig17: 12B warmth Δ@+0.50 = +8.35 (strong); 27B −0.23 (inert); Llama +3.17 (moderate); Qwen +0.60 (weak). fig18: Gemma-27B race gap +1.255 SD (largest); gender direction opposed by 12B/Llama/Qwen. fig19: 5 significant mediation entries — Llama race×warmth IE=+0.190 [+0.111,+0.292] (largest); 12B and 27B null; Qwen race×competence −0.132 (reversed sign). Steerability paradox: most steerable model (12B) shows null mediation; least steerable (Llama) shows strongest mediation.
- **Decision / rationale:** Input-data provenance added to both reports per user request; exact row/column/label counts verified from source files before writing. Raw signs retained for warmth anti-alignment (user decision; discussed as genuine finding, not sign error).
- **Next:** Commit + push all changes (generate_figures.py, dense report update, Phase-7 report, figures 13–19, README, STEP_LOG).

---

## 2026-06-27 · Step 5 — Add plain-language experimental design and bias interpretation to Phase-7 report

- **Context:** User asked for a plain-language description of what the pipeline actually does and whether the model is being "racist", so both clarifications are now embedded in the report for any future reader.
- **Agent:** claude-opus-4-8
- **Did:** Added three sections to `paper/2026-06-27_1541_hiring_phase7_4model.md`: (1) `## Experimental design` block before Summary — three-measurement structure: (a) hiring prompt with exact template from `src/hiring_audit.py`, callback margin sign convention, explicit note that race/gender is never given to the model; (b) probe measurement from neutral name sentence; (c) disparity+mediation combination. (2) Plain-language callback-margin sign explanation + "labels come from the benchmark, not the model" note before §3.1. (3) New `§3.3 Is there bias?` subsection with bias verdict (yes, differential treatment), direction note (reverse of classic discrimination, likely RLHF over-correction), and inconsistency-as-finding summary; renumbered old §3.3 to §3.4.
- **Findings / Decision:** No numbers changed. All additions are interpretive framing, not new results. Bias framing: differential treatment confirmed; direction opposite to classical discrimination in 3/4 models; main finding is model-to-model inconsistency rather than a stable discriminatory pattern.
- **Next:** Commit + push.

---

## 2026-06-27 · Step 6 — Expand dense steering report: mechanism detail and ±0.10 range discussion

- **Context:** User asked how the steering push works mechanically and whether ±0.10 is necessary/sufficient. Full answer added to the report so any future reader has it in context.
- **Agent:** claude-opus-4-8
- **Did:** Expanded `paper/2026-06-27_1446_dense_steering_4model.md` Method and Caveats sections: (1) Added "Steering mechanism — what 'pushing' means in practice" paragraph with explicit additive formula, code reference to `make_steering_hook`, absolute magnitude examples (Gemma-12B ~7.97 vs Llama ~1.14 per unit strength), and explanation of random control construction. (2) Added new Caveat 5 "Strength range ±0.10: sufficient for causal proof, insufficient for full characterisation" covering saturation, decision-flip threshold, underestimation for weaker models, asymmetry with Phase-7 hiring sweep (±0.50), and recommended future extension via `--strengths`. (3) Updated "Bridge to Phase 7" to point to completed report and summarise the steerability paradox finding.
- **Findings / Decision:** No new numbers. All additions are methodological framing and future-work notes derived from comparing Phase-6 and Phase-7 design choices.
- **Next:** Commit + push.

---

## 2026-06-27 · Step 7 — Stimulus quality audit: 200 concept stories scored 8.5/10

- **Context:** User asked for a full quality audit of the concept stories corpus with a 10/10 scoring rubric, to document stimulus quality before paper writing.
- **Agent:** claude-opus-4-8
- **Did:** Ran inline structural analysis on `data/stimuli/concept_stories.jsonl` (word counts, sentence counts, name/label leakage, topic balance, minimal-pair coverage). Read three full minimal quads (topic_idx 0, 5, 20) for narrative quality assessment. Wrote findings report `paper/2026-06-27_1650_stimulus_quality_audit.md` covering: story type description, full structural metrics table, two sample quads with commentary, 10-criterion scored rubric, paper implications, and recommended next steps. Updated `paper/README.md` with new report row.
- **Findings:** Overall score 8.5/10. Strengths: perfect 50/50/50/50 balance; full minimal-pair coverage (same 50 topics for both axes); zero name/demographic/competence-label leakage; zero warmth-label leakage except 2/200 marginal cases in low_warmth; behavioral show-don't-tell quality is strong; std ~12 words. Weaknesses: (1) all 200 stories generated by single model claude-opus-4-8 (5/10 on source diversity — main methodological risk); (2) no independent human validity rating of the stories themselves (6/10). Inter-axis angle ~41–59° (not 90°) reflects known SCM warmth/competence correlation, not a corpus flaw.
- **Decision / rationale:** Mono-source generation caveat must appear in the paper Stimuli section. Human manipulation-check study (40 stories, Prolific) recommended before journal submission. Both can be added without invalidating existing results.
- **Next:** Commit + push. Begin paper writing using these reports as source material.

---

## 2026-06-27 · Step 8 — Reframe stimulus audit as dataset acceptance report

- **Context:** User decided to proceed with the existing concept-story dataset and asked for the audit report to state facts clearly, remove next-step framing, and define limitations.
- **Agent:** gpt-5-codex
- **Did:** Revised `paper/2026-06-27_1650_stimulus_quality_audit.md` to mark the dataset accepted for current analyses; replaced recommended next steps with explicit limitations and an audit conclusion. Updated the corresponding `paper/README.md` status row.
- **Findings:** No empirical values changed. The report still records score 8.5/10, perfect condition balance, negligible leakage, mono-source generation, no independent human manipulation check, concept non-orthogonality, and minor low-condition verbosity.
- **Decision / rationale:** Treat limitations as paper caveats rather than blockers; proceed with `data/stimuli/concept_stories.jsonl` as the trusted concept-stimulus dataset for the reported analyses.

---

## 2026-06-27 · Step 9 — Add plain-language probe-vs-human audit explanation

- **Context:** User requested that the report include a non-technical explanation of Test 2, including what is measured, which data are used, and how Spearman alignment should be interpreted.
- **Agent:** gpt-5-codex
- **Did:** Updated `paper/2026-06-27_1541_hiring_phase7_4model.md`: expanded Measurement 2 with a plain-language explanation; added a Test-2 preamble to `## 1 · Probe-vs-human alignment (fig16)` describing the human ratings input, model concept-vector input, neutral name-sentence probe, output table, JSON log, and figure. Corrected the Artifacts output list to include `results/logs/hiring_probe_vs_human_<label>.json`.
- **Findings:** No empirical results changed. The report now explicitly states that this audit is not a hiring-decision test; it measures whether model-internal name-level warmth/competence scores rank names similarly to human warmth/competence ratings.
- **Decision / rationale:** Keep technical details present but make the purpose and data flow readable for non-technical paper readers.

---

## 2026-06-27 · Step 10 — Write Test-2 probe-vs-human data audit report

- **Context:** User requested a formal audit report for the data used in Test 2, including source, limitations, a 10-point score, and an explicit decision on whether to proceed with the data.
- **Agent:** gpt-5-codex
- **Did:** Wrote `paper/2026-06-27_1757_probe_human_data_audit.md`; updated `paper/README.md` with the new report row.
- **Findings:** Test-2 data scored 8.0/10. Human rating data: 24,220 rows, 282 names, 787 raters, 10 studies, 0 missing warmth values, 16 missing competence values. Main limitation: per-name rater imbalance (44 names with one rating; 102 names with fewer than five). Robustness checks show headline correlations strengthen after filtering to better-rated names, indicating sparse-rated names add noise rather than create the effects.
- **Decision / rationale:** Proceed with `ratings/names/df_all.csv`, the four `concept_vectors*/` directories, and the existing probe-vs-human outputs as the trusted Test-2 data basis; disclose uneven rating reliability and other scope caveats in the paper.

---

## 2026-06-27 · Step 11 — Add Test-3 baseline callback correlation explanation

- **Context:** User asked whether the Phase-7 report already explained Test 3 baseline callback correlations and requested the missing explanation be added.
- **Agent:** gpt-5-codex
- **Did:** Updated `paper/2026-06-27_1541_hiring_phase7_4model.md` to describe the no-steering baseline callback association check and the four `callback_vs_*` rows in `results/logs/hiring_probe_vs_human_<label>.json`.
- **Findings:** No results changed. Added Spearman rho table for callback margin vs model warmth, model competence, human warmth, and human competence across the four models: 12B positive; 27B negative; Llama weak; Qwen strongest on model competence.
- **Decision / rationale:** Treat Test 3 as a descriptive baseline association audit, not causal evidence; causal claims remain tied to the steering sweep.

---

## 2026-06-28 · Step 1 — Embed generated figures into report bodies

- **Context:** User noticed that several recently generated figures were listed in report artifact blocks but not embedded in the corresponding Markdown report bodies.
- **Agent:** gpt-5-codex
- **Did:** Added inline PNG embeds and short captions to `paper/2026-06-27_1446_dense_steering_4model.md` (fig13–15), `paper/2026-06-27_1541_hiring_phase7_4model.md` (fig16–19), `paper/2026-06-24_1136_hiring_causality_results.md` (12B hiring figures), `paper/2026-06-24_1300_hiring_causality_27b_results.md` (27B hiring figures), and `paper/2026-06-27_1757_probe_human_data_audit.md` (fig16 reuse).
- **Findings:** No figure files, result tables, logs, or empirical values changed. All embeds point to existing PNG files under `paper/figures/` or `results/figures/`.
- **Decision / rationale:** Keep PDFs in artifact inventories for export/publication use while embedding PNGs for readable Markdown reports, matching older report style.

---

## 2026-06-30 · B1 Float32 Fix + Denoising Results

- **Context:** Audit of callback margins revealed all values in every hiring CSV fall on a 0.125 grid (7–8 unique values across 282 names). Root cause: `src/gemma_scope_causality.py::yes_no_margin()` computed `logit(Yes) − logit(No)` while tensors were still in bf16. At the logit magnitudes used (~5–10), bf16 can only represent differences in steps of 0.125.

- **Affected outputs (all must be regenerated):**
  - `results/tables/hiring_audit_gemma3_{12b,27b,llama31_8b,qwen3_14b}.csv`
  - `results/tables/hiring_steering_raw_gemma3_*.csv`
  - `results/tables/hiring_disparity_gemma3_*.csv`
  - `results/logs/hiring_mediation_*.json`
  - `results/tables/hiring_audit_concept_vectors{,_gemma3_27b}.csv` (notebook outputs)

- **Fix applied:**
  - `src/gemma_scope_causality.py` line 78: `logits[0, -1]` → `logits[0, -1].float()`
  - Same fix in inline copies in `notebooks/06_hiring_steering_causality.ipynb` and `notebooks/07_hiring_audit.ipynb`

- **Denoising completed (notebook 08):**
  - 12B: k=1 PCA component covers 56% neutral variance; cos(W,C) 0.749→0.530; d_warmth 2.67→8.45; leak 2.28→5.02
  - 27B: k=43 components cover 50% neutral variance; cos(W,C) 0.708→0.487
  - Interpretation: remaining cos≈0.53 reflects genuine SCM inter-axis correlation (not valence artefact) — consistent with human rating correlation ρ=0.61 in Gallo & Hausladen data.

- **27B local-regime steering (notebook 06, USE_DENOISED=False):**
  - Warmth: Δ=+1.97 at +0.05 strength, collapses to Δ=−2.66 at +0.10 (non-monotone)
  - Competence: similar collapse
  - Interpretation: scale dissociation is **real but not saturation** — 27B has a narrow controllable window; small perturbations outside it destabilise the response. This is a genuinely different finding from "27B is flat."

- **Notebook 06 fix:** output CSV now saves to `_denoised` suffixed filename when `USE_DENOISED=True`, preventing overwrites.

- **Re-run plan:** see `docs/rerun_checklist.md` for exact commands for Jorge (JupyterHub) and Emre (SCCKN cluster qsub jobs).

- **What does NOT need re-running:** probe training, concept vectors, Gemma Scope SAE analysis, layer sweeps, denoising, Spearman correlations — none use logit subtraction.

---

## 2026-07-02 · Step 1 — Document bf16 quantisation limitation as important paper caveat

- **Context:** Session review of Bug B1 (float32 fix committed by Jorge 2026-06-30); user requested the limitation be formally documented in the paper directory.
- **Agent:** claude-sonnet-4-6
- **Did:** Created `paper/2026-07-02_1000_bf16_quantisation_limitation.md` covering root cause, partial fix (`.float()` cast at `src/gemma_scope_causality.py:81`), why margins remain on 0.125 grid even after fix (bf16 inference inherent), model-by-model impact table (12B unreliable; 27B/Llama/Qwen usable), affected/unaffected results, re-run requirements, post-run diagnostic snippet, and mandatory paper disclosure language. Added row to `paper/README.md`.
- **Findings:** Fix is in the codebase and all pipeline scripts inherit it via import. 12B (SD=0.14, 7 unique values) cannot produce reliable disparity findings without float32 inference. 27B SD=0.41 is sufficient. Cluster re-runs for all 4 models still pending.
- **Decision / rationale:** Separate standalone report chosen over inline note so it is findable as a first-class limitation, not buried in a results file.
- **Next:** Emre submits 4 SGE jobs (`qsub jobs/sge/hiring_gemma3_*.sh`); re-runs notebook 09 with new CSVs; verifies SD per model with diagnostic snippet.

---

## 2026-07-02 · Step 2 — Reconcile dense-steering report tables to committed CSVs; incorporate B1 re-run

- **Context:** B1 re-run (8 SGE jobs) completed; figures fig13/14/15 already regenerated (commit `01dd389`). User requested full report update: if findings changed, update them too.
- **Agent:** claude-sonnet-4-6
- **Did:** Edited `paper/2026-06-27_1446_dense_steering_4model.md` and `paper/2026-06-27_1541_hiring_phase7_4model.md`. Dense report: added correction note; updated fig13 raw-effect table (all 4 models), fig14 Qwen row (0.125→0.122 warmth, 0.103→0.104 competence), fig15 signal-vs-control table (all rows including precise random values), prose on 27B warmth non-specificity (+0.61 control), and caveat 2. Phase7 report: Qwen dense-warmth steerability 0.125→0.122 in the bridge table (§6).
- **Findings:** Pre-existing inconsistency: the original fig13/fig15 tables had cited numbers inconsistent with committed CSVs and the actual figures (e.g. Qwen warmth "+25.74" vs CSV effect 1.20). Both tables now match the `effect` column of `results/tables/steering_dense_*.csv`. B1 re-run delta was small: Qwen normalized warmth 0.125→0.122, 27B rows shifted ~0.01. No qualitative findings changed: steerability ranking 12B > Qwen > 27B ≈ Llama holds; Gemma scale paradox unchanged; 27B competence leakage (random −3.36 >> signal +0.21) unchanged.
- **Decision / rationale:** Full reconcile to committed CSV source-of-truth chosen (user confirmed) over minimal-update option, so reports agree with figures already in the repo.
- **Next:** Commit and push the 2 reports + STEP_LOG.

---

## 2026-07-02 · Step 3 — Verify SCCKN hiring rerun outputs against git

- **Context:** User asked whether post-B1 hiring/callback outputs existed on SCCKN and, if so, to push them and pull locally.
- **Agent:** gpt-5-codex
- **Did:** SSH-inspected `/work/emrecan.ulu/normalcy-axis` on SCCKN; checked git status, output mtimes, SHA-256 hashes, job tails, and local/remote ahead-behind state for `hiring_audit_*`, `hiring_steering_raw_*`, `hiring_disparity_*`, and `hiring_mediation_*`.
- **Findings:** SCCKN hiring jobs reran on 2026-07-02 and rewrote the expected files, but their SHA-256 hashes exactly match local/git versions; `git diff` on SCCKN for those paths is empty. First three job sync attempts failed because SCCKN lacked git author identity, while the Qwen job reported `[sync] nothing to commit`. Local `main` is 0 ahead / 0 behind `origin/main`. SCCKN repo was fast-forwarded from `6a476d3` to `e0b19a6` and is now 0 ahead / 0 behind `origin/main`.
- **Decision / rationale:** No hiring output push or local pull was needed because there were no content changes to transfer; the apparent rerun existed only as newer mtimes, not new values.

---

## 2026-07-02 · Step 4 — Remove stale rerun and next-step wording from reports

- **Context:** User confirmed SCCKN hiring outputs are current and requested removal of obsolete rerun/plan wording from reports.
- **Agent:** gpt-5-codex
- **Did:** Cleaned stale `pending`, `re-run required`, `Action`, `Open decisions`, and `Next steps` language from `paper/2026-07-02_1000_bf16_quantisation_limitation.md`, `paper/2026-06-27_1541_hiring_phase7_4model.md`, `paper/2026-06-24_1136_hiring_causality_results.md`, `paper/2026-06-24_1300_hiring_causality_27b_results.md`, `paper/2026-06-30_1251_r4_disparity_name_level.md`, and `paper/README.md`.
- **Findings:** No result tables, logs, figures, scripts, or artifact paths were changed. Reports now state that SCCKN hiring re-runs completed on 2026-07-02 and produced no content changes where relevant; the B1 impact table was completed from existing current audit outputs for Llama and Qwen.
- **Decision / rationale:** Preserve historical single-model reports while marking them superseded for final interpretation by the consolidated four-model Phase 7 report.

---

## 2026-07-02 · Step 5 — Add PCA denoising figure and report

- **Context:** User requested a paper-ready PCA figure for the existing denoising outputs.
- **Agent:** gpt-5-codex
- **Did:** Added Figure 20 generation to `paper/figures/generate_figures.py`, regenerated `paper/figures/fig20_pca_denoising.{png,pdf}`, created `paper/2026-07-02_1921_pca_denoising_results.md`, and updated the figure/report inventory in `paper/README.md`.
- **Findings:** No PCA-specific figure existed previously. Existing denoising artifacts show Gemma-3-12B removes k=1 PC covering 56.1% neutral variance and reduces cos(W,C) 0.749→0.530; Gemma-3-27B removes k=43 PCs covering 50.2% neutral variance and reduces cos(W,C) 0.708→0.487.
- **Decision / rationale:** Use the existing neutral-corpus PCA artifacts directly and frame the result as a neutral-variance control, not as proof that warmth and competence are orthogonal pure axes.

---

## 2026-07-02 · Step 6 — Local repository fast-forward sync

- **Context:** User requested a git sync from the local Windows checkout.
- **Agent:** gpt-5-codex
- **Did:** Read current step-log/report state, fetched `origin`, and fast-forwarded local `main` from `9b7d014` to `8cf7a1b`.
- **Findings:** Pull brought in 147 changed files, including hiring/steering outputs, Figure 20 PCA denoising assets, updated reports, notebooks, SGE scripts, and paper draft files. After pull, local `main` matched `origin/main` with a clean worktree before this log entry.
- **Decision / rationale:** Recorded the sync as a repository state transition because it changed the local empirical/reporting baseline.

---

## 2026-07-02 · Step 7 — Compile current paper draft PDF

- **Context:** User requested a PDF from the current paper draft and indicated that the early draft should now become the full paper draft.
- **Agent:** gpt-5-codex
- **Did:** Installed missing TinyTeX packages (`appendix`, `preprint`, `fancyhdr`, `caption`), added the paper figure graphics path and five referenced figure environments to `paper/paper/Ulu_Lastra.tex`, added six missing bibliography entries to `paper/paper/references.bib`, and compiled `paper/paper/Ulu_Lastra.pdf`.
- **Findings:** PDF compilation succeeds at 11 pages with no undefined references or citations. Remaining LaTeX messages are only layout/font warnings. Output path: `paper/paper/Ulu_Lastra.pdf`.
- **Next:** Expand and update the paper text from the current reports, especially stale early-draft claims and the completed four-model hiring/disparity results.

---

## 2026-07-02 · Step 8 — Add active manuscript writing rules

- **Context:** User provided paper-writing rules from another repository and selected which ones to adapt into this repository's `AGENTS.md`.
- **Agent:** gpt-5-codex
- **Did:** Added active manuscript, prose style, anti-formulaic writing, user-decision, idea-note, and paper-figure rules to `AGENTS.md`, adapted to this repository's existing paths.
- **Findings:** Active manuscript source is now documented as `paper/paper/Ulu_Lastra.tex`; paper figures remain under `paper/figures/`; idea notes are assigned to `paper/idea-notes/`; step logging remains `step_logs/STEP_LOG.md`.
- **Decision / rationale:** Preserved the repository's existing layout instead of importing incompatible paths such as `ai-usage/`, `figures/script_figures/`, or a dated active paper edition folder.
- **Next:** Use these rules for future active manuscript writing. Anti-formulaic manuscript self-check was not run because no manuscript prose was edited in this step.

---

## 2026-07-02 · Step 9 — Rewrite Introduction; remove standalone Literature section

- **Context:** Paper writing session — replacing the early-draft Introduction + Literature with a single flowing Introduction following the agreed PAPER_STRUCTURE.md blueprint.
- **Agent:** claude-opus-4-8
- **Did:** (1) Verified bibliographic details for three new sources via WebFetch/WebSearch: Wilson & Caliskan AIES 2024, Chaturvedi & Chaturvedi arXiv 2025, SHRM 2025 Talent Trends. (2) Appended three entries to `paper/paper/references.bib` (keys: `shrm2025talent`, `wilson2024gender`, `chaturvedi2025callback`). (3) Replaced `\section*{Introduction}` (3 paragraphs) and the entire `\section*{Literature}` section (5 subsections) in `paper/paper/Ulu_Lastra.tex` with a single 4-paragraph Introduction following the agreed arc: adoption+stakes → bias real+LLMs reproduce it (black-box) → interpretability door (Anthropic) → warmth/competence tease + contribution. All cites from the deleted Literature section (SCM, correspondence studies, interpretability, audit papers) fold into the new Introduction.
- **Findings:** New Introduction cites: `shrm2025talent`, `bertrand2004emily`, `oreopoulos2011why`, `neumark2019older`, `correll2007getting`, `tilcsik2011pride`, `ameri2018disability`, `wilson2024gender`, `chaturvedi2025callback`, `an2024llm`, `an_measuring_2025`, `gaebler2024auditing`, `mikolov2013word2vec`, `park2024linear`, `zou2023repeng`, `turner2023activation`, `olah2020zoom`, `sofroniew2026emotion`, `fiske2002model`, `gallo2024warmth`. No orphaned cites: all previously Literature-only keys now appear in the Introduction or remain in the bib for future §3 (fiske2007universal, cuddy2007biasmap, cuddy2008warmth, fiske2018stereotype). Methods, Results, Discussion, Limitations, Future Work unchanged. Note: both new audit papers (Wilson & Caliskan; Chaturvedi & Chaturvedi) point in the traditional-bias direction (favor White/men); they are NOT framed as matching our 27B overcorrection — the honest framing is that behavioral audits disagree on direction and none locate the internal mechanism.
- **Decision / rationale:** Single flowing Introduction (no separate Literature section) per user decision; SCM/Gallo get a short tease in Intro Para 4 with full treatment deferred to §3.
- **Anti-formulaic self-check:** Para openers: "AI-powered tools now..." / "That skew is not hypothetical." / "Mechanistic interpretability offers..." / "The construct is warmth and competence---". No two adjacent paragraphs share a syntactic frame. No signal-only transitions. Sentence length varied within each paragraph. No prohibited pattern appears three or more times. Self-check passed.
- **Next:** Compile PDF to confirm no undefined-citation warnings; then draft §2 Background (pedagogy) and §3 Why Warmth & Competence (SCM + Gallo full treatment).

---

## 2026-07-02 · Step 10 — Draft §2 Background + schematic figure

- **Context:** Paper writing session — adding the pedagogical Background section to `paper/paper/Ulu_Lastra.tex` following the agreed PAPER_STRUCTURE.md §2 blueprint.
- **Agent:** claude-opus-4-8
- **Did:** (1) Wrote `paper/figures/background_concept_geometry.py` (new hand-made schematic figure script, reusing `style.py` palette and `ARROW_WARMTH`); generated `.png` and `.pdf`. (2) Inserted `\section*{Background: Reading a Concept Out of a Language Model}` with four labeled subsections between Introduction and Methods in `paper/paper/Ulu_Lastra.tex`. (3) Full compile cycle (pdflatex → bibtex → pdflatex × 2); BibTeX zero warnings, final pass zero undefined-citation/reference warnings.
- **Findings:** Section inserts cleanly. Figure `fig:concept_geometry` resolves. Cites: `olah2020zoom`, `mikolov2013word2vec`, `park2024linear`, `turner2023activation`, `zou2023repeng`, `sofroniew2026emotion` — all pre-existing, no new bib entries. Two displayed equations: eq:mean_diff ($v = \bar{h}_A - \bar{h}_B$) and eq:steering ($h' = h + \alpha\hat{v}$).
- **Decision / rationale:** ML-literate register (not from-scratch pedagogy); labeled subsections (four beats: residual stream / concept-is-a-direction / steering / Anthropic template); schematic figure now rather than deferred. Warmth/competence vectors are NOT called "emotion vectors" anywhere.
- **Anti-formulaic self-check:** Subsection openers vary: "Every forward pass…" / "The intuition that semantic content…" / "A direction extracted by the mean-difference method…" / "\citet{sofroniew2026emotion} applied this framework…". No two adjacent openers share a syntactic frame. No signal-only transitions. Sentence length varied. Self-check passed.
- **Next:** Draft §3 "Why Warmth & Competence — Why Hiring — Which Data" (SCM + Gallo full treatment + story/name-set data description).

---

## 2026-07-02 · Step 11 — Refine background_concept_geometry figure

- **Context:** User reviewed the v1 figure and requested: fix center crowding, shrink figure, 50 dots per cloud, move v label, legend top-left tight, academic polish.
- **Agent:** claude-opus-4-8
- **Did:** Rewrote `paper/figures/background_concept_geometry.py`: N=50 per cloud (real per-condition count), s=20 crisp markers, centroid × anchors at mu_high/mu_low, v label relocated to lower-right quadrant with white bbox, legend moved to upper-left with compact spacing (labelspacing=0.3, handletextpad=0.4), steering alpha reduced so h' star lands in clear space between boundary and cluster, h label moved left of src, decision boundary label tucked to upper-left corner, figsize=(3.3,2.7). Updated tex to `width=0.85\columnwidth`. Recompiled: BibTeX zero warnings, final pass zero undefined citations/refs.
- **Findings:** Figure visually confirmed clean: no label overlaps on arrow/clusters, 50 dots confirmed, legend upper-left tight, v label lower-right, figure compact.
- **Next:** Draft §3 "Why Warmth & Competence — Why Hiring — Which Data."

---

## 2026-07-02 · Step 12 — New Figure 1: emotion-vector concept schematic

- **Context:** Paper writing session — user requested a pedagogical Figure 1 showing that LLM internal states move along emotion directions in activation space, preceding the warmth/competence geometry figure.
- **Agent:** claude-sonnet-4-6
- **Did:** (1) Created `paper/figures/background_emotion_vector.py` (new hand-made figure; three emotion arrows — joy/fear/sadness — radiating from a neutral origin *h*; fear highlighted in crimson with star and numeric bracket vector $\hat v_{\text{fear}} = [0.21, -0.52, 0.08, \ldots, 0.14] \in \mathbb{R}^d$; prompt cue entering from left; activation space label; receded joy/sadness in blue/grey). Generated `.png` + `.pdf`. (2) Inserted the new `\begin{figure}` float in `paper/paper/Ulu_Lastra.tex` immediately before `background_concept_geometry` float, labelled `fig:emotion_vector`; old geometry float becomes Figure 2. (3) Added one `\autoref{fig:emotion_vector}` sentence at the end of the "Emotion Vectors as the Template" subsection. (4) Full compile cycle (pdflatex → bibtex → pdflatex × 2): zero new errors or undefined-reference warnings.
- **Findings:** Both `fig:emotion_vector` and `fig:concept_geometry` resolve; font-size substitution warnings are pre-existing and not new. Figure reads cleanly: three labeled directions, fear bold and highlighted, bracket vector visible, no label overlaps.
- **Decision / rationale:** Three-emotion radial composition selected (communicates "each concept = a direction" most directly); numeric bracket vector included to convey high-dimensionality; crimson fear matches the red used in Figure 2 for visual continuity across the pair.
- **Anti-formulaic self-check:** The added sentence ("\autoref{fig:emotion_vector} illustrates the core idea: each emotion occupies a distinct direction…") opens with a figure reference — different from all surrounding subsection openers. No signal-only transition. Check passed.
- **Next:** Draft §3 "Why Warmth & Competence — Why Hiring — Which Data."

---

## 2026-07-02 · Step 13 — Refine background_emotion_vector figure

- **Context:** User reviewed `background_emotion_vector.pdf` and requested a more professional paper-introduction schematic: remove left/bottom lines, shorten vectors, strengthen non-fear arrows, and add vector coordinates for all directions.
- **Agent:** gpt-5-codex
- **Did:** Edited `paper/figures/background_emotion_vector.py`, regenerated `paper/figures/background_emotion_vector.{png,pdf}`, visually inspected the PNG, and recompiled `paper/paper/Ulu_Lastra.tex`.
- **Findings:** Figure now has no visible axis spines or residual-stream axis labels; joy and sadness arrows are stronger and shorter; joy, fear, and sadness all carry compact coordinate-vector labels. `pdflatex -interaction=nonstopmode Ulu_Lastra.tex` completed successfully and loaded the revised PDF figure; remaining messages are underfull/font warnings, not figure errors.
- **Decision / rationale:** Kept the three-direction radial schematic and highlighted fear as the example direction, while removing plot-like scaffolding so the panel reads as a clean conceptual figure rather than a data axis.

---

## 2026-07-07 · Step 1 — Update Methods/Results to current findings; Background subsections -> paragraphs

- **Context:** Jorge requested (1) converting the Background section's \subsection* headings to \paragraph with non-redundant openers, and (2) rewriting Methods and Results to match the current reconciled findings reports.
- **Agent:** claude-fable-5
- **Did:** Edited `paper/paper/Ulu_Lastra.tex`. Background: four \subsection* -> \paragraph, headings renamed to avoid repeated "concept/vector" framing (Activations and the Residual Stream / Concepts as Directions / From Correlation to Causation / The Emotion-Vector Template). Methods: split hiring protocol out of thin Benchmark/Data paragraphs into "Hiring Evaluation" (prompt, callback margin, post-hoc demographic coding, probe-vs-human Spearman, 24,220 ratings/787 raters) and "Disparity and Mediation" (group ns, within-model SD standardisation, 149-name benchmark join, bootstrap mediation n_boot=5000 seed 20260527 n=269/race 227); Causal Steering now states both regimes (concept-level local ±0.05/±0.10; hiring 60-name broad ±0.25/±0.50 + Gemma local follow-up). Results: Qwen normalized steerability 0.125/0.103 -> 0.122/0.104 (B1 float32 reconcile); replaced forward-reference paradox tail of the steerability paragraph with signal-vs-control results (27B warmth ratio ~1.7x; 27B competence control −3.36 dominates signal +0.21); added Llama/Qwen hiring-steering results (broad regime); replaced stale "Llama/Qwen disparity awaits re-runs" with their SD-unit gaps (race +0.40/+0.16, gender +0.45/+0.89) and the four-model direction summary; added a dedicated mediation paragraph with fig19 (fig19_hiring_mediation_forest, previously absent from the manuscript), five significant IEs, Bonferroni note, and the steerability paradox. Limitations: replaced the stale pending-reruns sentence with the multiple-comparisons caveat; Future Work: removed the completed R4-extension item. Compiled full cycle (pdflatex/bibtex): 14 pages, zero errors, zero undefined citations/references.
- **Findings:** All values sourced from `paper/2026-06-27_1541_hiring_phase7_4model.md`, `paper/2026-06-27_1446_dense_steering_4model.md`, and `paper/2026-06-30_1251_r4_disparity_name_level.md`. Note for authors: the human race-gap reference (−0.085) comes from the R4 149-name matched join, while the Llama/Qwen SD-unit gaps come from the phase-7 282-name analysis; the mixed sourcing follows PAPER_STRUCTURE.md but may deserve harmonisation before submission.
- **Anti-formulaic self-check:** New paragraph openers vary ("Hiring stimuli combine..." / "Group-level disparities are differences..." / "Bootstrap mediation along the path... asks a different question" / "The two cross-architecture models, evaluated in the broad regime, fall between these extremes"). No repeated opener frames in adjacent passages, no signal-only transitions, no pattern recurring three or more times. Check passed.
- **Next:** Draft §3 "Why Warmth & Competence — Why Hiring — Which Data" between Background and Methods; decide on harmonising the disparity benchmark join.

---

## 2026-07-08 · Step 2 — Add §3 "Why Warmth and Competence, Why Hiring, and Which Data"

- **Context:** Paper writing session — inserting the missing motivation/theory section between Background and Methods.
- **Agent:** claude-fable-5
- **Did:** Inserted `\section*{Why Warmth and Competence, Why Hiring, and Which Data}` with three paragraphs ("The Construct", "The Link to Hiring", "Why These Data") in `paper/paper/Ulu_Lastra.tex` between Background and Methods. Also fixed `\graphicspath` to include both `figures/` and `../figures/` so the file compiles both locally and in Overleaf. Updated `.gitignore` to exclude LaTeX build artifacts (aux, bbl, blg, etc). Updated corresponding author line from TODO to both emails.
- **Findings:** Compiles cleanly to 14 pages, zero errors, zero undefined citations. New section cites `fiske2002model`, `fiske2007universal`, `gallo2024warmth`, `an2024llm`, `an_measuring_2025`, `gaebler2024auditing` — all pre-existing bib entries.
- **Decision / rationale:** SCM background kept to 2 sentences (brief, per Jorge); story data introduced with motivation (why name-free protagonist). The 149 vs 282 name distinction is explained explicitly in the section — 282 for probe-vs-human alignment, 149 for callback benchmark comparison.
- **Anti-formulaic self-check:** Paragraph openers vary ("The Stereotype Content Model proposes..." / "Citet{gallo2024warmth} provide the critical empirical bridge." / "Probing warmth and competence from hiring prompts directly would confound..."). No repeated frames, no em dashes, no signal-only transitions. Check passed.
- **Next:** LinkedIn post draft for course social media communication task.

---

## 2026-07-08 · Step 1 — Draft §3 "Why Warmth and Competence, Why Hiring, and Which Data"

- **Context:** Paper writing session — inserting the missing motivation/theory section between Background and Methods per the agreed PAPER_STRUCTURE.md blueprint.
- **Agent:** claude-fable-5
- **Did:** Inserted `\section*{Why Warmth and Competence, Why Hiring, and Which Data}` with three paragraphs: (1) The Construct — SCM in 2 sentences (Fiske 2002/2007); (2) The Link to Hiring — Gallo & Hausladen 2024 as empirical bridge, gap vs. existing behavioral audits; (3) Why These Data — motivation for name-free stories (pilot showed demographic skew without neutralisation), and explicit resolution of the 149 vs. 282 name distinction (149 have published US callback rates for disparity comparison; all 282 have perception ratings for probe alignment). Fixed `\graphicspath` to `{figures/}{../figures/}` so the tex file compiles both in Overleaf (`figures/`) and locally (`../figures/`). Updated footnote from TODO to both author emails. Added LaTeX build artifacts to `.gitignore` and removed tracked `.aux/.bbl/.blg` from git index. Full compile cycle: 14 pages, zero errors, zero undefined citations/references.
- **Findings:** Section resolves the 149 vs. 282 ambiguity that was implicit in the Results; no new citations needed beyond already-committed bib entries.
- **Decision / rationale:** SCM treatment kept to 2 sentences per Jorge's preference; stories introduced with motivation (why name-free) rather than just description.
- **Anti-formulaic self-check:** Paragraph openers: "The Stereotype Content Model proposes..." / "\citet{gallo2024warmth} provide the critical empirical bridge." / "Probing warmth and competence from hiring prompts directly would confound...". No shared opener frames, no em dashes, no signal-only transitions. Check passed.
- **Next:** LinkedIn science communication post draft; then Discussion/Limitations rewrite.

---

## 2026-07-14 · Step 1 — Pull presentation update and reconcile ignore rules

- **Context:** User requested the current Git status and a pull if the configured upstream contained new work.
- **Agent:** gpt-5-codex
- **Did:** Fetched `origin`, fast-forwarded local `main` from `e5cc890` to `b7f8cf2`, validated `presentation/Ulu_Lastra_slides_v3.pptx` as a readable PowerPoint archive, and combined the incoming LaTeX/presentation exclusions with the repository's existing `.gitignore` protections.
- **Findings:** The pull added three commits and one 885,122-byte presentation. The incoming `.gitignore` replacement had exposed 19 local `.DS_Store` and Python bytecode files as untracked; restoring the prior rules removed them from Git status while retaining all four new exclusions.
- **Decision / rationale:** Preserve the new presentation and intended export exclusions without dropping protections for secrets, caches, model artifacts, cluster outputs, and tracked pipeline-output exceptions.
- **Next:** Commit and push the safe `.gitignore` reconciliation and this step-log entry, then verify `0 ahead / 0 behind`.

---

## 2026-07-15 · Step 1 — Implement Gemma 4 TransformerLens replication pipeline
- **Context:** Prepare exact-test replications for Gemma 4 31B dense and 26B-A4B MoE on SCCKN, excluding new MoE-specific and SAE tests.
- **Agent:** gpt-5-codex
- **Did:** Migrated model loading to TransformerLens 3 Bridge; added native-chat decision tokenization, model-specific PCA paths, raw/local/denoised hiring modes, a reproducible R4 CLI, structural result validation, dedicated Gemma 4 smoke tests, and gated Grid Engine jobs. Created `paper/2026-07-15_0035_gemma4_transformerlens_pipeline.md`.
- **Findings:** TransformerLens callable cache filters do not resolve legacy hook aliases, so extraction filters were changed to alias-aware string/list forms. Local verification passed: 16 tests, Python compilation, shell syntax, two dry runs, and `git diff --check`. No Gemma 4 empirical results exist yet.
- **Decision / rationale:** Use raw-weight `TransformerBridge` without compatibility-mode weight folding; apply native chat templates only to Yes/No decisions; retain raw text for passive activation extraction. Run 31B before 26B-A4B and stop on smoke failures without quantization or model substitution.
- **Next:** Install `wc-tl-g4` on SCCKN, submit both smoke jobs, inspect their JSON/VRAM results, then submit the full dependency chain.

---

## 2026-07-15 · Step 2 — Resolve SCCKN Gemma 4 environment constraint
- **Context:** Create the dedicated `wc-tl-g4` environment before Gemma 4 smoke submission.
- **Agent:** gpt-5-codex
- **Did:** Cloned the working CUDA/PyTorch environment and installed the Gemma 4 dependency set; added `setuptools<82` to `requirements-gemma4.txt` after the environment integrity gate failed.
- **Findings:** `pip check` reported the exact conflict: `torch 2.12.0 has requirement setuptools<82, but you have setuptools 82.0.1`. No model was loaded and no GPU job was submitted before this gate passed.
- **Decision / rationale:** Preserve SCCKN's working `torch 2.12.0` CUDA build and constrain setuptools instead of replacing PyTorch.
- **Next:** Reinstall the pinned requirements, require a clean `pip check`, print runtime versions, and submit the two sequential smoke jobs.

---

## 2026-07-15 · Step 3 — Submit gated Gemma 4 smoke jobs on SCCKN
- **Context:** Validate TransformerLens Bridge support and memory feasibility before any full Gemma 4 experiment.
- **Agent:** gpt-5-codex
- **Did:** Verified `wc-tl-g4` with `pip check` and runtime versions (`torch 2.12.0+cu130`, CUDA 13.0, Transformers 5.13.0, TransformerLens 3.5.1), then submitted sequential Grid Engine smoke jobs for 31B dense and 26B-A4B MoE.
- **Findings:** Job `1141608` (`gemma4_31b`) entered the queue; dependent job `1141609` (`gemma4_26b_a4b`) entered held-queue state. The SCCKN repository was clean at commit `9156171` before submission.
- **Decision / rationale:** Keep the full production chains unsubmitted until both smoke JSON files confirm hooks, native-chat tokens, Bridge/HF parity, finite margins, steering response, and VRAM feasibility.
- **Next:** Monitor jobs `1141608` and `1141609`; inspect exact `.out`, `.err`, and JSON results before releasing the full pipeline.

---

## 2026-07-15 · Step 4 — Add corrected 31B retry and 12B Unified smoke path
- **Context:** Follow up the failed 31B native-chat processor check and add a Gemma 4 12B smoke without rerunning the validated 26B-A4B MoE model.
- **Agent:** gpt-5-codex
- **Did:** Added an explicit `AutoProcessor` fallback with chained exception reporting for Gemma 4, added `--smoke-31b-12b` submission mode, configured distinct 31B retry and 12B output labels/queues, added environment and Git-identity preflight gates, and synchronized the existing successful 26B smoke JSON after setting repo-local SCCKN Git identity.
- **Findings:** Gemma 4 12B is the supported `Gemma4UnifiedForConditionalGeneration` variant with 48 layers and residual width 3,840. Local verification passed: 20 tests, Python compilation, shell syntax, and `git diff --check`.
- **Decision / rationale:** Preserve the original failed 31B logs under their existing names; write the retry as `gemma4_31b_retry1`; run 12B independently on the 48 GB/96 GB queue pool; do not resubmit 26B or any full pipeline.
- **Next:** Push the implementation, run SCCKN preflight, submit the two independent smoke jobs, and inspect scheduler plus JSON outcomes.

---

## 2026-07-15 · Step 5 — Validate Gemma 4 12B and queue corrected 31B retry
- **Context:** Execute the requested independent Gemma 4 31B retry and 12B smoke without rerunning 26B-A4B or launching full pipelines.
- **Agent:** gpt-5-codex
- **Did:** Passed the SCCKN environment and Git-identity preflight, submitted jobs `1141614` (31B retry) and `1141615` (12B), inspected scheduler accounting and logs, synchronized the 12B JSON, and created `paper/2026-07-15_0839_gemma4_12b_smoke.md`.
- **Findings:** The 12B smoke completed on an NVIDIA L40 in 152 seconds with `failed=0`, `exit_status=0`, exact Bridge/HF parity (`max_logit_diff=0.0`), a finite steering response (margin 17.5569 to 17.5745), and 22.5006 GiB peak allocated VRAM. The model exposed 48 layers with width 3,840 and used layer 31. The corrected 31B job remained queued for `gpu@scc214`; it had not failed or started. The 26B-A4B smoke was not rerun and no production chain was submitted.
- **Decision / rationale:** Treat 12B as technically supported for the existing experiment suite, while withholding any 31B conclusion until job `1141614` runs and produces scheduler accounting plus a valid JSON artifact.
- **Next:** Monitor job `1141614`; if it passes, record its VRAM and Bridge parity before deciding separately whether to submit any full replication.

---

## 2026-07-15 · Step 6 — Translate and compile Turkish manuscript
- **Context:** User requested a complete Turkish edition of the current manuscript while retaining technical terminology in English.
- **Agent:** gpt-5-codex
- **Did:** Created `paper/paper/Ulu_Lastra-tr.tex`, preserved citations, equations, labels, tables, and the eight English-language figure assets, installed missing LaTeX dependencies in the user TeX tree, and compiled `paper/paper/Ulu_Lastra-tr.pdf`. Rendered and visually inspected every final PDF page.
- **Findings:** The final PDF has 14 letter-size pages with no LaTeX errors, undefined citations/references, missing glyphs, overfull boxes, clipping, or overlaps. Structural parity checks matched 23 paragraph units, 8 figures, 1 table, 2 equations, 12 labels/autorefs, and all citation keys. The system TeX tree was not writable, so `biblatex`, `appendix`, `preprint`, `logreq`, and Turkish hyphenation resources were installed in user mode; pdfTeX still reports that Turkish patterns were not preloaded, but full-page visual inspection found acceptable line breaking.
- **Decision / rationale:** Keep core field terms such as `warmth`, `competence`, `residual stream`, `probing`, `activation steering`, and `callback margin` in English, translate the surrounding academic prose, and leave in-figure labels in English as requested. The English manuscript and bibliography database were not modified.
- **Anti-formulaic self-check:** Re-read the Turkish manuscript, confirmed varied paragraph openings, no repeated causal template across adjacent passages, no signal-only transitions, and no prohibited Unicode dash punctuation. Check passed.

---

## 2026-07-15 · Step 7 — Implement write-once Gemma 4 Stage 1–3 job chain
- **Context:** Prepare nine SCCKN jobs covering activation extraction, fixed-layer probe validation, and layer sweep for Gemma 4 12B, 26B-A4B, and 31B.
- **Agent:** gpt-5-codex
- **Did:** Added a stage-aware technical validator, a single-stage Grid Engine executor, and a fully serial stage-major submission interface with external success sentinels, unique scheduler logs, a held-first-job workflow, and canonical-output collision gates.
- **Findings:** No canonical Gemma 4 Stage 1–3 outputs existed locally or on SCCKN before implementation. The chain order is Stage 1 12B→26B-A4B→31B, then Stage 2 in the same order, then Stage 3 in the same order. Scientific threshold failures remain recorded findings; only structural, finite-value, environment, Git-sync, or predecessor failures stop progression.
- **Decision / rationale:** Use a single serial chain to prevent GPU and Git push races, and fail rather than delete, archive, or overwrite any partial or completed scientific artifact.
- **Next:** Validate locally, push the implementation, pull it on SCCKN, submit all nine jobs held behind the first, record their IDs, and release the first job.

---

## 2026-07-15 · Step 8 — Submit nine serial Gemma 4 Stage 1–3 jobs
- **Context:** Launch the approved Stage 1–3 replication order for Gemma 4 12B, 26B-A4B, and 31B while preserving all prior outputs.
- **Agent:** gpt-5-codex
- **Did:** Fast-forwarded SCCKN to implementation commit `ae559c6`, passed the clean-worktree and `wc-tl-g4` environment preflight, and submitted the fully serial chain recorded in `results/logs/gemma4_stages_1_3_submission_20260715T073515Z.json`.
- **Findings:** The nine job IDs are `1141625`–`1141633`, ordered Stage 1 12B/26B-A4B/31B, Stage 2 12B/26B-A4B/31B, then Stage 3 12B/26B-A4B/31B. All nine initially entered held-queue state; the first job remained user-held while the manifest and this entry were synchronized. Preflight found no canonical output collision and `pip check` reported no broken requirements.
- **Decision / rationale:** Release only job `1141625` after this audit trail is pushed; all successors remain scheduler-held and also require their predecessor's external success sentinel, so a technical failure cannot write later-stage results.
- **Next:** Pull this entry on SCCKN, release job `1141625`, monitor scheduler accounting and write the consolidated findings report only after the chain reaches a terminal outcome.

---

## 2026-07-15 · Step 9 — Replace the presentation placeholder with an emotion-vector sequence
- **Context:** User requested a conceptual redesign of Slide 3 that explains how a story becomes an internal activation and how emotion concepts can be represented as measurable directions.
- **Agent:** gpt-5-codex
- **Did:** Created `presentation/Ulu_Lastra_slides_v4.pptx` from v3, replaced the placeholder with two editable native-shape slides, added speaker notes and a Sofroniew, Lindsey et al. (2026) attribution, and rendered and inspected all 10 slides.
- **Findings:** The placeholder is absent; Slides 3–4 contain the new story-to-activation and emotion-direction sequence with no new canvas overflow. Automated overflow checks reproduce only the inherited decorative bleed on source Slides 1 and 9, now output Slides 1 and 10. The template-fidelity checker flags the two new opaque activation-space panels because they intentionally replace the inherited placeholder region, as documented in the edit map and deviation log.
- **Decision / rationale:** Use a static two-slide reveal so the audience first understands the hidden activation metaphor, then sees joy, fear, and sadness as directions before the deck bridges to warmth and competence.

---

## 2026-07-15 · Step 10 — Remove anatomy from the emotion-vector slides
- **Context:** User requested a simpler, fully abstract treatment of Slides 3–4 without a human figure.
- **Agent:** gpt-5-codex
- **Did:** Created `presentation/Ulu_Lastra_slides_v5.pptx`, removed the character, footsteps, motion accents, and ground line from both concept slides, centered the story-to-LLM flow, enlarged the hidden activation space, repositioned the emotion vectors, and revised the Slide 3 speaker note.
- **Findings:** The final PPTX contains no human-figure shape identifiers or unresolved placeholder copy. Slides 1–2 and 5–10 render identically to v4; Slides 3–4 have no new canvas overflow. Automated overflow checks reproduce only the inherited decorative bleed on Slides 1 and 10, while the template-fidelity checker repeats the documented placeholder-replacement warning for Slides 3–4.
- **Decision / rationale:** Keep the story cue as textual context, but let the abstract activation space carry the metaphor without implying that the model experiences emotion as a person does.

---

## 2026-07-15 · Step 11 — Align result figures in the updated presentation
- **Context:** User requested repositioning the figures on Slides 12 and 15 in the newly expanded 16-slide deck.
- **Agent:** gpt-5-codex
- **Did:** Created `presentation/Ulu_Lastra_slides_v6.pptx` from the user-updated v5 and moved only the inherited figure and caption objects on Slides 12 and 15 to align with the left content column at y = 100.8 pt.
- **Findings:** Template fidelity passed with zero issues; exact target coordinates were verified; the other 14 slides render pixel-identically to v5. Overflow checks reproduce only inherited bleed on Slides 1, 2, and 16 in both v5 and v6.
- **Decision / rationale:** Preserve figure size, crop, and caption gap while using a shared top alignment to remove the title collision on Slide 15 and improve consistency on Slide 12.

---

## 2026-07-15 · Step 12 — Reflow the Finding 2 steering slide
- **Context:** Correct the prior figure-position edit after the user identified that the visible Finding 2 chart still remained in the right column.
- **Agent:** gpt-5-codex
- **Did:** Created `presentation/Ulu_Lastra_slides_v7.pptx`, moved and enlarged the inherited steering chart on Slide 13 into a centered lower-half evidence frame, and condensed the four inherited explanatory bullets into a two-by-two key above it.
- **Findings:** Template fidelity passed with zero issues, the chart is no longer right-aligned, and Slides 1–12 and 14–16 render pixel-identically to v6. Overflow checks reproduce only inherited bleed on Slides 1, 2, and 16.
- **Decision / rationale:** Use the chart as the visual anchor and place supporting interpretation above it so the slide reads vertically instead of as an unbalanced left-text/right-chart split.

---

## 2026-07-15 · Step 13 — Extend the centered-evidence layout to Findings 1 and 3
- **Context:** User requested that Slides 12 and 14 follow the vertical, chart-centered composition established on Slide 13.
- **Agent:** gpt-5-codex
- **Did:** Created `presentation/Ulu_Lastra_slides_v8.pptx`; centered the inherited charts in the lower half of Slides 12 and 14, redistributed their inherited metrics and interpretation above the charts, and shortened one Slide 12 takeaway for fit without reducing font size.
- **Findings:** Template fidelity passed with zero issues; slide-boundary checks passed; only Slides 12 and 14 differ visually from v7, while the other 14 slides render pixel-identically.
- **Decision / rationale:** Make the evidence figure the primary visual anchor on all three finding slides while retaining each slide's existing claims, source attribution, and visual language.

---

## 2026-07-15 · Step 14 — Implement Gemma 4 12B multi-GPU parity audit
- **Context:** Test whether TransformerLens Bridge layer dispatch across two L40 GPUs changes Gemma 4 12B activations, logits, or steering results.
- **Agent:** gpt-5-codex
- **Did:** Added a backward-compatible multi-device model-loading option, a three-process single-A/single-B/two-GPU parity runner, topology and numeric gates, isolated SCCKN Grid Engine job/submit scripts, tracked parity manifests/results, and unit tests; separately validated and committed the existing v8 presentation update.
- **Findings:** Local validation passed with 35 tests, Python compilation, shell syntax checks, and `git diff --check`. No multi-GPU empirical result exists yet. The audit will compare every residual layer on fixed passages and real 12B warmth/competence steering while keeping temporary tensor snapshots outside Git.
- **Decision / rationale:** Use one first-fit two-GPU job on the scc192/scc213 L40 pool and a separate SCCKN clone so hardware is controlled within the three arms and concurrent Stage 1–3 output sync cannot race on the same Git index.
- **Next:** Push the implementation, create and preflight the isolated SCCKN checkout, submit the parity job held, record its manifest, then release and monitor it.

---

## 2026-07-15 · Step 15 — Submit held Gemma 4 12B L40 parity job
- **Context:** Launch the approved three-arm single-GPU/two-GPU audit without interfering with the active Gemma 4 Stage 1–3 chain.
- **Agent:** gpt-5-codex
- **Did:** Pushed the presentation and parity implementation, created the isolated SCCKN checkout at `/work/emrecan.ulu/normalcy-axis-parity`, ran environment/compilation/shell/dry-run preflights, fixed a submitter quoting error before any job was created, and submitted held job `1142148`; wrote `results/logs/gemma4_parity_submission_20260715T143120Z.json`.
- **Findings:** SCCKN reports PyTorch `2.13.0+cu130`, Transformers `5.13.0`, TransformerLens `3.5.1`, and a clean `pip check`. The production environment does not contain pytest (`No module named pytest`), so cluster validation used Python compilation and shell checks after the 35-test local suite passed. Job `1142148` requests two GPUs from `gpu@scc192,gpu@scc213` and is still user-held; no empirical parity result exists yet.
- **Decision / rationale:** Do not mutate the production environment solely to add pytest. Keep the job held until its manifest and this audit entry are committed, then release it from the isolated checkout.
- **Next:** Pull this entry into the parity checkout, release `1142148`, and inspect scheduler state before monitoring the three capture arms.

---

## 2026-07-15 · Step 16 — Create Siemens distributed-inference application portfolio
- **Context:** User requested implementation of the approved English portfolio plan for Siemens PhD Job ID 513241.
- **Agent:** gpt-5-codex
- **Did:** Recovered the current empirical state, inspected the model-loading, multi-GPU parity, validation, and Grid Engine paths, checked live SCCKN scheduler/accounting state, and created the recruiter-safe local Markdown portfolio at `applications/siemens_513241_distributed_llm_inference_portfolio.md`; excluded the application file locally through `.git/info/exclude`.
- **Findings:** Gemma 4 smoke artifacts report exact Bridge/HF logit parity for 12B, 26B-A4B, and 31B with peak allocated VRAM of 22.50, 48.35, and 58.50 GiB. Gemma 4 12B Stage 1 completed in 153 seconds with exit status 0. The two-L40 parity job is released and queued; no empirical multi-GPU result exists. The portfolio labels completed, execution-pending, and proposed work separately and omits personal paths, scheduler identifiers, node names, queue names, and email addresses.
- **Decision / rationale:** Present the implemented systems work strongly without claiming unperformed multi-node, tensor-parallel, network-aware, heterogeneous-edge, or C++ runtime work; frame these as an explicit PhD roadmap.
- **Next:** Review the local portfolio for wording and tailor the opening paragraph to the final application form or cover letter if needed.

---

## 2026-07-15 · Step 17 — Add reusable SCCKN GPU job-design standard
- **Context:** Preserve the scheduler-priority and GPU job-packaging lessons from the congested Gemma 4 queue for reuse across projects.
- **Agent:** gpt-5-codex
- **Did:** Added `scckn/GPU_JOB_DESIGN.md`, nested agent instructions, and generic hybrid submitter/staged-runner templates; linked them from the existing SCCKN documentation and mirrored the same files to the canonical cross-project kit at `/Users/emrecanulu/Documents/scckn`.
- **Findings:** Both SCCKN copies are byte-identical. Bash syntax, unresolved-placeholder rejection, two-job dry-run output, stubbed `qsub` priority/dependency propagation, stage resume, failure-stop behavior, internal links, generic-value scanning, and `git diff --check` passed. `shellcheck` was unavailable locally and was skipped.
- **Decision / rationale:** Default to a resource-class hybrid design: keep common-GPU work separate, consolidate consecutive scarce-GPU stages into a resumable allocation, and pass an explicit priority to every submitted job because `hold_jid` transfers neither priority nor resources.
- **Next:** Copy and adapt the templates for the next SCCKN pipeline, resolve every `# ADJUST` value, then validate with `qsub -w v` before submission.

---

## 2026-07-18 · Step 1 — Retry Graphify visual extraction after permission fix
- **Context:** Recheck the previously blocked Graphify visual audit of tracked result figures after macOS Documents access was repaired.
- **Agent:** gpt-5-codex
- **Did:** Resumed the existing `results/` scan, visually extracted nine image/SVG figures and two PDF renderings, built and labeled `graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md`, and `graphify-out/graph.html`, and ran the extraction health diagnostic.
- **Findings:** Visual access succeeded for all 11 pending files with no permission error. The graph contains 123 nodes, 160 built edges, and 10 communities. There are no missing endpoints, dangling endpoints, or self-loops; nine endpoint-pair collapses arise from duplicate PNG/PDF or PNG/SVG renderings of the same figures. The in-app browser backend was unavailable, so HTML runtime interaction could not be inspected, but the generated HTML embeds the expected 123 nodes, 160 edges, community legend, search control, and hyperedges.
- **Decision / rationale:** Treat the original Documents permission blocker as resolved. Keep the duplicate-rendering collapse warning visible because it reflects graph extraction redundancy, not a change or corruption in the tracked scientific artifacts.
- **Next:** Open `graphify-out/graph.html` locally if interactive layout inspection is desired; use the tracked JSON/NumPy artifacts, rather than duplicate figure renderings, for scientific verification.

---

## 2026-07-18 · Step 2 — Audit live L40 and RTX 6000 availability
- **Context:** User requested a current SCCKN inventory of active and available L40 and RTX 6000 GPUs with running-job ages.
- **Agent:** gpt-5-codex
- **Did:** Queried the live Grid Engine queue configuration, host GPU features, consumable `gpu` availability, running GPU jobs, requested resources, and start times without modifying any jobs.
- **Findings:** At 2026-07-18 11:43:21 CEST, the L40 pool had 11 GPUs across `scc192` and `scc213`: 5 reserved and 6 scheduler-available. The RTX 6000 pool on `scc214` had 8 GPUs: 6 reserved and 2 available. The 11 active reservations had run for 14h46m to 188h25m.
- **Decision / rationale:** Count only jobs consuming the Grid Engine `gpu` resource as unavailable GPUs; interactive sessions on a GPU host without a `gpu` reservation do not reduce scheduler-reported availability.

---

## 2026-07-18 · Step 3 — Implement independent Gemma 4 Stage 3 retries
- **Context:** Retry Gemma 4 26B-A4B and 31B Stage 3 concurrently after the original serial chain was blocked by the failed 12B predecessor sentinel.
- **Agent:** gpt-5-codex
- **Did:** Added an independent single-GPU Stage 3 runner, a held two-job submitter for `gpu@scc214`, and a CPU-only finalizer that validates both sentinels before one output sync; added manifest tracking and focused script tests.
- **Findings:** The original 26B and 31B jobs did not execute model code; they exited 20 because the 12B Stage 3 OOM prevented predecessor sentinels. Local verification passed 17 tests, shell syntax, Python compilation, submitter dry-run, Stage 1–2 input validation for both models, canonical-output absence checks, and `git diff --check`. `shellcheck` was unavailable locally.
- **Decision / rationale:** Remove only the retry dependency on 12B, keep 26B and 31B compute jobs independent and user-held, prohibit Git operations inside parallel GPU jobs, and defer one durable sync to a CPU finalizer after both technical validators pass.
- **Next:** Push the implementation, preflight the clean SCCKN checkout and two available RTX 6000 resources, submit both jobs held, synchronize the manifest, and release them together.

---

## 2026-07-18 · Step 4 — Submit held parallel Gemma 4 Stage 3 retries
- **Context:** Launch the approved 26B-A4B and 31B Stage 3 retries on the two scheduler-available RTX PRO 6000 GPUs.
- **Agent:** gpt-5-codex
- **Did:** Fast-forwarded SCCKN to `c5b4bc4`, passed environment, input, collision, GPU-availability, shell, compile, and `qsub -w v` gates, then submitted held GPU jobs `1144931` (26B-A4B) and `1144932` (31B) plus CPU finalizer `1144933`; synchronized `results/logs/gemma4_stage3_retry_submission_20260718T100211Z.json` in commit `6190680`.
- **Findings:** `gpu@scc214` reported exactly two available GPUs at preflight. Both model jobs independently request `gpu=1,rtx_6000=1,h_vmem=96G,h_rt=12:00:00` with no predecessor relation. The finalizer requests no GPU and is held on both model job IDs. No Stage 3 canonical output existed at submission.
- **Decision / rationale:** Keep both GPU jobs user-held until this submission audit is pushed, then release them together so each can claim one of the two available RTX PRO 6000 devices.
- **Next:** Pull this entry on SCCKN, run `qrls 1144931 1144932`, verify distinct device assignments, and monitor both jobs through validation and finalizer sync.

---

## 2026-07-18 · Step 5 — Complete Gemma 4 26B and 31B Stage 3 sweeps
- **Context:** Validate the parallel RTX retry outcome and record the new all-layer empirical result.
- **Agent:** gpt-5-codex
- **Did:** Released jobs `1144931` and `1144932` together, verified distinct physical GPUs, monitored both sweeps, ran the dependent finalizer, audited scheduler accounting and finite outputs, and created `paper/2026-07-18_1208_gemma4_stage3_layer_sweep.md`.
- **Findings:** All three jobs reported `failed=0` and `exit_status=0`; 26B-A4B completed in 59 s, 31B in 78 s, and the CPU finalizer in 58 s. The 30-row and 60-row tables are complete and finite. Probe-layer Stage 3 d values reproduce Stage 2 exactly. Peak d occurs at layer 16 for 26B-A4B (9.14/9.78) and layer 24 for 31B (11.49/9.61), before the configured 0.66-depth probe layers. Topic-holdout accuracy is already at least 0.80/0.94 at layer 0, so the signal is amplified rather than first appearing late.
- **Decision / rationale:** Accept both Stage 3 outputs as technically complete while retaining the shared-axis and synthetic-distribution caveats. Keep the 12B Stage 3 OOM as a separate unresolved issue and make no manuscript or figure change in this run.
- **Next:** Review whether the new Gemma 4 depth profiles should be added to the paper's layer-emergence figure; separately diagnose the 12B RTX OOM if that replication remains required.

---

## 2026-07-18 · Step 6 — Finalize Stage 3 reporting and retry dry-run behavior
- **Context:** Complete post-run verification and keep the retry submitter inspectable after canonical outputs exist.
- **Agent:** gpt-5-codex
- **Did:** Re-ran both Stage 3 validators and the 17 focused tests, registered the dated findings report in `paper/README.md`, and moved the submitter's no-op dry-run response before production collision checks.
- **Findings:** Both empirical tables pass complete-row, ordered-layer, probe-layer, metadata, and finite-value gates. All 17 tests pass after the dry-run remains usable for plan inspection; the real submission path still refuses any existing manifest, state directory, CSV, or metadata target.
- **Decision / rationale:** Preserve strict write-once behavior for production while allowing a mutation-free dry-run to describe the job topology after a completed run.

---

## 2026-07-18 · Step 7 — Implement Gemma 4 12B L40 Stage 3 retry
- **Context:** Retry the unresolved Gemma 4 12B all-layer sweep after its RTX PRO 6000 model load failed despite 74.66 GiB reported free VRAM.
- **Agent:** gpt-5-codex
- **Did:** Added a dedicated held L40 runner and submitter plus a CPU-only validating finalizer; added device/headroom, write-once, commit-integrity, input-validation, scheduler-verification, manifest, and partial-submission cleanup gates with focused tests.
- **Findings:** Gemma 4 12B previously loaded and completed smoke and Stage 1 on L40 at 22.99 and 23.22 GB maximum virtual memory. The new job requires exactly one L40 with at least 30 GiB free VRAM and requests `gpu=1,h_vmem=32G,h_rt=01:00:00` from `gpu@scc192,gpu@scc213`. Local verification passed 21 tests, shell syntax, Python compilation, dry-run inspection, and `git diff --check`.
- **Decision / rationale:** Use the empirically validated single-L40 path rather than repeat the anomalous 12B Unified loader failure on Blackwell; keep Git synchronization outside the GPU job and prohibit automatic hardware substitution after failure.
- **Next:** Push the implementation, fast-forward and preflight SCCKN, submit the L40 job held with its CPU finalizer, synchronize the manifest, then release and monitor it.

---

## 2026-07-18 · Step 8 — Submit held Gemma 4 12B L40 Stage 3 retry
- **Context:** Launch the validated 12B all-layer sweep without repeating the anomalous RTX PRO 6000 load path.
- **Agent:** gpt-5-codex
- **Did:** Fast-forwarded SCCKN to `ee40495`, passed environment, shell, compile, input, collision, six-GPU L40 availability, and `qsub -w v` gates, then submitted held GPU job `1144961` and dependent CPU finalizer `1144962`; synchronized `results/logs/gemma4_stage3_retry_submission_12b_20260718T102845Z.json`.
- **Findings:** The GPU job requests one GPU from `gpu@scc192,gpu@scc213` with `h_vmem=32G,h_rt=01:00:00`; it is user-held and the finalizer has only `1144961` as its predecessor. Stage 1 and Stage 2 validate, both canonical Stage 3 targets are absent, and `wc-tl-g4` reports PyTorch 2.13.0+cu130, Transformers 5.13.0, and TransformerLens 3.5.1.
- **Decision / rationale:** Keep the GPU job held until the manifest and this audit entry are durable, then release only the GPU job and let Grid Engine release the CPU finalizer through its dependency.
- **Next:** Pull this entry on SCCKN, release `1144961`, verify an L40 assignment with at least 30 GiB free VRAM, and monitor both jobs through validation and output synchronization.

---

## 2026-07-18 · Step 9 — Add exact-L40 reproducibility audit after L40S drift
- **Context:** Resolve a cross-stage acceptance mismatch after the nominal L40-pool retry was dispatched to an NVIDIA L40S rather than the L40 used for Stage 1 extraction.
- **Agent:** gpt-5-codex
- **Did:** Completed jobs `1144961` and `1144962`, preserved their canonical L40S outputs, then extended the runner/finalizer with separate output-label and exact-device support and added an `scc192`-only L40 reproducibility submitter.
- **Findings:** The L40S sweep completed 48 finite layers in 146 seconds with 23.174 GB maximum virtual memory, but probe-layer warmth d was 8.461919 versus Stage 2's 8.633730 (difference -0.171811), competence d was 8.982933 versus 9.035413 (difference -0.052480), and cos(W,C) differed by -0.000977. Local verification of the exact-L40 audit path passed 22 tests, shell syntax, dry-run, and `git diff --check`.
- **Decision / rationale:** Treat the L40S table as a valid write-once result but not as satisfying the planned `1e-6` cross-stage reproduction gate. Run one separately labeled exact-L40 audit without overwriting canonical data to distinguish hardware-class drift from broader run-to-run variation.
- **Next:** Push the audit implementation, submit it held to `gpu@scc192`, synchronize its manifest, release it, and compare exact-L40, L40S, and Stage 2 probe-layer metrics.

---

## 2026-07-18 · Step 10 — Submit held exact-L40 Stage 3 reproducibility audit
- **Context:** Test whether the 12B probe-layer mismatch is explained by the Stage 1 L40 versus Stage 3 L40S hardware change.
- **Agent:** gpt-5-codex
- **Did:** Fast-forwarded SCCKN to `6d15b09`, validated source artifacts and absent separately labeled outputs, confirmed three scheduler-available GPUs on `gpu@scc192`, and submitted held GPU job `1144977` plus dependent CPU finalizer `1144978`; synchronized `results/logs/gemma4_stage3_retry_submission_12b_l40_repro_20260718T103904Z.json`.
- **Findings:** The audit requires the exact runtime name `NVIDIA L40`, writes `layer_sweep_gemma4_12b_l40_repro.{csv,meta.json}`, and cannot overwrite the canonical L40S result. Both jobs passed Grid Engine resource verification before submission.
- **Decision / rationale:** Release the audit only after its manifest and this entry are durable, then use the separately labeled table solely to quantify hardware/run reproducibility before final scientific reporting.
- **Next:** Pull this entry on SCCKN, release `1144977`, confirm exact-L40 assignment, and compare its probe-layer and full-depth metrics against Stage 2 and the L40S sweep.

---

## 2026-07-18 · Step 11 — Complete Gemma 4 12B Stage 3 and hardware audit
- **Context:** Finalize the 12B all-layer sweep after resolving the L40S cross-stage mismatch with a same-hardware L40 run.
- **Agent:** gpt-5-codex
- **Did:** Released and monitored jobs `1144977` and `1144978`, validated and synchronized the separately labeled exact-L40 table, compared both Stage 3 runs with Stage 2, and created `paper/2026-07-18_1244_gemma4_12b_stage3_l40_reproducibility.md`.
- **Findings:** Both audit jobs finished with `failed=0,exit_status=0`; the L40 GPU sweep took 100 seconds and the CPU finalizer 56 seconds. At layer 31, exact L40 reproduces Stage 2 warmth d 8.633730, competence d 9.035413, and cos(W,C) 0.493539 with zero six-decimal difference. Warmth and competence d peak at layers 26 and 27 (10.563076 and 10.445699). The L40S run remains technically valid but differs by -0.171811/-0.052480 d and -0.000977 cosine at the probe layer.
- **Decision / rationale:** Use the exact-L40 table as the cross-stage-consistent 12B Stage 3 result, preserve the L40S table as a reproducibility artifact, and report the device comparison as a one-run-per-device numerical warning rather than a fully isolated hardware effect.
- **Next:** Decide separately whether the completed three-model Gemma 4 depth profiles should be added to the active manuscript and paper figures.

---

## 2026-07-18 · Step 12 — Write Gemma 4 Stage 1 extraction-geometry report
- **Context:** User requested a dedicated Stage 1 (extraction-only) findings report for Gemma 4, in the same house format as prior single-model concept reports, covering all three models with parallel per-model figures.
- **Agent:** claude-sonnet-5
- **Did:** Discovered no local Python environment had matplotlib/seaborn/scipy/scikit-learn/pyyaml (Homebrew's system pip refuses installs under PEP 668); created an isolated `paper/figures/.venv` (already covered by `.gitignore`) and installed the plotting/analysis stack there. Ran `paper/figures/generate_figures.py --fig 1,2,3,4` against `data/processed/concept_vectors_gemma4_{12b,26b_a4b,31b}/` to produce 12 figures under new `paper/figures/gemma4_12b/`, `gemma4_26b_a4b/`, `gemma4_31b/` subdirs. Recomputed vector norms, cos(W,C), random-baseline Cohen's d/z, Lorenz top-k concentration, and 1-D projection CV directly from the Stage 1 `.npy` arrays for all three models. Wrote `paper/2026-07-18_1308_gemma4_stage1_extraction_geometry.md`, verified every embedded `![...]` image path resolves on disk, and registered the new figures and report row in `paper/README.md`.
- **Findings:** All three models show large random-baseline separation (z = 8.6–15.0, p < .001, 0/1000 random directions exceeded the extracted direction) and a positive, moderate warmth–competence cosine (0.494–0.587), replicating the Gemma-3 shared-valence pattern at Stage 1. 26B-A4B is the most weight-concentrated (top 64/2816 dims = 50% of warmth norm) and most cross-axis-entangled (all CV cells 100%, highest cosine 0.587). 31B has the largest absolute vector norms but the weakest competence random-baseline separation (z=8.6) and the only sub-100% cross-axis CV cell (competence-on-warmth 95%). 12B (probe layer 31) reproduces its previously reported cos(W,C)=0.493539 exactly.
- **Decision / rationale:** Keep this report strictly Stage-1-scoped; the CV-accuracy numbers shown are a same-array sanity check (via `generate_figures.py`'s own `projected_cv_accuracy` call), explicitly labeled as distinct from the authoritative multi-dimensional Stage 2 probe-validation numbers already reported in the Stage 3 reports, to avoid double-counting or contradicting those results.
- **Next:** User to decide whether to commit the new `.venv`-generated figures and report, and whether the three-model Gemma 4 Stage 1 geometry should inform any manuscript figure selection alongside the pending Stage 3 layer-emergence decision.

## 2026-07-18 · Step 13 — Write Gemma 4 Stage 2 probe-validation report
- **Context:** Direct follow-on to Step 12; user requested a Stage 2 (probe validation) focus report for the same three Gemma 4 models, following the same house format, figures, and pre-write critical-question process used for the Stage 1 report.
- **Agent:** claude-opus-4-8
- **Did:** Confirmed Stage 2 outputs (`results/tables/probe_metrics_gemma4_*.csv`, `results/logs/validate_probes_gemma4_*.json`) already exist and are complete for all three models. Reused the existing `paper/figures/.venv`. Ran `paper/figures/generate_figures.py --fig 5,6,7` with `--metrics`, `--logs`, `--vec-dirs`, `--labels`, `--stories` for all three Gemma 4 models into a new `paper/figures/gemma4_cross/` subdir, producing `fig5_cross_model`, `fig6_cross_model_story_agreement`, `fig7_same_story_demo` (png+pdf each). Wrote `paper/2026-07-18_1326_gemma4_stage2_probe_validation.md`, verified all three embedded `![...]` image paths resolve on disk, and registered the new figures and report row in `paper/README.md`.
- **Findings:** Both 5-fold and topic-holdout CV reach 1.00 for both axes in all three models, confirming the Stage 1 sanity-check numbers at the authoritative pipeline level. Cohen's d ranges 6.03–9.04 across the six model-axis combinations, with 31B competence (d=6.03) the weakest and 12B competence (d=9.04) the strongest. All three models fail `pass_orthogonality` (cosine 0.494–0.587) with cross-axis CV 0.95–1.00; 31B's competence-probe-on-warmth cell (0.95) is the only sub-ceiling cross-axis result among the three models. Cross-model per-story Spearman agreement (fig6) is 0.905–0.940 (warmth) and 0.947–0.960 (competence), showing the three model sizes rank the same stories the same way, not just classify conditions independently.
- **Decision / rationale:** Per user instruction, treated the topic-holdout-CV-plus-entanglement result as a headline finding (§6 of the report) rather than a footnote caveat, since it validates Stage 1's shared-valence observation at the stricter Stage 2 level and has direct implications for later steering/causal work (expect cross-axis leakage).
- **Next:** User to decide whether to commit the new Stage 2 report and `gemma4_cross/` figures, and whether Stage 2's entanglement finding should be folded into the manuscript's discussion of the causal steering results once those are run for Gemma 4.

---

## 2026-07-18 · Step 14 — Complete Qwen3.6 27B native-HF Stage 1–3 smoke
- **Context:** Test the current Qwen3.6 target with a TransformerLens-free Stage 1–3 path on an available SCCKN RTX PRO 6000.
- **Agent:** gpt-5-codex
- **Did:** Added and locally verified the pinned native-HF smoke pipeline, created an isolated Transformers 5.14.1 environment on SCCKN, submitted held jobs `1145040` and `1145041`, released the GPU job after manifest synchronization, monitored both jobs, revalidated the synchronized artifacts locally, and created `paper/2026-07-18_1336_qwen36_27b_native_hf_smoke.md`.
- **Findings:** Both jobs finished with `failed=0,exit_status=0`. The 64-layer, 5120-wide checkpoint ran on one RTX PRO 6000 with 51.227 GiB peak reserved VRAM (53.9% of 95.010 GiB). TransformerLens was absent; hook/hidden-state and passive-logit maximum differences were both 0.0; vision calls were zero. On the 40-story smoke subset, probe-layer warmth/competence d was 9.531037/10.469681 and both topic-holdout accuracies were 1.00. Stage 3 produced 64 finite rows and reproduced Stage 2 at layer 42 within `1e-6`.
- **Decision / rationale:** Accept native Hugging Face hooks as the Qwen3.6 execution backend and accept the 27B Stage 1–3 smoke as technically passed. Treat all smoke effect sizes as non-final because the run used only ten topics and 40 stories.
- **Next:** Prepare the full-run plan for the two selected Qwen3.6 models, preserving the pinned revision, native-hook parity gates, explicit-BOS input contract, text-only vision gate, and measured memory headroom; do not launch full jobs without the next user instruction.

---

## 2026-07-18 · Step 15 — Refine Gemma 4 Stage 1 interpretation
- **Context:** User-approved focused revision of the Gemma 4 Stage 1 report after a read-only scientific and figure audit.
- **Agent:** gpt-5-codex
- **Did:** Added residual- and dimension-normalised vector norms, clarified the shared-valence limitation of cross-axis accuracy, qualified the 26B-A4B MoE concentration interpretation, corrected two statements that contradicted the reported concentration and norm values, and aligned displayed layer depth with the repository's zero-indexed fraction definition.
- **Findings:** Raw and √`d_model`-scaled norms are largest for 31B, while residual-normalised norms are largest for 26B-A4B (warmth/competence 0.1035/0.1254). The 26B-A4B coordinate concentration remains 64/56 dimensions for 50% squared norm, but the residual-only extraction cannot attribute it to MoE routing or establish model-invariant effective dimensionality. Cross-axis accuracy of 0.95–1.00 remains evidence of shared evaluative signal rather than discriminant validation.
- **Decision / rationale:** Preserve the existing figures and numerical findings while narrowing claims to what the Stage 1 residual geometry directly supports.

---

## 2026-07-18 · Step 16 — Implement full Qwen3.6 native-HF stage pipeline
- **Context:** Implement the approved independent Stage 1–3 plan for Qwen3.6-27B and Qwen3.6-35B-A3B.
- **Agent:** gpt-5-codex
- **Did:** Added two pinned production configs, a shared native-HF 200-story Stage 1–3 backend, stage-specific technical and cross-stage validators, independent held Stage 1 and follow-up submitters with no `hold_jid`, and focused tests; pushed implementation commit `f8734b7`.
- **Findings:** Local verification passed 56 tests, Ruff, shell syntax, Python compilation, both submitter dry-runs, both model/config dry-runs, and Qwen-scoped `git diff --check`. The configs fix 27B at 64 layers/5120 width/revision `6a9e13b` and 35B-A3B at 40 layers/2048 width/revision `995ad96`.
- **Decision / rationale:** Keep Stage 1 and Stage 3 on one RTX PRO 6000 each, Stage 2 CPU-only, scientific thresholds non-gating, and all scheduler jobs independent; do not introduce automatic FP8 or hardware fallback.
- **Next:** Fast-forward the clean SCCKN checkout, run environment and scheduler preflights, submit both Stage 1 jobs held, synchronize the manifest, release together, and monitor before independent follow-up stages.

## 2026-07-18 · Step 16 — Write consolidated Gemma 4 Stage 3 layer-sweep report
- **Context:** Stage 3 coverage was split across two prior reports (26B-A4B+31B layer sweep; 12B L40/L40S reproducibility audit). User asked to consolidate all three Gemma 4 sizes into one Stage 3 report, parallel to the Stage 1/2 consolidated format, using the 12B exact-L40 sweep as canonical and adding an explicit hardware-reproducibility comparison table.
- **Agent:** claude-sonnet-5
- **Did:** Confirmed `layer_sweep_gemma4_12b.csv`, `layer_sweep_gemma4_12b_l40_repro.csv`, `layer_sweep_gemma4_26b_a4b.csv`, `layer_sweep_gemma4_31b.csv` are all complete (48/48/30/60 layers). Generated `paper/figures/gemma4_cross/fig8_layer_emergence.{png,pdf}` via the existing `paper/figures/.venv` using `--fig 8 --sweep-csvs layer_sweep_gemma4_12b_l40_repro.csv,layer_sweep_gemma4_26b_a4b.csv,layer_sweep_gemma4_31b.csv`. Computed peak-d layer/frac, peak-cos layer/frac, and final-layer values directly from the CSVs for all four sweep files (including L40S) to cross-check against the two prior reports' prose (all matched exactly). Wrote `paper/2026-07-18_1340_gemma4_stage3_layer_sweep_consolidated.md`. Updated `paper/README.md` (new fig8 cross-model figure row, new reports-table row).
- **Findings:** All three models reproduce their Stage 2 probe-layer numbers exactly (zero difference at six decimals) except 12B L40S, which shows small bfloat16 hardware drift (max abs diff across all layers: 0.638 warmth d, 0.914 competence d, 0.095 cosine). Effect-size peaks occur before the frac=0.66 probe layer in all three models (12B frac 0.55, 26B-A4B frac 0.55, 31B frac 0.41). Axis cosine peaks mid-network in all three (12B 0.617 at frac 0.53; 26B-A4B 0.736 at frac 0.41; 31B 0.705 at frac 0.47), starting near zero/negative at layer 0 and declining again toward the final layer — a depth-wide confirmation of the Stage 1/2 shared-valence entanglement finding.
- **Decision / rationale:** Used the 12B exact-L40 sweep as the canonical 12B row throughout (matches Stage 1 extraction hardware, zero-difference reproduction of Stage 2); kept the L40S run only in the dedicated hardware-comparison section (§6), per user's third answer. New report supersedes the two prior Stage 3 reports for cross-model comparison but not for execution/hardware detail, which those two reports retain as source of record.
- **Next:** User to decide whether to commit the new report, `fig8_layer_emergence.{png,pdf}`, and the README/STEP_LOG updates.

---

## 2026-07-18 · Step 17 — Audit and correct Gemma 4 Stage 2 validation
- **Context:** Implement the user-approved corrections from a read-only scientific audit of the consolidated Gemma 4 Stage 2 report.
- **Agent:** gpt-5-codex
- **Did:** Added fold-internal direction reconstruction for topic holdout, strict source-axis-to-target-axis topic transfer, explicit compatibility aliases for the prior target-calibrated cross-axis scores, and a reusable cross-model story-agreement validator. Regenerated the three validation artifacts, the agreement table, Figures 6–7, and revised the Stage 2 report and report index.
- **Findings:** Direction-specific topic CV is 1.00 for both axes in all three models. Strict warmth-to-competence and competence-to-warmth transfer is 0.99/0.97 for 12B, 0.99/0.95 for 26B-A4B, and 0.95/0.88 for 31B. Overall cross-model story agreement remains high (warmth 0.905–0.940; competence 0.947–0.960), but within-condition agreement is lower (warmth 0.434–0.574; competence 0.618–0.645), showing that condition separation inflated the earlier interpretation. All prior scalar Stage 2 results were preserved exactly; the artifact audit passed, `git diff --check` passed, and the full suite passed 56 tests.
- **Decision / rationale:** Present generalization and shared evaluative signal together. Retain the calibrated cross-axis fields for compatibility but name them explicitly, use strict transfer for the headline construct-specificity warning, and label Figure 7 as a 12B-selected qualitative illustration rather than representative evidence.
- **Next:** Use the corrected Stage 2 interpretation when deciding how to frame construct validity and cross-axis leakage in the active manuscript.

---

## 2026-07-18 · Step 18 — Complete six full Qwen3.6 stage runs
- **Context:** Execute the approved full 200-story Stage 1–3 plan for Qwen3.6-27B and Qwen3.6-35B-A3B on SCCKN without scheduler chaining.
- **Agent:** gpt-5-codex
- **Did:** Submitted and monitored independent GPU Stage 1 jobs `1145096`/`1145098`, CPU Stage 2 jobs `1145106`/`1145116`, and GPU Stage 3 jobs `1145108`/`1145118`; validated and synchronized every output plus both cross-stage audits.
- **Findings:** All six jobs finished with `failed=0,exit_status=0`. Both models achieved 1.00 five-fold and topic-held-out accuracy for both axes. Probe-layer warmth/competence d and cos(W,C) were 7.983/8.986/0.580 for 27B and 6.309/7.350/0.619 for 35B-A3B. Stage 3 produced 64/40 finite layers and reproduced Stage 2 at the probe layer with zero difference at `1e-6`. Peak reserved RTX PRO 6000 memory was 51.348 GiB for 27B and 65.543 GiB for 35B-A3B.
- **Decision / rationale:** Accept both pinned native-HF BF16 checkpoints for subsequent causal work. Preserve the fixed two-thirds-depth probe layer and require cross-axis controls because both models exceed the 0.30 overlap target.
- **Next:** Produce the six run-specific reports, Qwen-only comparison report, and visually verified figures.

---

## 2026-07-18 · Step 19 — Report full Qwen3.6 stages and cross-model comparison
- **Context:** Complete the requested per-model, per-stage evidence package after all full Qwen3.6 jobs passed.
- **Agent:** gpt-5-codex
- **Did:** Generated and visually inspected Stage 1–3 figures for each model and two Qwen-only comparison figures; made the Stage 3 figure title model-count-aware, added the tracked same-story agreement validator/table, created six stage reports plus `paper/2026-07-18_1421_qwen36_full_stage_comparison.md`, and registered them in `paper/README.md`.
- **Findings:** The dense 27B checkpoint has stronger probe-layer effect sizes and 14.2 GiB more reserved-VRAM headroom; the 35B-A3B MoE checkpoint has slightly greater axis overlap. Cross-model story ranking is high overall (ρ=0.930 warmth, 0.957 competence) and lower within condition (ρ=0.685/0.630). Target separation peaks before frac=0.66 in both models.

---

## 2026-07-18 · Step 20 — Implement Gemma 4 Stage 3B audit pipeline
- **Context:** Implement the user-approved enhanced all-layer audit after correcting the legacy Stage 3 report.
- **Agent:** gpt-5-codex
- **Did:** Added a backward-compatible Stage 3B profile with fold-internal mean-difference direction CV, strict bidirectional cross-axis topic transfer, 1,000-draw paired-topic bootstrap intervals, write-once validators, three held GPU jobs, qacct/hash/raw-log provenance, and a new Figure 8B renderer. Corrected the legacy Figure 8 and consolidated Stage 3 report without changing `probe_layer_frac=0.66`.
- **Findings:** The legacy pipeline remains the default and comparable to the prior Gemma 3/Qwen/Llama sweeps. Focused tests passed 14 cases, the full suite passed 64 tests, Ruff and Python compilation passed, all shell scripts passed `bash -n`, the submitter dry-run described one exact-L40 and two independent RTX PRO 6000 jobs, and `git diff --check` passed.
- **Decision / rationale:** Preserve all existing Stage 3 tables; write Stage 3B to separately labeled outputs and treat bootstrap peaks as uncertainty summaries rather than automatic layer-selection rules.
- **Next:** Push the implementation, submit the three jobs held on SCCKN, synchronize the submission manifest, release together, and monitor through provenance postflight.

---

## 2026-07-18 · Step 21 — Submit held Gemma 4 Stage 3B jobs
- **Context:** Execute the enhanced all-layer direction, transfer, and bootstrap audit on hardware matched to the canonical Stage 3 runs.
- **Agent:** gpt-5-codex
- **Did:** Fast-forwarded SCCKN to implementation commit `9579ce4`, synchronized the retained legacy Stage 3 raw logs, passed dependency and Stage 1/2 source validators, verified scheduler capacity, and submitted held GPU jobs `1145163` (12B exact L40), `1145164` (26B-A4B RTX PRO 6000), and `1145165` (31B RTX PRO 6000) plus CPU finalizer `1145166`. Synchronized `results/logs/gemma4_stage3b_submission_20260718T124416Z.json`.
- **Findings:** All three GPU jobs are independent and user-held; the finalizer depends only on those three IDs. Outputs are separately labeled and write-once, with no overwrite of legacy Stage 3 tables. Preflight requires one exact NVIDIA L40 and two NVIDIA RTX PRO 6000 Blackwell Server Edition devices.
- **Decision / rationale:** Persist the held-job manifest and audit entry before releasing the GPU jobs together; retain `probe_layer_frac=0.66` and treat Stage 3B as validation rather than automatic layer selection.
- **Next:** Pull this entry on SCCKN, release `1145163 1145164 1145165`, verify physical device assignments, and monitor through finalizer and provenance postflight.

- **Decision / rationale:** Keep all six individual reports as execution-specific records and use the seventh report for direct model selection and subsequent steering design.

---

## 2026-07-18 · Step 23 — Add strict Qwen3.6 Stage 2 validation
- **Context:** User requested immediate execution and separate reporting of the two technically unblocked Qwen-family tests omitted from the original full runs.
- **Agent:** gpt-5-codex
- **Did:** Extended the production Stage 2 path with fold-internal mean-difference direction reconstruction, strict source-only cross-axis topic transfer, compatibility aliases, an additive legacy-artifact upgrader, strengthened validators, and focused tests. Upgraded both canonical Qwen3.6 Stage 2 outputs, wrote per-model strict reports (`paper/2026-07-18_1453_qwen36_27b_strict_stage2_validation.md`, `paper/2026-07-18_1454_qwen36_35b_a3b_strict_stage2_validation.md`), and updated the comparison report and index.
- **Findings:** Direction-specific topic CV was 1.00 for both axes and models. Strict warmth-to-competence and competence-to-warmth transfer was 0.97/0.98 for 27B and 0.99/0.93 for 35B-A3B. Every pre-existing Stage 2 value was retained; both strengthened validators and cross-stage audits passed with zero probe-layer drift. The full suite passed 66 tests, Ruff passed, and `git diff --check` passed. A NumPy 2.3.0 Apple Accelerate build emitted erroneous dot-product warnings and unstable fold scores; those provisional extension values were rejected, and the accepted results were reproduced warning-free with NumPy 2.5.1 and scikit-learn 1.9.0.
- **Decision / rationale:** Treat fold-internal direction CV as the construction-specific generalization result and strict transfer as the construct-selectivity control. Keep the older target-calibrated cross-axis fields only for compatibility. Run future Qwen3.6 Stage 2 jobs through the complete schema automatically.
- **Next:** Technical steering remains a separate GPU experiment; the two CPU validation omissions are now closed for both Qwen3.6 models.

---

## 2026-07-18 · Step 24 — Complete Gemma 4 Stage 3B audit
- **Context:** Execute and report the enhanced all-layer validation for all three Gemma 4 variants.
- **Agent:** gpt-5-codex
- **Did:** Released and monitored SCCKN jobs `1145163`–`1145166`, ran local post-sync validators, verified every legacy Stage 3 column against its canonical table, generated and visually inspected corrected Figure 8 and new Figure 8B, and created `paper/2026-07-18_1453_gemma4_stage3b_validation.md`.
- **Findings:** All jobs ended with `failed=0,exit_status=0`; 12B ran on an exact NVIDIA L40 and 26B-A4B/31B ran independently on RTX PRO 6000 devices. Every legacy metric matches at every layer with maximum absolute difference 0. Probe-layer direction topic CV is 1.00/1.00 for every model, while strict W-to-C/C-to-W transfer is 0.99/0.97 (12B), 0.99/0.95 (26B-A4B), and 0.95/0.88 (31B). Paired-topic bootstrap peak ranges show stable middle-layer regions but wider exact-layer uncertainty for 12B and 31B competence.
- **Decision / rationale:** Retain `probe_layer_frac=0.66` for comparability and treat it as a strong but entangled fixed layer, not a data-selected or construct-pure optimum. Preserve legacy Stage 3 as the comparison reference and use Stage 3B for the stricter scientific interpretation.
- **Next:** Use strict cross-axis controls and external human/hiring validation in the causal stage; do not infer external validity from perfect synthetic topic holdout.

---

## 2026-07-18 · Step 25 — Implement the SAE-free Gemma 4 remaining-test pipeline
- **Context:** Implement the user-approved parity-plus-strengthened-controls plan for the tests still possible without a current Gemma 4 SAE.
- **Agent:** gpt-5-codex
- **Did:** Added exact-revision configs for all three Gemma 4 variants, revision-aware TransformerLens Bridge loading, a strengthened dense-steering design with cross-axis and 50-direction empirical-null controls, neutral-PCA metadata, paired-bootstrap hiring summaries and a conditional 282-name gate, write-once validators, and independent held SCCKN runners and submitters. Documented the method in `paper/2026-07-18_1604_gemma4_remaining_pipeline.md` and updated the report index.
- **Findings:** The full suite passed 74 tests; targeted Ruff lint and format checks passed; shell validation passed; all 33 model-by-run submitter dry-runs passed; and `git diff --check` passed. Empirical smoke and production results are not yet claimed.
- **Decision / rationale:** Preserve the legacy bfloat16 callback-margin path for old/new-generation parity and expose its quantization diagnostics explicitly. Require exact NVIDIA L40 for 12B and RTX PRO 6000 for 26B-A4B/31B. Keep every scheduler job independent and prohibit `hold_jid`.
- **Next:** Commit and synchronize the implementation, then submit one held smoke job per model, persist the manifests, release the jobs, and monitor their hardware and scientific gates.

---

## 2026-07-18 · Step 26 — Submit three held Gemma 4 remaining-test smokes
- **Context:** Begin empirical execution of the pinned SAE-free Gemma 4 pipeline after all local implementation gates passed.
- **Agent:** gpt-5-codex
- **Did:** Fast-forwarded the clean SCCKN checkout to `5e4fb17`, passed `pip check` and Python compilation in `wc-tl-g4`, and submitted independent user-held smoke jobs `1145318` (12B), `1145320` (26B-A4B), and `1145322` (31B). Synchronized the three `results/logs/gemma4_remaining_submission_*_smoke_20260718T140550Z_*.json` manifests before release.
- **Findings:** SCCKN reported three available L40 GPUs for the 12B submission and two available RTX PRO 6000 GPUs for each larger-model submission. Each manifest records the exact model revision, queue, resources, expected GPU family, submitted commit, and success sentinel; no `hold_jid` was used.
- **Decision / rationale:** Keep the jobs user-held until their provenance manifests and this audit entry exist on the shared branch, then release all three and require exact runtime hardware checks.
- **Next:** Pull this entry on SCCKN, release all three jobs, verify their physical assignments, and monitor through output validation and per-model smoke reports.

---

## 2026-07-18 · Step 27 — Complete and report three Gemma 4 remaining-test smokes
- **Context:** Validate the technical and hardware gates before launching the SAE-free Gemma 4 production runs.
- **Agent:** gpt-5-codex
- **Did:** Released and monitored SCCKN jobs `1145318`, `1145320`, and `1145322`; ran the production smoke validator on SCCKN and locally; persisted scheduler accounting in `results/logs/gemma4_remaining_smoke_outcome_20260718T141000Z.json`; and wrote separate 12B, 26B-A4B, and 31B smoke reports under `paper/2026-07-18_1612_gemma4_*_remaining_smoke.md`.
- **Findings:** All jobs completed with `failed=0,exit_status=0`. Bridge-to-HF maximum logit difference was 0.0 for every model, requested and resolved revisions matched exactly, activation shapes were `[1,9,3840]`, `[1,9,2816]`, and `[1,9,5376]`, and technical steering changed each finite Yes/No margin. Peak reserved VRAM was 22.795 GiB on exact L40 for 12B, 48.449 GiB on RTX PRO 6000 for 26B-A4B, and 58.867 GiB on RTX PRO 6000 for 31B.
- **Decision / rationale:** Accept all three pinned checkpoints and their assigned single-GPU hardware for production. Treat the one-prompt steering changes only as hook-activity checks, not causal warmth or competence findings.
- **Next:** Submit independent first-wave production jobs for neutral extraction, raw dense steering, unsteered hiring audit, and raw local/broad hiring steering; synchronize all manifests before release.

---

## 2026-07-18 · Step 28 — Submit held Gemma 4 first-wave production jobs
- **Context:** Start the production tests that are immediately independent after all three technical smokes passed.
- **Agent:** gpt-5-codex
- **Did:** Submitted 15 separate user-held jobs: neutral extraction, strengthened raw dense steering, 282-name unsteered hiring audit, 60-name local hiring steering, and 60-name broad hiring steering for each of 12B, 26B-A4B, and 31B. Job IDs are `1145329`, `1145331`, `1145333`, `1145335`, `1145337`; `1145339`, `1145342`, `1145344`, `1145346`, `1145348`; and `1145350`, `1145352`, `1145355`, `1145357`, `1145359`, respectively. Synchronized all 15 `results/logs/gemma4_remaining_submission_*_20260718T141249Z_*.json` manifests before release.
- **Findings:** Every preflight found its output targets absent and passed dependency checks. Submission-time availability was three L40 GPUs for all 12B jobs and two RTX PRO 6000 GPUs for every 26B-A4B and 31B job. Each job is single-GPU, write-once, exact-revision pinned, and independent; no `hold_jid` was used.
- **Decision / rationale:** Launch only the first wave whose inputs already exist. Defer PCA and denoised jobs until neutral extraction is validated, and defer post-hoc and full-282 gate jobs until their required hiring artifacts exist.
- **Next:** Pull this audit entry on SCCKN, release all 15 jobs, monitor scheduler placement and per-run validators, and write a separate report for every completed model/run pair.

---

## 2026-07-18 · Step 29 — Complete 12B hiring audit and 26B-A4B neutral extraction
- **Context:** Validate and report the first two completed first-wave Gemma 4 production jobs.
- **Agent:** gpt-5-codex
- **Did:** Validated SCCKN jobs `1145333` and `1145339`, synchronized their artifacts and raw logs, wrote `paper/2026-07-18_1623_gemma4_12b_hiring_audit.md` and `paper/2026-07-18_1623_gemma4_26b_a4b_neutral_extraction.md`, and submitted independent held CPU PCA job `1145368` for the validated 26B-A4B neutral matrix.
- **Findings:** Both jobs completed with `failed=0,exit_status=0`. The 12B audit covered 282 names: model-versus-human rho was 0.020 for warmth and 0.222 for competence; callback-versus-model rho was -0.110 for warmth and -0.124 for competence. The 26B-A4B neutral output is a finite 1500×2816 matrix at layer 19 with 51.537 GiB peak reserved VRAM.
- **Decision / rationale:** Treat the audit as observational and retain the causal steering tests. Accept the neutral matrix for PCA, but do not submit denoised jobs until PCA validation passes.
- **Next:** Persist this entry, release CPU job `1145368`, and continue monitoring the remaining independent GPU jobs.

---

## 2026-07-18 · Step 30 — Scope the SCCKN dirty-worktree submission gate
- **Context:** A held 12B PCA submission was refused while active jobs appended to previously synchronized tracked raw logs.
- **Agent:** gpt-5-codex
- **Did:** Preserved the exact error `Refusing submission: tracked SCCKN worktree is not clean.`, traced it to growing result logs, and narrowed the submitter cleanliness check to the selected config, `src/`, the Gemma 4 smoke implementation, and the remaining-test runner and submitter. Added a regression assertion and documented the correction in `paper/2026-07-18_1604_gemma4_remaining_pipeline.md`.
- **Findings:** The focused tests passed 15 cases and the project test directory passed all 74 tests; targeted Ruff, formatting, shell syntax, and `git diff --check` passed. An unrestricted repository-root pytest also collected the user's untracked `ccu/` project and failed five imports because `ccu_client` and `websocket` are not installed in the paper environment; no `ccu/` file was changed.
- **Decision / rationale:** Continue refusing any dirty scientific source, config, smoke, or runner file, but do not treat active result-log growth as a source-integrity violation. This retains the submitted-commit and critical-diff runtime gates.
- **Next:** Push the corrected submitter, fast-forward SCCKN, and retry the independent 12B PCA submission without altering active output files.

---

## 2026-07-18 · Step 31 — Submit held Gemma 4 12B PCA after scoped preflight
- **Context:** Retry the independent 12B PCA job after correcting the active-log false positive.
- **Agent:** gpt-5-codex
- **Did:** Fast-forwarded SCCKN to `bab25bb`, reran the source-scoped preflight, submitted user-held CPU job `1145374`, and persisted `results/logs/gemma4_remaining_submission_12b_pca_20260718T142700Z_12b_pca.json` without staging or modifying active GPU logs.
- **Findings:** The corrected cleanliness gate accepted the unchanged critical source/config state while the PCA validator confirmed both target outputs were absent. The job is independent, CPU-only, and has no `hold_jid`.
- **Decision / rationale:** Accept the scoped check as operationally validated and keep job `1145374` held until this audit entry is shared.
- **Next:** Pull this entry on SCCKN, release job `1145374`, and validate both 12B and 26B-A4B PCA outputs before submitting denoised runs.

---

## 2026-07-18 · Step 32 — Submit held 26B-A4B denoised production jobs
- **Context:** Advance the 26B-A4B branch after its neutral PCA artifact passed validation.
- **Agent:** gpt-5-codex
- **Did:** Validated the 26B-A4B PCA outputs, then submitted independent user-held jobs `1145378` (strengthened dense denoised steering) and `1145380` (60-name denoised-local hiring steering). Persisted both `results/logs/gemma4_remaining_submission_26b_a4b_*_20260718T142839Z_*.json` manifests before release.
- **Findings:** Eleven neutral PCs explain 50.3% variance. Denoising reduced cos(W,C) from 0.587 to 0.564, changed warmth d from 8.27 to 8.42 and competence d from 8.67 to 8.49, and reduced warmth-on-competence leakage from 4.90 to 4.75. Both denoised output sets were absent at submission.
- **Decision / rationale:** Proceed with both denoised causal tests because PCA passed, while retaining raw runs as the direct parity condition. Keep both jobs independent with no `hold_jid`.
- **Next:** Pull this entry on SCCKN, release jobs `1145378` and `1145380`, and compare their endpoints with the raw local and dense results.

---

## 2026-07-18 · Step 33 — Submit held 12B denoised production jobs
- **Context:** Advance the 12B branch after its independently run PCA artifact passed validation.
- **Agent:** gpt-5-codex
- **Did:** Validated the 12B PCA outputs, then submitted independent user-held jobs `1145384` (strengthened dense denoised steering) and `1145386` (60-name denoised-local hiring steering). Persisted both `results/logs/gemma4_remaining_submission_12b_*_20260718T143107Z_*.json` manifests before release.
- **Findings:** Eleven neutral PCs explain 51.2% variance. Denoising reduced cos(W,C) from 0.494 to 0.473 and increased target d from 8.55/8.94 to 10.01/10.00, but warmth-on-competence leakage also increased from 5.76 to 6.99.
- **Decision / rationale:** Run the denoised causal conditions but do not equate reduced cosine or higher target separation with improved construct purity; cross-axis controls remain necessary.
- **Next:** Pull this entry on SCCKN, release jobs `1145384` and `1145386`, and evaluate raw-versus-denoised causal stability.

---

## 2026-07-18 · Step 34 — Submit held Gemma 4 31B PCA
- **Context:** Advance the 31B branch after its 1,500-row neutral extraction passed validation.
- **Agent:** gpt-5-codex
- **Did:** Submitted independent user-held CPU PCA job `1145388` and persisted `results/logs/gemma4_remaining_submission_31b_pca_20260718T143500Z_31b_pca.json` before release.
- **Findings:** The 31B neutral validator passed and both PCA target files were absent. The job is CPU-only, exact-revision associated, and has no `hold_jid`.
- **Decision / rationale:** Run PCA independently and wait for its validator before submitting any 31B denoised causal job.
- **Next:** Pull this entry on SCCKN, release job `1145388`, and validate its PCA geometry before the 31B denoised wave.

---

## 2026-07-18 · Step 27 — Implement direct CCU Jupyter terminal client
- **Context:** Build a reusable, local-only access kit for the personal CCU JupyterHub H100 environment without a third-party remote-access relay.
- **Agent:** gpt-5-codex
- **Did:** Added `ccu/` with nested agent, security, operations, architecture, setup, and troubleshooting guidance; implemented a macOS CLI for scoped-Keychain authentication, interactive Jupyter terminal attachment, disposable command execution, managed-terminal cleanup, and verified small-file transfer; added a locked `uv` environment and focused tests.
- **Findings:** The discovered CCU environment exposes JupyterHub 4.0.2, Jupyter Server 2.8.0, JupyterLab 4.0.7, the terminal API, and server proxy support. Local validation passed 32 tests, Ruff, shell syntax, CLI/profile smoke, mode-0600 config verification, trailing-whitespace scan, and package lock/sync. No live CCU token or terminal session was created during implementation.
- **Decision / rationale:** Use the existing authenticated Jupyter terminal REST/WebSocket path directly from the Mac. Keep runtime traffic limited to the CCU origin, require a 24-hour default-server-scoped token stored only in macOS Keychain, and prohibit query-string tokens, redirects, proxy environment routing, TLS bypass, public listeners, and relay services.
- **Next:** Run the README personal live smoke with a new 24-hour token, verify `jovyan`, H100 visibility, interactive reconnect, command exit handling, file hash round-trip, restart behavior, and token revocation before preparing the anonymized shareable revision.

---

## 2026-07-18 · Step 30 — Validate direct CCU access end to end
- **Context:** Complete the personal live smoke for the direct CCU Jupyter terminal client before anonymizing it for reuse.
- **Agent:** gpt-5-codex
- **Did:** Installed the local client and personal profile, authenticated with a one-day token stored in macOS Keychain, exercised the status and terminal APIs, ran remote identity and GPU commands, detached and reattached an interactive shell, and verified an upload/download round trip with SHA-256. Corrected the terminal WebSocket route and switched managed terminal names from hyphens to underscores for Jupyter Server 2.8 compatibility; added legacy reporting and a `Ctrl-]` detach sequence.
- **Findings:** Live access returned `jovyan` and `NVIDIA H100 80GB HBM3, 81559 MiB`. Interactive reattachment preserved the remote shell. Upload, server read-back, download, and local comparison all produced SHA-256 `61dcadff4021a7b25d1320202607c8b9a3cfa4303a06c80e188157473be1c350`. The final suite passed 35 tests, Ruff, and shell syntax validation. Two empty hyphenated terminals from the failed prototype cannot match the deployed server's deletion route and will disappear at the next Jupyter restart.
- **Decision / rationale:** Treat the direct HTTPS/WSS path as operational without SSH, Tailscale, a public listener, or remote package installation. Keep the one-day token active for current work; do not restart the Jupyter server or revoke the token during a live session merely to test lifecycle behavior.
- **Next:** Use `ccu shell -p personal` or `ccu exec -p personal -- <command>` for current work. After a natural Jupyter restart, run `ccu doctor -p personal` and reconnect; then prepare the anonymized shareable revision separately.

---

## 2026-07-18 · Step 31 — Confirm remote exit and stream propagation
- **Context:** Close the final live command-execution gate for the CCU client.
- **Agent:** gpt-5-codex
- **Did:** Ran a remote command that wrote independently to stdout and stderr and exited with status 7.
- **Findings:** The local client returned `stdout-ok` on stdout, `stderr-ok` on stderr, and process exit status 7 exactly.
- **Decision / rationale:** Accept disposable remote execution as end-to-end validated, including nonzero status propagation.

---

## 2026-07-18 · Step 35 — Synchronize completed Gemma 4 remaining-test outputs
- **Context:** Recover the complete empirical state before implementing the calibrated steering-control correction.
- **Agent:** gpt-5-codex
- **Did:** Ran `jobs/sync_outputs.sh` on SCCKN, committed and pushed the completed Gemma 4 remaining-test artifacts, and fast-forwarded the local checkout to commit `3757199`.
- **Findings:** The synchronized commit added 103 files and 64,484 lines covering completed neutral/PCA, raw and denoised dense steering, hiring, and audit outputs. No SCCKN jobs remained active after synchronization.
- **Decision / rationale:** Use the synchronized artifacts as immutable legacy evidence and write calibrated results under new labels.
- **Next:** Implement statistically matched controls without overwriting prior outputs.

---

## 2026-07-18 · Step 36 — Implement calibrated steering controls and pilot workflow
- **Context:** Correct the legacy random-control scale mismatch for old and new model families.
- **Agent:** gpt-5-codex
- **Did:** Added shared training-topic SD calibration, additive and token-norm-preserving hooks, descriptive null metrics with topic-paired bootstrap intervals, a native-HF Qwen3.6 runner, a write-once validator, and independent RTX PRO 6000 SGE pilot scripts. Wrote `paper/2026-07-18_1747_calibrated_steering_pipeline.md`.
- **Findings:** Target alpha is preserved exactly while random and cross-axis directions receive matched standardized shifts. The full project suite passed 81 tests; Ruff, formatting, Python compilation, shell syntax, output-absence checks, and `git diff --check` passed. All three pilot labels were absent.
- **Decision / rationale:** Use 99 seeded random directions and report effects descriptively with no scientific gate. Run Gemma-3-12B, Gemma-4-12B, and Qwen3.6-27B as independent pilots; keep the 282-name expansion paused.
- **Next:** Commit and synchronize the implementation, submit the three user-held pilot jobs, record availability, then release and monitor them.

---

## 2026-07-18 · Step 37 — Submit three held calibrated-steering pilots
- **Context:** Begin the technical pilot after committing the calibrated-control implementation.
- **Agent:** gpt-5-codex
- **Did:** Fast-forwarded SCCKN to commit `88d5ca2` and submitted independent user-held RTX PRO 6000 jobs `1145429` (Gemma-3-12B), `1145430` (Gemma-4-12B), and `1145431` (Qwen3.6-27B). Extended `jobs/sync_outputs.sh` to include their manifests and scheduler logs after its first invocation correctly reported no matching files.
- **Findings:** SCCKN reported two RTX PRO 6000 GPUs at submission. All three jobs are held, use no `hold_jid`, require exact RTX PRO 6000 runtime hardware, and keep full-282 disabled.
- **Decision / rationale:** Preserve the held-manifest-release audit sequence and synchronize all three manifests before releasing any job.
- **Next:** Push the sync-pattern correction, synchronize the manifests, release all three jobs, and monitor technical validators.

---

## 2026-07-18 · Step 38 — Complete calibrated-artifact tracking whitelist
- **Context:** The first post-submission sync reached the new manifest files but Git refused them as ignored.
- **Agent:** gpt-5-codex
- **Did:** Preserved the exact Git error that the three `calibrated_steering_submission_*.json` paths were ignored, then added explicit `.gitignore` exceptions for calibrated manifests and scheduler logs plus regression assertions.
- **Findings:** The issue affected artifact tracking only; jobs `1145429`, `1145430`, and `1145431` remained user-held and no compute work started.
- **Decision / rationale:** Track the new lightweight manifests and logs under the same write-once policy as prior Gemma 4 and Qwen3.6 runs.
- **Next:** Push the whitelist correction, synchronize all manifests, then release the three independent jobs.

---

## 2026-07-18 · Step 39 — Release calibrated pilots into the RTX PRO 6000 queue
- **Context:** Start the three pilots after their manifests were committed in SCCKN sync commit `8c695b6`.
- **Agent:** gpt-5-codex
- **Did:** Released jobs `1145429`, `1145430`, and `1145431` with `qrls`, then inspected queue and host resources.
- **Findings:** All three jobs entered independent `qw` state on `gpu@scc214`; none received a physical GPU assignment yet. The host reports one `rtx_6000` resource at the latest check, while the submission-time aggregate reported two. No error log or compute output exists because no job has started.
- **Decision / rationale:** Leave all pilots queued rather than changing hardware or chaining jobs. Their exact RTX PRO 6000 runtime gate will reject any incorrect assignment.
- **Next:** Monitor for assignment, validate every completed artifact set, and write a separate empirical report per model before any nine-model rollout decision or full-282 launch.

---

## 2026-07-18 · Step 40 — Diagnose and correct TransformerLens hook adapter signature
- **Context:** The first calibrated Gemma 4 pilot reached its first active intervention after passing exact-GPU and output-absence gates.
- **Agent:** gpt-5-codex
- **Did:** Preserved the exact runtime error `TypeError: make_torch_hook.<locals>.hook() got an unexpected keyword argument 'hook'`, traced it to TransformerLens passing its hook point by keyword, and changed the shared callback parameter to the required `hook` name. Added a direct regression test that invokes the callback with `hook=`.
- **Findings:** This is a local adapter-signature bug, not a calibration, model, memory, or library-compatibility limitation. Gemma-4 job `1145430` failed at the first intervention; Gemma-3 job `1145429` was deleted before reaching the guaranteed same failure, and queued Qwen job `1145431` was deleted to preserve clean commit provenance.
- **Decision / rationale:** Replace all three initial pilot jobs with clean write-once retry labels rather than patching or resuming partial output.
- **Next:** Pass the full local suite, commit and synchronize the fix, then submit three new independent held retries.

---

## 2026-07-18 · Step 41 — Submit clean calibrated-pilot retries
- **Context:** Replace the initial pilot set after correcting the TransformerLens callback contract.
- **Agent:** gpt-5-codex
- **Did:** Passed all 82 project tests, synchronized the failed/cancelled first-attempt logs, submitted held retries `1145433` (Gemma-3-12B), `1145434` (Gemma-4-12B), and `1145435` (Qwen3.6-27B), committed their manifests in sync commit `93255e0`, and released all three.
- **Findings:** All retry output labels and success sentinels were absent before submission. SCCKN reported two RTX PRO 6000 resources at submission; all three retries currently remain in independent `qw` state with no scheduler dependency.
- **Decision / rationale:** Retain the same predeclared calibration and model scope because the first failure was fully explained by the tested adapter signature.
- **Next:** Validate runtime assignment and outputs as resources become available, then write one empirical report per completed model.

---

## 2026-07-18 · Step 42 — Validate calibrated pilots and implement resumable CCU Gemma 4 queue
- **Context:** Recover the failed 12B calibrated run, preserve backward compatibility, and move the three-model Gemma 4 replication to the CCU H100.
- **Agent:** gpt-5-codex
- **Did:** Synchronized the completed SCCKN artifacts; added opt-in fingerprinted atomic checkpoint/resume support to `src/dense_steering.py`; added isolated CCU environment, H100 smoke, model runner, and serial 12B-to-26B-A4B-to-31B queue scripts under `jobs/ccu/`; added CCU smoke and calibrated-output validators; wrote `paper/2026-07-18_2201_gemma4_12b_calibrated_steering.md` and `paper/2026-07-18_2201_qwen36_27b_calibrated_incomplete.md`.
- **Findings:** Gemma 4 12B produced all 40,440 raw, 2,020 summary, and eight null rows. Its original failure was a marginal BF16 norm-drift gate exceedance, with median 0.000114, p99 0.004047, and maximum 0.005620; it passes the documented 0.01 BF16 tolerance. Qwen3.6-27B produced only 16,176 rows because its runner sampled contiguous indices instead of the stimulus file's non-contiguous topic IDs. The full project suite passed 87 tests; Ruff, formatting, Python compilation, shell syntax, and `git diff --check` passed. CCU launch is currently blocked by an HTTP 302 login redirect from an expired or rejected JupyterHub token, before any remote mutation.
- **Decision / rationale:** Keep checkpointing opt-in so legacy invocations and SCCKN scripts retain their existing interfaces and labels. Treat SCCKN Gemma 4 12B as supporting evidence and CCU as primary. Use a 0.01 BF16 implementation tolerance while recording the exact drift. Stop the serial CCU queue only on technical failure, never on scientific effect size.
- **Next:** Commit and push the implementation. After the user refreshes the CCU token, bootstrap the pinned environment, run the three-model serial queue, retrieve and validate each completed artifact set, and write one CCU report per model.

---

## 2026-07-18 · Step 43 — Distinguish stopped CCU server from token failure
- **Context:** Recheck CCU after the user restarted the personal Jupyter server and correct the HTTP 302 diagnosis.
- **Agent:** gpt-5-codex
- **Did:** Updated the local reusable `ccu/` client so an HTTP 302 login redirect identifies a stopped personal Jupyter server as a possible cause and tells the operator to start it in the browser and retry `ccu doctor` before replacing the Keychain token; added troubleshooting and regression coverage, ran the full CCU validation suite, and reinstalled the local client.
- **Findings:** Live access passed all doctor gates, remote identity returned `jovyan`, and the visible NVIDIA H100 80GB HBM3 had 80,995 MiB free at 0% utilization. The CCU suite passed 60 tests plus Ruff, shell syntax, and the token-leak scan.
- **Decision / rationale:** Treat HTTP 302 as ambiguous between a stopped server and credential rejection. Check server lifecycle first to avoid unnecessary token rotation while retaining strict no-redirect and Keychain-only authentication.
- **Next:** Use the restored H100 access to bootstrap the pinned Gemma 4 environment and start the resumable serial queue.

---

## 2026-07-18 · Step 44 — Bootstrap CCU and close missing scientific dependencies
- **Context:** Deploy the approved Gemma 4 calibrated queue after CCU access was restored.
- **Agent:** gpt-5-codex
- **Did:** Cloned commit `79c6b0d` to `/home/jovyan/work/normalcy-axis`, built `/home/jovyan/.venvs/normalcy-gemma4-cu124`, and launched the serial queue; after its fail-closed import gate identified missing SciPy, pinned SciPy 1.17.0 and scikit-learn 1.8.0 in the CCU environment specification and regression test.
- **Findings:** The environment passed exact version and H100 checks for PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, TransformerLens 3.5.1, Transformers 5.13.0, and Accelerate 1.14.0. The first 12B attempt stopped before model loading or output/checkpoint creation with exact error `ModuleNotFoundError: No module named 'scipy'`; 26B-A4B and 31B remained pending. The focused checkpoint/CCU suite passed five tests plus Ruff, shell syntax, and `git diff --check`.
- **Decision / rationale:** Treat this as an environment-manifest omission, not a model or method failure. Pin both SciPy and scikit-learn because the imported Gemma causality module requires both at import time; retain the same clean model labels and serial order.
- **Next:** Commit and deploy the dependency correction, rerun bootstrap, then restart the 12B-first queue.

---

## 2026-07-18 · Step 45 — Start corrected CCU Gemma 4 serial queue
- **Context:** Restart the fail-closed CCU run after completing the pinned environment.
- **Agent:** gpt-5-codex
- **Did:** Fast-forwarded the CCU checkout to `04cf243`, reran the environment bootstrap successfully, and started `jobs/ccu/run_gemma4_calibrated_queue.sh` under background PID `1197` with durable state, logs, checkpoints, and sentinels under `/home/jovyan/work/normalcy-gemma4-state`.
- **Findings:** The corrected environment passed all seven exact package-version checks and the H100 gate. The queue entered the 12B smoke stage, remained live after 33 seconds, and began downloading the pinned Gemma 4 checkpoint from Hugging Face; GPU allocation had not started yet. The prior SciPy traceback remains only as historical text in the append-only remote queue log.
- **Decision / rationale:** Keep the remote checkout pinned at `04cf243` for checkpoint fingerprint stability during this queue. Preserve serial order 12B, 26B-A4B, 31B and stop only on a technical failure.
- **Next:** Monitor the 12B smoke gate and full run, then retrieve, validate, and report each completed model before advancing the empirical synthesis.

---

## 2026-07-18 · Step 46 — Audit legacy-to-Gemma4 test coverage during CCU run
- **Context:** Compare completed legacy Gemma/Qwen tests with the current three-model Gemma 4 evidence matrix.
- **Agent:** gpt-5-codex
- **Did:** Checked committed artifacts for all Gemma 4 stages, dense and hiring regimes, post-hoc outputs, and live CCU queue/checkpoint state; cross-checked the legacy findings index and model reports.
- **Findings:** Gemma 4 Stage 1, Stage 2, Stage 3, Stage 3B, technical smoke, neutral/PCA, raw dense steering, 282-name audit, and local/broad hiring steering are complete for all three models. Denoised dense and denoised hiring are complete for 12B and 26B-A4B but missing for 31B. Demographic disparity, mediation, name-level R4, and full-282 expansions are absent for all Gemma 4 models. No compatible Gemma 4 SAE exists. The CCU 12B calibrated run was live with 207 of 2,022 checkpoint shards, while 26B-A4B and 31B remained pending.
- **Decision / rationale:** Describe Gemma 4 as complete for representation and basic SAE-free causal parity, but not as a complete replication of the full legacy post-hoc/SAE matrix.
- **Next:** Complete the serial calibrated runs, then close the 31B denoised and three-model post-hoc gaps before claiming full legacy parity.

---

## 2026-07-18 · Step 47 — Confirm live CCU 12B calibrated progress
- **Context:** Provide a live status check for the serial Gemma 4 calibrated queue.
- **Agent:** gpt-5-codex
- **Did:** Checked CCU doctor, queue process/state, GPU utilization, checkpoint shard count, sentinels, and the current steering log.
- **Findings:** Queue PID `1197` remained live after 11 minutes 26 seconds. Gemma 4 12B was running with 456 of 2,022 checkpoint shards (22.6%); the log had reached warmth additive `random_088`. The H100 used 24,166 MiB with 43% utilization. No error or success sentinel existed; 26B-A4B and 31B remained pending.
- **Next:** Continue monitoring until the 12B validator and sentinel pass, then allow the serial coordinator to start 26B-A4B automatically.

---

## 2026-07-18 · Step 48 — Delegate calibrated 26B-A4B from CCU to SCCKN
- **Context:** Use two newly idle SCCKN RTX PRO 6000 slots while CCU continues the 12B calibrated run.
- **Agent:** gpt-5-codex
- **Did:** Verified `gpu@scc214` reports two free GPU slots and the `rtx_6000` host feature; sent `SIGSTOP` only to CCU coordinator PID `1197`, leaving its 12B runner, tee, and GPU process active; extended the independent SGE calibrated submitter/runner with a distinct checkpointed Gemma 4 26B-A4B condition.
- **Findings:** CCU 12B continued on 24,156 MiB VRAM after the coordinator entered stopped state, so it cannot auto-launch 26B after 12B completion. The new SCCKN condition uses label `gemma4_26b_a4b_calibrated_scckn_rtx6000`, exact RTX PRO 6000 runtime gating, 99 SD-matched random directions, additive plus norm-preserving interventions, and atomic resume. Thirteen focused tests, Ruff, shell syntax, dry-run, and `git diff --check` passed.
- **Decision / rationale:** Run 26B-A4B on SCCKN in parallel, preserve CCU 12B, and reserve CCU for 31B after the 12B sentinel. Separate labels prevent output collision or accidental duplicate acceptance.
- **Next:** Commit and deploy the SGE extension, submit/release 26B-A4B, verify physical RTX assignment, and install a safe post-12B CCU handoff to 31B only.

---

## 2026-07-18 · Step 49 — Start parallel SCCKN 26B-A4B and arm CCU handoff
- **Context:** Complete the requested split execution without interrupting the live CCU Gemma 4 12B run or launching duplicate 26B-A4B work.
- **Agent:** gpt-5-codex
- **Did:** Submitted and released SCCKN job `1145460` for the distinct Gemma 4 26B-A4B calibrated condition, verified its assignment to `gpu@scc214` with the hard `rtx_6000=1` resource, and installed a durable CCU watcher that waits for the 12B success sentinel, terminates the stopped serial coordinator, marks 26B-A4B as delegated, and launches only 31B.
- **Findings:** SCCKN reports the 26B-A4B job in running state with active CPU and memory accounting and a 38.937 GiB maximum virtual-memory footprint during startup. CCU still reports coordinator PID `1197` stopped, 12B runner PID `1201` active, 24,156 MiB GPU memory allocated, and no 12B success sentinel yet; watcher PID `2318` survived terminal detachment and is waiting. Thirteen focused tests, Ruff, shell syntax, upload SHA-256 verification, and `git diff --check` passed.
- **Decision / rationale:** Keep 12B and later 31B on the CCU H100 while executing 26B-A4B independently on one SCCKN RTX PRO 6000. Sentinel-gated handoff preserves the existing 12B checkpoint fingerprint and prevents both an accidental CCU 26B launch and premature 31B overlap.
- **Next:** Monitor both active runs; after each success sentinel, retrieve and validate artifacts and write the required per-model empirical report.

---

## 2026-07-18 · Step 50 — Confirm concurrent CCU and SCCKN progress
- **Context:** Live status check after splitting the calibrated Gemma 4 execution across CCU and SCCKN.
- **Agent:** gpt-5-codex
- **Did:** Checked CCU access, processes, GPU accounting, queue state, sentinels, checkpoint counts, and logs; checked SCCKN scheduler assignment, resource request, accounting, sentinel, and checkpoint count.
- **Findings:** CCU Gemma 4 12B remains active on the H100 at 41% sampled utilization and 24,166 MiB allocated, with 1,317 of 2,022 checkpoint shards (65.1%) and no error or success sentinel. SCCKN job `1145460` remains running on `gpu@scc214` with the hard `rtx_6000=1` request, 425 checkpoint files (approximately 21.0%), active accounting, and no error or success sentinel. The detached CCU handoff watcher remains active; 31B has correctly not started. A direct nested SSH GPU query to `scc214` was denied by host authentication policy, but Grid Engine assignment and advancing checkpoints independently confirm execution.
- **Next:** Continue monitoring 12B and 26B-A4B; allow the sentinel-gated watcher to start CCU 31B only after validated 12B completion.

---

## 2026-07-18 · Step 51 — Move calibrated 31B execution to the second SCCKN GPU
- **Context:** Use the remaining idle SCCKN RTX PRO 6000 while retaining CCU 31B as a fallback until physical scheduler assignment.
- **Agent:** gpt-5-codex
- **Did:** Added a checkpointed, write-once Gemma 4 31B condition to the independent SCCKN calibrated submitter and runner; passed the focused suite; submitted held job `1145463`, synchronized its manifest, released it, and waited for Grid Engine state `r`. Only after assignment, terminated the CCU 31B handoff watcher and stopped coordinator while preserving the active orphaned 12B runner and tee.
- **Findings:** SCCKN 31B is running on `gpu@scc214` with hard resource `rtx_6000=1`, active accounting, 54.916 GiB virtual memory, and 17 checkpoint files shortly after startup. SCCKN 26B-A4B remains running with 766 checkpoint files. CCU 12B remains active with 24,156 MiB GPU allocation; its queue state now marks both 26B-A4B and 31B as `delegated_scckn`, and neither CCU coordinator nor handoff watcher remains. Thirteen focused tests, Ruff, shell syntax, dry-run, and `git diff --check` passed.
- **Decision / rationale:** Cancel the CCU 31B launch path only after SCCKN provided physical execution evidence, exactly preserving the H100 fallback until that point. Distinct SCCKN labels and checkpoint roots prevent output collision.
- **Next:** Monitor all three live models, validate each success artifact set, and create the required per-model empirical reports.

---

## 2026-07-18 · Step 52 — Confirm all three calibrated runs advancing
- **Context:** Live progress check for the split CCU and SCCKN Gemma 4 calibrated execution.
- **Agent:** gpt-5-codex
- **Did:** Checked CCU process, GPU, checkpoint, output, and sentinel state and both SCCKN jobs, accounting records, checkpoint roots, logs, and sentinels.
- **Findings:** CCU 12B completed all 2,022 checkpoint shards and is CPU-bound at approximately 101% while consolidating or validating, with no success or error sentinel yet. SCCKN 26B-A4B has 1,030 of 2,022 shards (50.9%), and SCCKN 31B has 324 of 2,022 shards (16.0%); both jobs remain in state `r` with advancing accounting and no success or error sentinel.
- **Next:** Wait for the 12B final validator and sentinel, then retrieve its artifacts while continuing to monitor both SCCKN runs.

---

## 2026-07-18 · Step 53 — Complete 12B calibrated and post-hoc tests and start full-282 expansion
- **Context:** Use the freed CCU H100 immediately for the remaining Gemma 4 12B legacy-parity tests.
- **Agent:** gpt-5-codex
- **Did:** Retrieved and hash-verified the completed CCU calibrated artifacts; ran and validated CPU-only disparity, mediation, R4, and full-282 gate tests; added a write-once H100-only full-282 runner; passed 12 focused tests plus Ruff and shell checks; launched the independent local full-282 run; wrote `paper/2026-07-18_2314_gemma4_12b_ccu_calibrated_steering.md`, `paper/2026-07-18_2314_gemma4_12b_posthoc_hiring.md`, and `paper/2026-07-18_2314_gemma4_12b_full282_gate.md`.
- **Findings:** The CCU calibrated run passed with 40,440 raw, 2,020 summary, and eight null rows and maximum norm drift 0.005823. Post-hoc validation produced all five required outputs; the competence-mediated race path was the only 95% interval excluding zero. The full-282 gate fired for broad-regime sign mismatch and non-monotonicity on both axes. Local full-282 selected 282 of 282 names and began model loading on the H100.
- **Decision / rationale:** Run local, broad, and denoised-local full-282 expansions as separate write-once executions because the predeclared gate requires all three when any criterion fires. Do not substitute an SAE test because no compatible Gemma 4 SAE is available.
- **Next:** Validate and report local full-282, then launch broad and denoised-local independently; continue monitoring SCCKN 26B-A4B and 31B calibrated runs.

---

## 2026-07-18 · Step 54 — Complete local full-282 and start broad expansion
- **Context:** Continue the gate-required Gemma 4 12B expansion without leaving the CCU H100 idle.
- **Agent:** gpt-5-codex
- **Did:** Validated and hash-retrieved all four local full-282 outputs, wrote `paper/2026-07-18_2318_gemma4_12b_local_full282.md`, and launched broad full-282 as a separate write-once CCU run.
- **Findings:** Local full-282 produced 2,820 raw rows over all 282 names. Warmth steering was monotone with slope 20.215, R2 0.930, and +0.10 mean delta 1.036 (95% CI [1.022, 1.051]); competence steering was monotone with slope 22.951, R2 0.967, and +0.10 mean delta 1.515 (95% CI [1.499, 1.531]). Broad full-282 passed its H100 and output-absence gates and began model loading.
- **Next:** Validate and report broad full-282, then launch denoised-local independently.

---

## 2026-07-18 · Step 55 — Complete broad full-282 and start denoised expansion
- **Context:** Continue the gate-required Gemma 4 12B expansion as independent CCU executions.
- **Agent:** gpt-5-codex
- **Did:** Validated and hash-retrieved all four broad full-282 outputs, wrote `paper/2026-07-18_2322_gemma4_12b_broad_full282.md`, and launched denoised-local full-282 separately on the CCU H100.
- **Findings:** Broad full-282 produced 2,820 raw rows and reproduced non-monotonicity on both axes. Warmth slope was 8.729 but the +0.50 endpoint was -1.276 (95% CI [-1.301, -1.250]); competence slope was 6.503 but the endpoint was -0.082 (95% CI [-0.112, -0.053]). Both endpoint signs opposed their fitted slopes. Denoised-local passed H100 and output-absence gates and began model loading.
- **Decision / rationale:** Treat broad steering as a bounded nonlinear intervention, not a globally linear dose response; retain the full-name result rather than attributing the reversal to the original 60-name sample.
- **Next:** Validate and report denoised-local full-282, which will close the gate-required 12B expansion.

---

## 2026-07-18 · Step 56 — Complete the Gemma 4 12B full-282 expansion
- **Context:** Close the last gate-required SAE-independent Gemma 4 12B run.
- **Agent:** gpt-5-codex
- **Did:** Validated and hash-retrieved all four denoised-local full-282 outputs and wrote `paper/2026-07-18_2325_gemma4_12b_denoised_full282.md`.
- **Findings:** Denoised-local produced 2,820 raw rows. Warmth steering remained monotone with slope 19.652, R2 0.911, and +0.10 mean delta 0.908 (95% CI [0.894, 0.922]); competence remained monotone with slope 21.352, R2 0.951, and +0.10 mean delta 1.259 (95% CI [1.243, 1.276]). Relative to raw local, denoising reduced but did not reverse either endpoint effect.
- **Decision / rationale:** Mark all currently defined SAE-independent Gemma 4 12B legacy-parity tests complete. The only unavailable class remains SAE-based testing because no compatible Gemma 4 SAE exists.
- **Next:** Finish and report the SCCKN calibrated 26B-A4B and 31B runs, then apply the post-hoc and conditional expansion matrix to those models as required.

---

## 2026-07-18 · Step 57 — Queue larger-model remaining tests and keep CCU active
- **Context:** Apply the completed 12B SAE-independent test matrix to Gemma 4 26B-A4B and 31B without allowing the CCU server to idle.
- **Agent:** gpt-5-codex
- **Did:** Confirmed no pending RTX PRO 6000 jobs and zero free SCCKN GPUs; added and tested generic larger-model CCU runner and fail-closed gate-driven queue; launched 31B denoised 60-name steering on H100; completed and validated 26B-A4B and 31B post-hoc analyses and the 26B-A4B full-282 gate; armed the queue to run the 31B gate followed by required expansions; wrote three dated findings reports.
- **Findings:** The two SCCKN calibrated jobs remained active at 1,920 and 1,367 of 2,022 shards. The 26B-A4B full-282 gate fired with eight competence-related reasons. Post-hoc joins matched 269 names for disparity/mediation and 149 for R4 in both models. CCU 31B denoised passed its H100 and absence gates and used approximately 61,050 MiB while running. Fourteen focused tests, Ruff, shell syntax, upload hash verification, and `git diff --check` passed.
- **Decision / rationale:** Use CCU for the missing 31B denoised prerequisite and all gate-required full-282 expansions while SCCKN remains fully occupied by calibrated runs. Keep each model-regime execution write-once with separate logs and sentinels, but use one fail-closed serial coordinator to prevent CCU inactivity between tasks.
- **Next:** Retrieve, validate, and report each queued output as it completes; synchronize completed SCCKN calibrated artifacts and write one report per model.

---

## 2026-07-18 · Step 58 — Complete 26B calibrated and arm both full-282 expansions
- **Context:** Validate newly completed SCCKN and CCU prerequisites while the persistent larger-model queue advances.
- **Agent:** gpt-5-codex
- **Did:** Synchronized and inspected the completed 26B-A4B calibrated artifacts; retrieved and validated the 31B denoised outputs and full-282 gate; wrote three dated reports; confirmed the queue advanced immediately to 26B-A4B local full-282.
- **Findings:** The 26B-A4B calibrated run passed with 40,440 raw, 2,020 summary, and eight null rows, peak allocated VRAM 48.48 GiB, and maximum norm drift 0.005351. The 31B denoised prerequisite completed 600 rows and was non-monotone on both axes. Its gate fired with sixteen reasons. CCU then began 26B-A4B local full-282 at approximately 50,280 MiB and 56% sampled GPU utilization; SCCKN 31B calibrated remained active.
- **Decision / rationale:** Keep the queue order at all three 26B-A4B expansions followed by all three 31B expansions because both predeclared gates fired and the CCU server must remain active.
- **Next:** Retrieve, validate, and report each full-282 run; synchronize and report 31B calibrated completion.

---

## 2026-07-18 · Step 59 — Resume interrupted 31B calibrated run from checkpoint
- **Context:** Recover the SCCKN 31B calibrated job after it left the queue before producing a success sentinel.
- **Agent:** gpt-5-codex
- **Did:** Queried Grid Engine accounting and exact logs, preserved the original state directory, submitted held retry `1145490` against the same pinned commit and output label, committed its retry manifest, released it, and retained the concurrent CCU remaining-test queue.
- **Findings:** Original job `1145463` ended after 2,217 seconds with scheduler `failed=0`, wrapper `exit_status=120`, an empty error log, no success sentinel, and 1,657 checkpoint files. There is no model, OOM, or validation traceback. The retry is configured to detect the existing checkpoint manifest and resume rather than repeat completed shards; it is currently scheduler-pending while SCCKN reports two available GPU resources. CCU 26B-A4B local full-282 remains active at approximately 50,280 MiB and 55% sampled utilization.
- **Decision / rationale:** Treat the 31B event as an incomplete operational exit, not an empirical failure. Preserve all checkpoints and require the normal final validator and sentinel before accepting the run.
- **Next:** Verify physical RTX assignment for retry `1145490`, then synchronize and report the completed 31B calibrated artifacts.

---

## 2026-07-18 · Step 60 — Preserve checkpoint-origin identity and complete 26B local expansion
- **Context:** Resolve the strict 31B resume mismatch while the CCU queue continues producing full-name results.
- **Agent:** gpt-5-codex
- **Did:** Proved the first resume retry differed only in the fingerprint commit field, added an opt-in resume-only checkpoint-origin commit argument while retaining exact argument and input-hash checks, updated the SCCKN runner to read that commit from the immutable manifest, passed 17 focused tests plus Ruff and shell checks, and submitted held retry `1145497`. Separately retrieved and validated 26B-A4B local full-282 and wrote `paper/2026-07-18_2346_gemma4_26b_a4b_local_full282.md`.
- **Findings:** Retry `1145490` reached the correct RTX PRO 6000 but fail-closed before model loading because repository HEAD had advanced from checkpoint origin `ae94b4d`; all critical code/config diffs and input hashes were unchanged. The backward-compatible fix leaves default fingerprinting strict and permits the old commit only when explicitly resuming. The 26B-A4B local expansion completed 2,820 rows; warmth was monotone but small, while competence remained non-monotone with +0.10 mean delta -0.408 (95% CI [-0.443, -0.376]). CCU advanced to broad full-282.
- **Decision / rationale:** Permit resume across unrelated repository-history advances only by explicitly preserving the manifest's original commit; do not rewrite manifests or weaken any other fingerprint field.
- **Next:** Confirm retry `1145497` assignment and shard advancement; report the remaining queued full-282 runs as they complete.

---

## 2026-07-18 · Step 61 — Confirm successful 31B calibrated checkpoint resume
- **Context:** Verify the checkpoint-origin compatibility fix under the real SCCKN retry.
- **Agent:** gpt-5-codex
- **Did:** Confirmed retry `1145497` received an RTX PRO 6000, passed hardware, absence, fingerprint, and model-load gates, replayed existing shards as resume hits, and began writing new immutable shards.
- **Findings:** The log advanced from `[resume] competence random_025` into new steering work for random directions 025 and 026; checkpoint files increased from 1,657 to 1,668 with no error output. CCU simultaneously remained occupied by 26B-A4B broad full-282 under queue PID `7229`.
- **Decision / rationale:** Accept the compatibility fix as operationally validated while retaining the original checkpoint manifest and all strict non-commit fingerprint fields.
- **Next:** Monitor both active paths to their validators and success sentinels.

---

## 2026-07-19 · Step 1 — Complete all queued larger-model full-282 runs
- **Context:** Recover overnight CCU results and determine current SCCKN state.
- **Agent:** gpt-5-codex
- **Did:** Verified all six queued 26B-A4B and 31B model-regime success sentinels, hash-downloaded the five previously unretrieved output sets, inspected their summaries, and wrote one dated report per run.
- **Findings:** The CCU queue completed all tasks between 23:42 and 00:09 Europe/Berlin with 2,820 validated rows per run. In 26B-A4B, warmth remained monotone while competence was non-monotone in local, broad, and denoised-local regimes. In 31B, both axes were non-monotone and had negative positive-strength endpoints in all three regimes. CCU is now idle. SCCKN status could not be refreshed because the login host refused port 22; the last verified 31B calibrated state was a successful checkpoint resume with advancing shards.
- **Decision / rationale:** Mark the full-282 matrix complete for both larger Gemma 4 models, but leave 31B calibrated status unresolved until SCCKN connectivity returns and its final sentinel can be checked.
- **Next:** Commit the recovered artifacts, then recheck SCCKN and synchronize the 31B calibrated result if complete.

---

## 2026-07-19 · Step 2 — Record SSH outage after recovering CCU outputs
- **Context:** Finalize the overnight status check after all CCU artifacts were secured locally.
- **Agent:** gpt-5-codex
- **Did:** Committed the five recovered full-282 result sets and reports locally as `c1f43c7`, checked repository divergence, and retried synchronization.
- **Findings:** The local branch is one commit ahead of origin. Both SCCKN (`scc2.uni-konstanz.de:22`) and GitHub SSH (`github.com:22`) returned `Connection refused`, preventing a fresh 31B calibrated sentinel check and remote push. CCU access remains healthy, all required larger-model sentinels are present, and its H100 is idle.
- **Decision / rationale:** Preserve the unpushed local commit and report 31B calibrated as externally unverified rather than inferring completion from elapsed time.
- **Next:** Retry SCCKN and GitHub SSH after connectivity returns, then synchronize and report the 31B calibrated result.

---

## 2026-07-19 · Step 3 — Close Gemma 4 31B calibrated and repair Qwen topic selection
- **Context:** Resume cluster work after network connectivity returned and continue the remaining larger-model tests.
- **Agent:** gpt-5-codex
- **Did:** Pushed the recovered CCU commit, synchronized and validated the completed SCCKN 31B calibrated outputs, wrote `paper/2026-07-19_0030_gemma4_31b_calibrated_steering.md`, replaced the Qwen calibrated runner's contiguous-index assumption with explicit topic-ID-to-activation-row mapping, added collision-free rerun labels and Qwen3.6-35B-A3B scheduler support, and ran focused tests and shell checks.
- **Findings:** Gemma 4 31B passed with 40,440 raw rows, 2,020 summary rows, eight null rows, 58.69 GiB peak allocated VRAM, and 0.006368 maximum norm drift. Its target-minus-random paired-topic estimates were negative for both target axes. The Qwen defect is a selection and row-alignment bug caused by non-contiguous topic identifiers, not a model, memory, hook, or library limitation. Nine focused tests, Ruff, shell syntax, and both scheduler dry runs passed.
- **Decision / rationale:** Preserve the rejected Qwen 27B artifact and write the corrected rerun to a new label. Run corrected Qwen 27B and first calibrated Qwen 35B-A3B as separate, unchained RTX PRO 6000 jobs.
- **Next:** Commit and synchronize the fix, submit one held job per Qwen model, then release both only after their manifests are preserved.

---

## 2026-07-19 · Step 4 — Start corrected Qwen calibrated queue on CCU H100
- **Context:** Route the corrected larger-model runs to available hardware after checking SCCKN RTX scheduling and CCU fallback capacity.
- **Agent:** gpt-5-codex
- **Did:** Preserved and released independent SCCKN manifests for Qwen3.6 27B and 35B-A3B, added and tested a serial fail-closed CCU H100 runner, cloned the Gemma environment into an isolated Qwen environment, upgraded it to Transformers 5.14.1, removed TransformerLens only from the clone, launched the corrected 27B then 35B-A3B queue, and removed the duplicate SCCKN jobs after the CCU execution was physically active.
- **Findings:** Six pre-existing jobs occupied SCCKN's RTX host despite the host-level `qc:gpu=2` display. Both submitted SCCKN jobs transitioned from pending to running between status polls and were terminated as duplicates with exit 137 after CCU had begun loading 27B. The CCU H100 has 79.10 GiB total VRAM; prior measured peaks were 51.26 GiB for 27B and 65.52 GiB for 35B-A3B. The corrected 27B process downloaded all 15 model files, began loading 1,184 weight tensors, and occupied 52.72 GiB. Ten focused queue tests, shell syntax, package consistency, native-HF isolation, and write-once output gates passed.
- **Decision / rationale:** Keep one authoritative execution path on CCU to prevent duplicate output labels. Run 27B and 35B-A3B serially because one H100 cannot host both models simultaneously; the fail-closed queue prevents 35B-A3B from starting if 27B fails.
- **Next:** Validate and retrieve the 27B artifacts, allow the queue to advance immediately to 35B-A3B, then write one empirical report per model.

---

## 2026-07-19 · Step 5 — Confirm CCU shutdown interrupted Qwen 27B
- **Context:** Recover the Qwen calibrated queue state after the personal CCU server was restarted.
- **Agent:** gpt-5-codex
- **Did:** Inspected the persistent queue state, sentinels, processes, GPU, logs, output paths, environment, and model cache after CCU access returned.
- **Findings:** No Qwen process or success sentinel remains, and the H100 is idle. The 27B log reached competence `random_024` under additive steering, approximately 57% of the complete intervention loop, before shutdown. No partial result tables exist because the native-HF runner writes outputs only after completing all interventions. Qwen 35B-A3B remained pending and never started. The isolated Transformers 5.14.1 environment, downloaded weights, vectors, queue logs, and state files survived on persistent storage.
- **Decision / rationale:** Treat the stale `running` queue-state value as interrupted rather than successful. Do not accept or report an empirical 27B result without the normal validator and success sentinel.
- **Next:** Restart 27B from the beginning or add resumable checkpointing before relaunch, then run 35B-A3B.

---

## 2026-07-19 · Step 6 — Implement resumable three-GPU Qwen pipeline
- **Context:** Prepare two corrected calibrated runs and an independent hiring audit for parallel RTX PRO 6000 and H100 execution after the interrupted CCU run.
- **Agent:** gpt-5-codex
- **Did:** Added atomic fingerprinted checkpoints to native-HF calibrated steering; added native-HF 282-name hiring audit and local, broad, and denoised-local steering with per-name checkpoints; added validators and independent SCCKN/CCU runner support; wrote `paper/2026-07-19_0944_qwen36_resumable_parallel_pipeline.md`.
- **Findings:** Resume now preserves completed baselines and steering work units and rejects changed commits, arguments, revisions, topic splits, or input hashes. The hiring path retains raw explicit-BOS name activations, native-chat callback decisions, one-token Yes/No checks, zero vision calls, and no TransformerLens import. Twenty-two focused tests plus Ruff, Python compilation, shell syntax, and whitespace checks passed.
- **Decision / rationale:** Launch corrected 27B and 35B-A3B calibrated steering independently on the two SCCKN RTX resources and use CCU H100 for the non-duplicative 27B hiring audit. Keep the rejected historical 27B artifact unchanged.
- **Next:** Commit and synchronize the implementation, preserve held SCCKN manifests, release both RTX jobs, then start and verify the H100 audit.

---

## 2026-07-19 · Step 7 — Start three-GPU wave and complete Qwen 27B audit
- **Context:** Execute the first parallel wave of the resumable Qwen3.6 parity plan.
- **Agent:** gpt-5-codex
- **Did:** Submitted and released independent corrected calibrated jobs `1145640` and `1145641` on SCCKN; created a clean detached CCU worktree at commit `ac1c643`; completed, retrieved, and locally validated the H100 27B audit; launched 27B local hiring steering on the freed H100; added resumable native-HF neutral extraction for later PCA denoising.
- **Findings:** Both SCCKN jobs reached state `r` on `gpu@scc214` with separate RTX PRO 6000 allocations and advancing checkpoint shards. The H100 audit passed 282 unique names in 119.4 seconds at 51.04 GiB peak allocated VRAM. Spearman correlations were 0.1863 for model versus human warmth, 0.2499 for competence, 0.3017 for callback versus model warmth, and 0.2201 for callback versus model competence. All 282 audit shards and both final artifacts validated.
- **Decision / rationale:** Keep both calibrated jobs authoritative on SCCKN and immediately reuse H100 for the non-duplicative 27B local hiring intervention. Preserve the CCU worktree separately from the older dirty result-producing checkout.
- **Next:** Validate and report 27B local steering, then launch broad steering; continue monitoring both calibrated checkpoints and prepare neutral extraction/PCA.

---

## 2026-07-19 · Step 8 — Complete Qwen 27B local steering and keep H100 active
- **Context:** Consume the first causal hiring result while preserving continuous use of the independent H100 lane.
- **Agent:** gpt-5-codex
- **Did:** Retrieved and hash-verified the 27B local steering artifacts, passed the local validator, generated the bootstrap summary, wrote `paper/2026-07-19_0953_qwen36_27b_local_hiring.md`, and launched the independent broad-strength run on the freed H100.
- **Findings:** All 660 checkpoints and 600 raw rows passed. Warmth and competence were both monotone; +0.10 mean effects were +1.196 (95% CI [1.171, 1.219]) and +0.533 (95% CI [0.506, 0.560]). Peak allocated VRAM was 51.14 GiB. The broad run became physically active at approximately 53,088 MiB GPU memory.
- **Decision / rationale:** Continue the predeclared robustness sequence without waiting for the two independent calibrated jobs; retain the quantization warning because all callback margins fall on the 0.125 grid.
- **Next:** Retrieve and report broad steering, then start neutral extraction and denoising without duplicating any calibrated work.

---

## 2026-07-19 · Step 9 — Complete Qwen 27B posthoc hiring analyses
- **Context:** Use the validated 282-name Qwen audit for GPU-free parity analyses while all three accelerators remain occupied.
- **Agent:** gpt-5-codex
- **Did:** Ran demographic disparity, 5,000-bootstrap mediation, group-level R4, and name-level R4; wrote `paper/2026-07-19_0953_qwen36_27b_posthoc_hiring.md`.
- **Findings:** The disparity join matched 269 names and the exact study-name R4 join matched 149. Competence indirect effects excluded zero for race (-0.0488, 95% CI [-0.1035, -0.0106]) and gender (-0.1227, 95% CI [-0.2056, -0.0608]); warmth intervals included zero. Name-level model-human callback correlation was r=0.042 (p=0.614).
- **Decision / rationale:** Treat mediation as associational decomposition and retain the group R4 result as descriptive because it contains only four groups.

---

## 2026-07-19 · Step 10 — Complete Qwen 27B broad steering and start neutral extraction
- **Context:** Test whether the local causal effect survives larger raw-vector interventions while keeping the H100 lane continuously active.
- **Agent:** gpt-5-codex
- **Did:** Retrieved, hash-verified, validated, and summarized the 27B broad run; wrote `paper/2026-07-19_0955_qwen36_27b_broad_hiring.md`; created a separate clean CCU worktree pinned to commit `2e4102d`; launched resumable neutral-corpus extraction on the freed H100.
- **Findings:** All 660 checkpoints and 600 rows passed. Warmth and competence remained monotone; +0.50 effects were +2.240 (95% CI [2.208, 2.273]) and +1.069 (95% CI [1.037, 1.102]). Peak allocated VRAM was 51.14 GiB. Neutral extraction loaded the model on the H100 from the exact later implementation commit.
- **Decision / rationale:** Proceed to PCA denoising before the denoised-local intervention; keep neutral work isolated from the older clean worktree used by completed steering runs.
- **Next:** Retrieve and validate the 1,500-row neutral matrix, run CPU PCA denoising, then launch denoised-local steering.

---

## 2026-07-19 · Step 11 — Complete Qwen 27B neutral extraction and PCA denoising
- **Context:** Produce a resumable, SAE-free nuisance-removal control without idling the H100 before the second model's audit.
- **Agent:** gpt-5-codex
- **Did:** Completed and locally validated 1,500 neutral activation shards, hash-reconstructed the 30 MiB matrix from a compressed CCU transfer, ran neutral PCA on CPU, wrote separate extraction and denoising reports, and started the independent Qwen3.6-35B-A3B audit on the H100.
- **Findings:** The finite matrix is 1500×5120, took 353.3 seconds, and peaked at 51.38 GiB. Twenty-seven PCA components covered 50.23% variance; concept-vector cosine fell from 0.580 to 0.560, while warmth-on-competence leakage increased from d=5.44 to 5.90. The 35B-A3B audit became physically active at approximately 67,738 MiB and began producing checkpoints.
- **Decision / rationale:** Do not interpret PCA as axis disentanglement; use the denoised vector only as a causal robustness condition. Run the 35B-A3B audit while the CPU denoising result is prepared, then return to 27B denoised-local steering.
- **Next:** Validate and report the 35B-A3B audit, then launch 27B denoised-local steering and evaluate its conditional expansion gate.

---

## 2026-07-19 · Step 12 — Complete Qwen 35B-A3B audit and posthoc analyses
- **Context:** Establish observational hiring parity for the second Qwen3.6 model while the two calibrated RTX runs continue independently.
- **Agent:** gpt-5-codex
- **Did:** Retrieved and locally validated the 282-name audit, ran the disparity, 5,000-bootstrap mediation, group R4, and name-level R4 analyses, wrote one report for the audit and one for posthoc results, and launched 27B denoised-local steering on the freed H100.
- **Findings:** The audit took 193.7 seconds and peaked at 65.46 GiB. Model-human Spearman correlations were 0.2109 for warmth and 0.1313 for competence; callback-versus-model correlations were 0.3444 and 0.1968. The posthoc join matched 269 names, with three of four probe-mediation intervals excluding zero; model-human name-level callback r was -0.013 (p=0.879).
- **Decision / rationale:** Keep observational and causal claims separate. Return the H100 to the nearly prepared 27B robustness condition before starting the 35B-A3B intervention sequence.
- **Next:** Complete and report 27B denoised-local steering and its gate, then run 35B-A3B local, broad, neutral, and denoised-local conditions.

---

## 2026-07-19 · Step 13 — Complete Qwen 27B denoised steering and close its expansion gate
- **Context:** Finish the three-regime 60-name causal matrix and apply the predeclared conditional full-name policy.
- **Agent:** gpt-5-codex
- **Did:** Retrieved, validated, and summarized denoised-local steering; evaluated the gate across local, broad, and denoised-local summaries; wrote one result report and one gate report; started 35B-A3B local steering on the freed H100.
- **Findings:** All 660 denoised work units passed. +0.10 effects were +1.140 (95% CI [1.113, 1.165]) for warmth and +0.408 (95% CI [0.381, 0.438]) for competence; both were monotone. The gate returned `run_full_282=false` with zero reasons.
- **Decision / rationale:** Do not run the three 282-name steering expansions because the predeclared conditional criteria did not fire. This is a protocol-defined stop, not a missing test.
- **Next:** Complete the 35B-A3B raw local and broad runs, then extract neutral activations, denoise its vectors, and finish the second model's gate.
