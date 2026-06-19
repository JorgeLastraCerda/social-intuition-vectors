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
