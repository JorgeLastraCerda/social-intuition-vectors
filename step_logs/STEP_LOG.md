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
