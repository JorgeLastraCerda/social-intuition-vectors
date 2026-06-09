# Step Log — Warmth & Competence Probing

> Append-only research log. Every meaningful step/finding gets one entry, newest at the bottom.
> Do not edit or delete past entries. English only. See the "Step Logging" rule in CLAUDE.md.
>
> Entry format:
> ```
> ## YYYY-MM-DD · Step N — <short title>
> - **Context:** which task/session this belongs to (1 sentence)
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
