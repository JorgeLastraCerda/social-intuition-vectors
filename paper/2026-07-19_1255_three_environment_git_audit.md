# Three-Environment Git and Artifact Audit

- **Produced:** 2026-07-19 12:55 Europe/Berlin
- **Model:** Gemma and Qwen research workspace
- **Scope:** Local, SCCKN, CCU, and GitHub branch, artifact, and cache reconciliation
- **Status:** Complete; canonical active checkouts synchronized without model-weight publication

## Artifacts

- **Inputs:** `.gitignore`, `data/processed/`, `results/`, `paper/README.md`, `step_logs/STEP_LOG.md`
- **Outputs:** `paper/2026-07-19_1255_three_environment_git_audit.md`, `step_logs/STEP_LOG.md`

## Initial State

Local and GitHub `main` were already equal at `9d113ab`. The SCCKN active checkout was at ancestor `b138eb6` with 12 untracked calibrated Qwen artifacts. The CCU active checkout was at ancestor `c9eced7`, and its active and detached worktrees contained 108 dirty files from completed Gemma and Qwen runs.

The content audit found no unique unpushed scientific artifact. All 12 SCCKN files matched current GitHub blobs byte for byte. Across CCU, 93 files matched current blobs byte for byte, 13 CSVs contained identical headers and row multisets with different serialization or row order, and two source files matched historical commits already integrated into current `main`.

## Reconciliation

SCCKN dirty files were copied and hash-verified under `/data/scc/emrecan.ulu/git-sync-backups/20260719T105102Z/scckn-active/`, then removed from the active worktree before a fast-forward update. CCU files were similarly preserved under `/home/jovyan/work/git-sync-backups/20260719T105102Z/`; its active checkout was fast-forwarded and its redirected remote was replaced with the canonical HTTPS repository URL.

The CCU provenance worktrees remain pinned at `ac1c643`, `2e4102d`, `44b05c3`, and `b0f87b1`. Their completed outputs are already tracked in canonical `main`, so each worktree is now clean while retaining its original experiment commit.

## Artifact Boundary

The canonical tree contains 1,700 tracked files, including 504 result artifacts and 98 files under the concept-vector output families. No `.safetensors`, PyTorch model shard, GGUF, or comparable foundation-model checkpoint is tracked. The approximately 248 GiB CCU and 408 GiB SCCKN Hugging Face caches remain outside Git, as do virtual environments and resumable scheduler state.

Local-only `ccu/`, `graphify-out/`, `results/graphify-out/`, and `presentation/community-notes-final-presentation-v25.pptx` were preserved unchanged and intentionally excluded from this result-and-weight synchronization.

## Validation

The active local, SCCKN, and CCU checkouts reported the same commit, named `main` branches with configured upstreams, zero ahead and zero behind, and no tracked changes. `git fsck --no-dangling` passed in all three environments. SCCKN and CCU backup manifests rehashed all 12 and 108 preserved files successfully. After this audit record is committed, both remote active checkouts are fast-forwarded once more to the commit containing the report and log entry.
