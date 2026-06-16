# AI Usage Log

Append-only record of every meaningful step taken by an AI agent in this project.
Never edit or delete previous entries — only append.

## Entry Format

```
## YYYY-MM-DD HH:MM · <model-id> — <short title>
- **Step:** what was done
- **Files:** files touched / commands run
- **Result:** outcome (numbers, pass/fail, file paths)
- **Next:** immediate next action (omit if none)
```

- Timestamp: local time, 24-hour clock.
- `<model-id>`: exact model identifier (e.g. `claude-sonnet-4-6`, `gpt-4o`, `cursor-agent`).
- Complements `step_logs/STEP_LOG.md` (human research log). This file is the raw agent-action trail.
- Append even for failed/partial steps — record what was attempted and the error.

---

## 2026-06-16 · claude-sonnet-4-6 — Created AGENTS.md and ai-usage logging infrastructure

- **Step:** Created `AGENTS.md` as tool-agnostic single source of truth for project context; rewrote `CLAUDE.md` to point at it; created `ai-usage/steps.md` with header and entry format.
- **Files:** `AGENTS.md` (new), `CLAUDE.md` (rewritten), `ai-usage/steps.md` (new)
- **Result:** All three files created. `AGENTS.md` contains full project context + AI Usage Logging rule. `CLAUDE.md` is now a 3-line pointer.
- **Next:** Commit the new files to git.

## 2026-06-16 17:00 · claude-opus-4-7 — Implemented Phase 4+5 pipeline (extract_vectors + validate_probes)

- **Step:** Replaced stubs in `src/extract_vectors.py` and `src/validate_probes.py` with full implementations reusing `src/utils/{hooks,model_loader,config,plotting}.py`; rewrote `jobs/sge/extract_vectors.sh` to use 3-node SGE fan-out (`gpu@scc192,gpu@scc213,gpu@scc214`), `wc-tl` conda env, and correct `HF_HOME`.
- **Files:** `src/extract_vectors.py`, `src/validate_probes.py`, `jobs/sge/extract_vectors.sh`
- **Result:** Dry-run PASS — stimulus balance 4×50 confirmed; argparse/config clean. Cross-axis orthogonality test added to validate_probes (new vs smoke tests).
- **Next:** Submit `qsub jobs/sge/extract_vectors.sh` on SCCKN after `git pull`.

## 2026-06-16 21:06 · gpt-5-codex — Revised probe report after figure audit

- **Step:** Implemented the approved report-edit plan: removed unsupported behavioural-independence framing, removed the Fig 4 embed from the report, reframed Fig 1 and §7 around valence overlap, fixed Fig 3 annotations, and regenerated Fig 3 outputs.
- **Files:** `paper/2026-06-16_concept_stories_probe_findings.md`; `paper/figures/generate_figures.py`; `paper/figures/fig3_lorenz_concentration.{png,pdf}`; commands run: `python3 paper/figures/generate_figures.py --fig 3`, `rg` verification checks.
- **Result:** Fig 3 regenerated successfully; visual inspection confirmed Top 11 / 479 / 1426 labels and no horizontal threshold dotted lines. Verification grep found no remaining report references to Fig 4, Cross-axis CV, or full behavioural-independence claims.
- **Next:** Commit and push the revised report, Fig 3 outputs, and logs.
