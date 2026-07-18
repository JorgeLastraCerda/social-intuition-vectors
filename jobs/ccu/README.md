# CCU Gemma 4 calibrated queue

This directory is the tracked, project-specific execution backend for the CCU
H100. It does not contain CCU credentials or modify the reusable local client in
`ccu/`.

The workflow uses one visible H100 and separate model processes in this fixed
order: 12B, 26B-A4B, then 31B. A technical failure stops the queue. Scientific
effect size is descriptive and never gates later models.

Remote defaults:

- repository: `/home/jovyan/work/normalcy-axis`
- virtual environment: `/home/jovyan/.venvs/normalcy-gemma4-cu124`
- Hugging Face cache: `/home/jovyan/work/hf_cache`
- checkpoints, logs, queue state, and sentinels:
  `/home/jovyan/work/normalcy-gemma4-state`

After cloning the committed repository on CCU, run:

```bash
bash jobs/ccu/bootstrap_gemma4.sh
bash jobs/ccu/run_gemma4_calibrated_queue.sh
```

Rerunning the queue is safe. Completed models are validated against their
sentinels, and interrupted models resume only when their code, configuration,
model revision, input hashes, split, seed, strengths, and intervention settings
match the checkpoint manifest exactly.
