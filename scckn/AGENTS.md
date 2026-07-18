# SCCKN Agent Instructions

This directory is a reusable, cross-project SCCKN operations kit.

Before designing or changing GPU jobs, read `GPU_JOB_DESIGN.md`. Treat its
resource-class hybrid pattern as the default unless the workload requires a
different topology.

## Required job-design behavior

- Inspect the live Grid Engine policy and capacity before drawing conclusions
  about queue position. Never assume that waiting time increases priority.
- Split work by hardware class. Keep work that needs a scarce GPU in as few
  resumable allocations as practical; do not occupy that GPU with work that can
  run on a more common device.
- Pass the chosen priority explicitly to every `qsub` call in a dependency
  chain. `hold_jid` does not inherit priority or reserve the predecessor's host.
- Use realistic `h_rt` values based on `qacct`, with a measured safety margin.
- A bundled job must validate and persist outputs after every stage and create
  a success sentinel only after those checks finish.
- Keep model and dataset caches on work or scratch storage, not in the home
  directory.
- Leave queue names, module versions, scratch paths, GPU resources, priorities,
  and project commands as `# ADJUST` placeholders until confirmed for the
  current project.
- Run `bash -n`, a template dry run, and `qsub -w v` before a real submission.

All new cross-project documentation, filenames, comments, and identifiers in
this directory must be written in English.
