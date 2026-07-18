# SCCKN GPU Job Design Standard

This guide captures the default design rules for GPU pipelines on SCCKN. It is
intended to be copied between projects. Re-check live scheduler configuration
and host resources before applying any dated observation below.

## 1. Start from the scarce resource

Design the job graph around the resource that is hardest to reacquire, usually
a high-memory GPU. Queue dependencies sequence jobs, but they do not keep the
same allocation.

Use this decision table:

| Workload shape | Default design | Reason |
|---|---|---|
| Independent tasks that may run concurrently | Separate jobs or an array job | Exposes parallelism and isolates failures |
| Sequential stages on the same common hardware | Separate jobs when queue reacquisition is cheap | Simple retries and accounting |
| Sequential stages on the same scarce GPU | One resumable staged job | Avoids repeated waits for the scarce allocation |
| Stages requiring different GPU classes | Hybrid split by hardware class | Prevents common-device work from occupying a scarce GPU |
| Very long or weakly checkpointed pipeline | Several checkpointed bundles | Limits failure blast radius |

The cross-project default is a **hybrid pipeline**:

1. Put small-model or common-GPU stages in one job or job family.
2. Put consecutive stages that require the same scarce GPU in one staged job.
3. Add `hold_jid` only when a real data dependency exists between the groups.
4. Validate, sync, and checkpoint after every internal stage.

Avoid both extremes: a long chain that repeatedly relinquishes the scarce GPU,
and a mega-job that wastes the scarce GPU on compatible common-device work.

## 2. What Grid Engine dependencies do not do

`-hold_jid` makes a job ineligible until its predecessor finishes. It does not:

- reserve the predecessor's GPU or host;
- inherit the predecessor's priority;
- guarantee that the successor starts immediately;
- preempt a running job; or
- preserve an in-memory model between jobs.

Consequently, pass the same intended `-p` value to every submitted job rather
than setting it only on the first job.

```bash
first_id=$(qsub -terse -p "$PRIORITY" first_job.sh)
qsub -terse -p "$PRIORITY" -hold_jid "$first_id" second_job.sh
```

For already queued jobs, change priority only when there is a justified project
decision and the site permits it:

```bash
qalter -p "$PRIORITY" <pending-job-id> [<pending-job-id> ...]
```

Do not raise priority merely because a job has waited for a long time. Priority
does not interrupt jobs that are already running.

## 3. SCCKN priority policy observed on 2026-07-15

The live scheduler configuration observed on 2026-07-15 used:

| Component | Weight |
|---|---:|
| Waiting time | `0` |
| Urgency | `0.1` |
| User-assigned priority | `1.0` |
| Tickets | `0.01` |
| Functional tickets | `10000` |
| Share-tree tickets | `0` |

The policy hierarchy was `OFS`, and the functional user, project, department,
and job weights were each `0.25`. Under that snapshot, the displayed priority
was effectively:

```text
priority = 0.1 * nurg + 1.0 * npprior + 0.01 * ntckts
```

A job submitted with `ppri=200` was observed with `npprior=0.59766`; its waiting
contribution was zero. This is evidence about that scheduler snapshot, not a
permanent site guarantee and not a recommendation to use `200` everywhere.

Re-check before relying on the formula:

```bash
qconf -ssconf
qstat -pri -u "$USER"
qstat -urg -u "$USER"
qstat -ext -u "$USER"
```

Use one explicit, moderate priority for a pipeline. Do not silently use the
maximum. If priority does not affect the scheduling bottleneck, leave it alone.

## 4. Staged-job contract

A bundled job must make each stage independently recoverable. For every stage:

1. Skip it when the run-specific success sentinel already exists.
2. Run a pre-stage validation or input check.
3. Execute exactly one stage.
4. Validate expected outputs and metadata.
5. Sync or persist outputs to durable storage.
6. Atomically create the success sentinel.

Use a stable `RUN_ID` and durable `STATE_ROOT` when resubmitting so completed
stages are reused. A failed stage must stop the bundle and must not create its
sentinel. Log the stage name, UTC start and finish times, host, and job ID.

The supplied templates implement this control flow:

```bash
# Inspect without submitting anything
PROJECT_ROOT=/work/<user>/<project> \
RUNNER=/work/<user>/<project>/jobs/sge/staged_gpu_runner.sh \
STATE_ROOT=/work/<user>/<project>/results/scckn-runs \
PRIORITY=<chosen-value> \
COMMON_QSUB_ARGS='-q <common-gpu-queue> -l gpu=1 -l h_rt=02:00:00' \
SCARCE_QSUB_ARGS='-q <scarce-gpu-queue> -l gpu=1 -l h_rt=06:00:00' \
bash hybrid_gpu_submit_template.sh --dry-run
```

Copy the templates into the project and replace their placeholder stage
functions. Do not edit the shared master into a project-specific script.

## 5. Runtime and resource sizing

After a representative run, inspect actual usage:

```bash
qacct -j <job-id>
```

Set `h_rt` from the measured critical path plus a deliberate safety margin. A
useful starting point is 1.5 to 2 times the measured runtime, with at least 30 to
60 minutes for model loading, validation, syncing, and transient slowdown.
Avoid multi-day requests for pipelines that normally finish in hours because
oversized requests can reduce scheduling flexibility and hide missing recovery
logic.

Request only the CPU, RAM, GPU count, and GPU class the stage needs. Confirm
site-specific GPU resource names with `qconf` or the SCCKN administrator.

## 6. Queue diagnosis

Use scheduler evidence instead of inferring queue position from elapsed time:

```bash
qstat -u "$USER"                 # own jobs and states
qstat -f -u '*'                  # jobs by queue instance and user
qstat -j <job-id>                # holds, requests, and scheduling messages
qstat -pri -u '*'                # normalized priority components
qstat -urg -u '*'                # urgency components
qstat -ext -u '*'                # extended scheduling view
qhost -q                         # host and queue availability
qacct -j <job-id>                # completed-job accounting
```

Common states:

| State | Meaning |
|---|---|
| `r` | Running |
| `qw` | Eligible and waiting for scheduling |
| `hqw` | Waiting but held, commonly by the user or `hold_jid` |
| `Eqw` | Queue error; inspect with `qstat -j` |

When a scarce host is full, identify the running owners and resource requests.
Contact another user politely only when coordination is actually useful; their
running job remains entitled to its allocation.

## 7. Submission checklist

Before submission:

- confirm model/data configuration and output paths;
- resolve every `# ADJUST` placeholder;
- put large caches on work or scratch storage;
- run `bash -n` on submitter and runner;
- run the submitter with `--dry-run`;
- run `qsub -w v` with the final resource arguments;
- verify every submitted job receives the intended priority;
- verify `hold_jid` matches real data dependencies; and
- ensure logs, validation, durable output sync, and run-specific sentinels exist.

After completion, inspect `qacct`, record failures and runtimes, and use those
measurements to tune the next submission.
