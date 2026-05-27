# Compute Notes

The default compute target is SCCKN at Universität Konstanz.

## Scheduler

SCCKN uses Grid Engine / `qsub`. Do not use SLURM / `sbatch` scripts unless the project is moved to a different cluster and a new scheduler backend is added.

Heavy work should run through scripts in `jobs/sge/`.

## SCCKN Commands

```bash
qsub jobs/sge/<job>.sh
qstat -u emrecan.ulu
qdel <job_id>
qacct -j <job_id>
qstat -j <job_id>
```

## Storage

Large model caches should not live in the home directory. Set `HF_HOME` to scratch or work storage in every job script.

```bash
export HF_HOME=/path/to/scratch/hf_cache  # ADJUST
```

## Future Scheduler Backends

If another machine or cluster becomes available, add a separate backend directory such as `jobs/slurm/` or `jobs/local/`. Keep the Python entrypoints unchanged and only swap the job wrapper.
