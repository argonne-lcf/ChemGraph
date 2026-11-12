# Bebop machine reference

## Machine

Use this reference for Argonne LCRC's Bebop cluster with the
[pbs-hpc workflow](../SKILL.md).

## Resources

Bebop uses PBS Professional. Broadwell nodes have 36 cores and 128 GB physical
memory; choose requests for the workload and verify allocatable resources.
LCRC uses whole-node scheduling with `select` chunks and default `scatter`
placement. Specify project, resources, and walltime explicitly.

## Accounts, queues, and limits

Published limits, to verify with current `qstat -Qf` output and site policy:

| Setting | Documented value |
| --- | --- |
| Default queue | `bdwall` |
| Per-user jobs | 32 running, 100 queued |
| Running nodes | 40 per user, 60 per project |
| Maximum walltime | 72 hours |
| `backfill` | For exhausted allocations; 12-hour limit, subject to change |

Confirm project membership, queue eligibility, balance, and expiration.
Use `sbank-list-allocations -p PROJECT`; accounting is in node hours.
Always set walltime explicitly: the Bebop page lists a default, while the
facility PBS guide requires it. Limits and hardware capacities are not defaults
for individual jobs.

## Filesystems and scratch

Use the verified shared project directory, conventionally
`/lcrc/project/PROJECT`, for persistent work. Check quota with `lcrc-quota`;
`df -h PATH` reports filesystem capacity rather than project quota.

`/scratch` is node-local and cleared at job end. Normal nodes provide about
15 GB; request `bigdata=true` in the select specification for nodes with a
4 TB disk. Verify available space and stage results back before exit.
`/lcrc/globalscratch` is shared, unbacked temporary storage subject to deletion
without notice.

## Software environment

Use Lmod: `module list`, `module avail`, `module spider SOFTWARE`, and
`module show MODULE`/`module help MODULE`. Verify versions, prerequisites,
licenses, and compiler/MPI compatibility; reproduce the selected setup in the
job shell. Check the executable with `command -v EXECUTABLE` after setup.

## Example directives

When adapting the shared PBS template, omit its ALCF `system` and `filesystems`
directives and use the Bebop resource selection below. Replace placeholders
with workload settings; include units in the memory value.
Append `:bigdata=true` to the selection only when large local scratch is needed.

```text
#PBS -A PROJECT
#PBS -q QUEUE
#PBS -l select=CHUNKS:ncpus=CPUS_PER_CHUNK:mpiprocs=RANKS_PER_CHUNK:mem=MEMORY_PER_CHUNK
#PBS -l walltime=HH:MM:SS
```

## Official sources

Verified **2026-11-12**. Recheck changing limits and software availability before
execution; resolve discrepancies with the live configuration.

- [Bebop jobs and queues](https://docs.lcrc.anl.gov/bebop/running-jobs-bebop/)
- [PBS conventions](https://docs.lcrc.anl.gov/running-jobs-at-lcrc/pbs-pro/)
- [Allocation accounting](https://docs.lcrc.anl.gov/allocation-management/sbank-allocation-accounting-system/)
- [Filesystems](https://docs.lcrc.anl.gov/data-management/filesystem-and-storage/file-systems/)
- [Quotas](https://docs.lcrc.anl.gov/data-management/filesystem-and-storage/disk-quota/)
- [Software modules](https://docs.lcrc.anl.gov/using-software/getting-started/)
