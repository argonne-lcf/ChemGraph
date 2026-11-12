---
name: bebop
description: Supply Bebop machine guidance for PBS resources, queues, allocations, storage, and software environments at Argonne LCRC; combine with the pbs skill for job operations.
license: Apache-2.0
metadata:
  authors: Thang Pham
  maintainers: tdpham2
---

# Bebop machine reference

## 1. Machine identity and use

Use this reference when the target is Argonne LCRC's Bebop cluster. Confirm the
actual host and user configuration before applying its settings. Combine it with
the **pbs** skill for preparation, submission, inspection, control, and recovery;
this reference supplies machine details rather than a second command lifecycle.
Software-specific launch, validation, and restart instructions belong in the
selected software guidance.

## 2. Scheduler and resource conventions

Bebop uses PBS Professional after its 2024 rebuild; older Slurm examples are
historical. Broadwell nodes have 36 cores and 128 GB physical memory. These are
hardware capacities, not a default request or a guarantee of allocatable memory.
([Bebop job guide](https://docs.lcrc.anl.gov/bebop/running-jobs-bebop/))

LCRC documents whole-node scheduling with `select` chunks and default `scatter`
placement. Supply the project account, selection, and walltime explicitly; size
CPU, MPI-rank, thread, and memory requests for the workload. Verify available
resources and any placement override before use.
([PBS conventions](https://docs.lcrc.anl.gov/running-jobs-at-lcrc/pbs-pro/))

## 3. Accounts, queues, and limits

The documented public queues are `bdwall` (default) and `backfill`. Published
limits are 32 running jobs and 100 queued jobs per user, 40 running nodes per
user, 60 per project, and 72 hours maximum walltime. Backfill is restricted to
jobs whose allocation hours ran out, with a documented 12-hour limit subject to
change. Do not select it merely because the regular queue is busy.
([Bebop queues](https://docs.lcrc.anl.gov/bebop/running-jobs-bebop/#queues))

Treat these as a dated snapshot. Inspect current queues/limits with `qstat -Qf`
and confirm project membership, balance, expiration, and queue eligibility at
execution time. Do not infer an allocation from the directory name or another
user's script. `sbank-list-allocations -p PROJECT` inspects the chosen allocation;
Bebop accounts in node hours.
([Allocation accounting](https://docs.lcrc.anl.gov/allocation-management/sbank-allocation-accounting-system/),
[Bebop guide](https://docs.lcrc.anl.gov/bebop/running-jobs-bebop/))

The Bebop page lists a one-hour default walltime while the facility PBS guide
says walltime is mandatory. Always request it explicitly rather than relying on
that discrepancy; 72 hours is a ceiling, not a recommended duration.

## 4. Filesystems, scratch, and quota inspection

Home and project/group storage are shared filesystems. Use the user's verified
project directory, conventionally `/lcrc/project/PROJECT`, for persistent work.
Inspect quota with `lcrc-quota`; check capacity and write access at the actual
output and scratch destinations. `df -h PATH` reports filesystem capacity, not
remaining personal/project quota.
([Quotas](https://docs.lcrc.anl.gov/data-management/filesystem-and-storage/disk-quota/))

Node-local job storage is exposed at `/scratch`, is not shared across nodes,
and is cleared at job end. Normal nodes provide about 15 GB; selected nodes
provide a 4 TB disk. Stage required results back before the job ends. Shared
`/lcrc/globalscratch` is temporary, unbacked storage that may be purged without
notice; it is not an archive. Verify the job's actual mount and free space.
([Filesystems](https://docs.lcrc.anl.gov/data-management/filesystem-and-storage/file-systems/))

For a workload needing the large local disk, Bebop documents `bigdata=true` in
the selection. Verify availability before requesting this limited resource.
([Large local scratch](https://docs.lcrc.anl.gov/bebop/running-jobs-bebop/#special-request-for-large-local-scratch-disk))

## 5. Modules and software-environment discovery

LCRC uses Lmod. Inspect `module list`, `module avail`, and `module spider SOFTWARE`;
use `module show MODULE` and `module help MODULE` to understand prerequisites
and environment changes. Record the selected versions and reproduce their setup
in the job shell. Avoid purging or replacing an existing environment without
checking what the workload needs. Verify the executable with `command -v
EXECUTABLE` after setup. Module availability, versions, licenses, compiler/MPI
compatibility, and user installations must be checked at execution time.
([Software environment](https://docs.lcrc.anl.gov/using-software/getting-started/))

## 6. Machine-specific examples

The following is an illustrative directive fragment, not a runnable job or
resource default. Replace every uppercase placeholder using the workload and
verified site settings. `QUEUE` may be `bdwall` for an eligible Broadwell job.
Use the pbs skill to finish and operate the script.

```text
#PBS -A PROJECT
#PBS -q QUEUE
#PBS -l select=CHUNKS:ncpus=CPUS_PER_CHUNK:mpiprocs=RANKS_PER_CHUNK:mem=MEMORY_PER_CHUNK
#PBS -l walltime=HH:MM:SS
```

Use a PBS memory quantity with units for `MEMORY_PER_CHUNK`. For the large local
disk, append `:bigdata=true` to the chosen chunk specification. Do not copy the
36-CPU, 120-GB, 72-hour request from a particular project as a universal default.
No software module or application installation path is implied by this example.

## 7. Official sources and verification date

Verified against the linked official LCRC pages on **2026-11-12**: Bebop jobs,
facility PBS conventions, sbank accounting, filesystems, quotas, and software
modules. Revisit those sources and inspect the target environment when planning
execution; queue policy and software availability can change. If documentation
and live configuration differ, record the discrepancy and resolve it before
relying on the disputed setting.
