# ALCF Aurora

Aurora uses Intel GPUs, PBS Pro, and the PALS MPI launcher. Consult the official
[machine overview](https://docs.alcf.anl.gov/aurora/) and
[job guide](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/) for current site
settings. Queue and GPU-binding guidance below was checked on 2026-09-14; verify
the deployed environment before preparing a job. Troubleshooting observations
from local operations are identified separately from documented site policy.

## System basics

- Login through `aurora.alcf.anl.gov`; login nodes use names such as
  `aurora-uan-0012`. Build and submit from the appropriate project environment;
  run substantial computation on allocated compute nodes.
- The system has 10,624 compute nodes. Each has two Intel Xeon Max 9470C CPUs
  (104 physical cores, 208 hardware threads) and six Intel Data Center GPU Max
  1550 (PVC) devices, each with two tiles. Use `lscpu` and `numactl -H` to inspect
  the allocation's CPU/NUMA layout.
- Compute names have the form `xRRRRcCsSbBnN`, such as `x4204c1s0b0n0`;
  high-speed-network names add `.hsn.cm.aurora.alcf.anl.gov`.
- Cores 0 and 52 are reserved for system services. Exclude them from CPU binding.
  GPUs 0–2 are closest to socket-0 cores 1–51; GPUs 3–5 to socket-1 cores 53–103.
- Project storage is under `/lus/flare/projects/PROJECT`; obtain the actual
  allocation and project directory from deployment configuration. Submit from
  that directory. Declare required filesystems, for example `filesystems=flare`;
  `qstat -Bf` reports `resources_available.valid_filesystems`.
- Use the site's oneAPI compilers, Aurora MPICH, MKL, and Level Zero environment.
  Inspect `module list` and `module avail` instead of pinning a release from an
  example. Confirm the calculator's Intel GPU support; Polaris CUDA settings
  are not interchangeable with Aurora settings.
- Direct SSH to assigned compute nodes is supported. The job guide requires
  both the user's home and `.ssh` directories to have mode `700` for SSH/SCP.

## PBS queues and submission

Use `qsub`, `qstat`, `qdel`, `pbsnodes`, and `pbsdsh`. The primary submission
queues below have a five-minute minimum walltime. Confirm their current limits
with `qstat -Qf QUEUE` before choosing resources.

| Submit queue | Nodes | Maximum walltime | Notes |
| --- | --- | --- | --- |
| `debug` | 1–2 | 1 h | 64-node pool; one running/accruing/queued job per user |
| `debug-scaling` | 2–256 | 1 h | One running/accruing/queued job per user |
| `capacity` | 1–16 | 168 h | 512 nodes across the queue; at most five queued-or-running jobs and two running jobs per user |
| `prod` | 256–10,624 | Depends on routed tier | Submit here for production execution queues |

`prod` routes by node count: `small` (256–1,024, 12 h), `medium` (1,025–1,999,
18 h), and `large` (2,000–10,624, 24 h). The corresponding `backfill-*` queues
have lower priority and support projects with negative balances. These are
execution queues: **submit through `prod`, not directly to a tier**. Check
current project limits, access restrictions, and usable node counts; the
system's total size is not a promise of simultaneous availability.

Temporary evaluation queues and the by-request `visualization` queue may also
be available. Consult their current eligibility rules rather than using them as
general alternatives. Do not assume a fixed allocation discount for backfill.

For a one-node interactive test, replace the project value and first change to
its shared project directory:

```bash
PROJECT="your-project"
qsub -I -A "$PROJECT" -q debug -l select=1 \
    -l walltime=00:30:00 -l filesystems=flare
```

For a batch job, read the parent skill's PBS template, copy it into the execution
workspace, fill every placeholder, and choose Aurora-specific launch settings.
Keep directives before executable statements, validate with `bash -n`, and
follow the parent skill's submit-once and job-tracking workflow.

## Choosing resources and interpreting scheduling

1. Inspect queue and node state before recommending a layout:
   ```bash
   pbsnodes -aSj
   qstat -Qf debug debug-scaling capacity prod small medium large
   qstat -u "$USER"
   ```
2. For short tests, use `debug` or `debug-scaling` when the job fits and policy
   permits. For longer jobs of up to 16 nodes, consider `capacity`, including
   long-lived workflow drivers. Compare its assigned nodes with the current
   aggregate cap; unused queue capacity alone does not establish eligibility.
3. For 256 or more nodes, choose `prod` and a walltime within the routed tier's
   limit. A 17–255-node job lasting over one hour does not fit the general queues
   above; check site options rather than increasing the node count solely to
   qualify for a queue.
4. Request a realistic walltime, including setup, staging, and final output.
   Shorter requests may fit reservation gaps more easily, but do not guarantee
   an earlier start. An empty execution queue does not prove that it can run a
   new job immediately.

Local observations have included WFP-related attributes such as `enable_wfp`,
`base_score`, and `score_boost`. Use exposed attributes and scheduler comments
as diagnostics; do not infer a particular WFP formula, fixed seed, or guaranteed
ranking from them. If `qstat -T` reports `--`, report that no start estimate is
available. Queue-state snapshots cannot establish a predicted start time.

A rejection mentioning the generic queue's per-user limit can reflect limits
across queues. Inspect the current user's jobs and server/queue limits instead
of assuming a fixed three- or four-job threshold. Wait for eligibility or combine
compatible work in one allocation; repeated submissions do not resolve the limit.

## Job queries and allocated nodes

```bash
JOB_ID="your-job-id"
qstat -f "$JOB_ID"
qstat -f "$JOB_ID" | grep -E 'queue|base_score|score_boost|walltime|nodect|comment'
```

`Q` means queued, `R` running, `E` exiting, and `H` held. Preserve the complete
job ID. Check scheduler history, exit status, stdout/stderr, and application
results before claiming completion; a job disappearing from `qstat` is not
proof of success.

Inside an allocation, obtain distinct hosts from `PBS_NODEFILE`:

```bash
sort -u "${PBS_NODEFILE:?An active PBS allocation is required}" > hosts.txt
xargs -P 8 -I{} ssh -o ConnectTimeout=10 {} hostname < hosts.txt
```

Use only allocated hosts and preserve the site's SSH host verification. Replace
`hostname` with a lightweight diagnostic appropriate to the task. `pbsdsh` can
also launch commands on the allocation's PBS tasks. Avoid broad filesystem scans
or large process fan-out from shared login nodes.

Outside the allocation, inspect the running job's `exec_host`/`exec_vnode`
attributes. PBS text output can wrap values across lines: a parser must join
continuations until the next attribute (which may contain dots) or end of output.
Strip slot suffixes from `exec_host` entries and deduplicate hostnames; compare
with `Resource_List.nodect` when present. Queued jobs may have no assigned hosts.

## PALS and Python MPI environments

Check `command -v mpiexec` after loading modules and again after activating a
Python environment. Aurora uses PALS, commonly installed under
`/opt/cray/pals/<version>/bin/mpiexec`. Python environments containing other MPI
launchers can shadow it. PALS supports `--depth` and `--cpu-bind depth`; do not
pass those options to an unrelated launcher such as Intel MPI Hydra.

Local PALS launches have exposed `PALS_RANKID`, `PMIX_RANK`, `PALS_LOCAL_RANKID`,
`PALS_LOCAL_SIZE`, and `PALS_NODEID`, without exporting a global size variable.
Code requiring both a rank and an assumed `PALS_WORLD_SIZE` can therefore fall
back to serial incorrectly. A launcher rank variable establishes launch context,
not that there is more than one rank. Initialize the intended MPI runtime and
use `MPI.COMM_WORLD.Get_size()` for the actual size. Deriving size from local
size times node count is valid only for a confirmed uniform layout.

Build mpi4py against the loaded Aurora MPI when a prebuilt wheel does not match
the site's MPI stack. In the intended Python environment, with the matching
compiler wrapper on PATH:

```bash
MPICC="$(command -v mpicc)" uv pip install --no-binary mpi4py mpi4py
```

Confirm the installed MPI library and launcher agree before scaling a test.

## GPU hierarchy and per-rank affinity

`ZE_AFFINITY_MASK` limits visible devices for each process; a comma-separated
mask does not assign a different device to each rank. When all ranks see the same
mask and the application always selects device 0, ranks contend for that device.
Applications that implement their own rank-to-device mapping can intentionally
share visibility, so do not overwrite existing application/launcher assignments
without checking their purpose.

Follow the [Aurora binding examples](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/#binding-mpi-ranks-to-gpus):

- In `COMPOSITE` mode, six GPU devices expose two tiles each. The site's
  `gpu_tile_compact.sh` binds ranks to tiles such as `0.0` and `0.1`;
  `gpu_dev_compact.sh` binds whole GPUs. Confirm the script is on PATH and place
  it immediately before the application in `mpiexec`.
- The `frameworks` module uses `FLAT` mode, exposing 12 tiles as devices numbered
  0–11. Do not reuse the COMPOSITE tile script or dotted tile IDs unchanged.
  Prefer the framework/application's supported local-rank device selection.
- The documented `--gpu-bind` option has limitations for whole-device binding
  and `FLAT` mode. Validate the current MPI version and a small affinity test
  before using it at scale.

For a FLAT-mode application that relies on the default visible device, the
following example assigns one tile per rank. It assumes all 12 tiles are
available, no preexisting device restrictions, and at most 12 local ranks.
Save it as `set_affinity_gpu_aurora_flat.sh` in the execution workspace:

```bash
#!/bin/bash
: "${PALS_LOCAL_RANKID:?Run this wrapper under PALS mpiexec}"
if [[ "${ZE_FLAT_DEVICE_HIERARCHY:-}" != FLAT ||
      ! "$PALS_LOCAL_RANKID" =~ ^([0-9]|1[01])$ ]]; then
    echo "Expected FLAT hierarchy and a local rank from 0 to 11" >&2
    exit 1
fi
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK="$PALS_LOCAL_RANKID"
exec "$@"
```

After loading the application environment and confirming `FLAT`, a one-node
test within an allocation can use:

```bash
mpiexec -n 12 --ppn 12 --depth 1 --cpu-bind=list:1:2:3:4:5:6:53:54:55:56:57:58 \
    bash ./set_affinity_gpu_aurora_flat.sh ./app
```

Adapt CPU binding and threading to the application. Inspect each rank's selected
device before a long run. For applications that select devices themselves,
restricting visibility to one device can conflict with their local-rank indexing.

## Group-restricted /soft paths

Some `/soft/applications` installations require an application-specific UNIX
group. Local operations have reported access failures even when `id -Gn` lists
the required group, particularly for users with many supplementary groups. This
is an observed diagnostic lead, not a verified universal NGROUPS threshold.

First check group membership, parent-directory permissions, and the site's
application access requirements. If the user already belongs to the authorized
application group, test access with that group made primary:

```bash
sg APPLICATION_GROUP -c 'ls -l /soft/applications/APPLICATION/VERSION/bin/EXECUTABLE'
```

Replace all uppercase placeholders with the installed application's values.
`newgrp APPLICATION_GROUP` provides an interactive subshell; exit it to restore
the previous group context. Neither command grants membership or an application
license. Group switching can remove `LD_LIBRARY_PATH` and other environment
settings; load the required oneAPI/MKL/MPI environment inside the new group
context and verify its propagation to ranks before launching the application.
For persistent failures, report the membership and path evidence to ALCF support
rather than assuming that trimming the group list is the correct fix.

## GPU monitoring

Check tools in the compute-node environment. ALCF documents
[`xpu-smi`](https://docs.alcf.anl.gov/aurora/performance-tools/xpu-smi/), including
`module load xpu-smi`, device discovery, and sampled utilization. Availability
depends on the installed image and module environment; an absent command on a
login node does not prove that the facility lacks it.

Local deployments have also used Intel PTI-GPU `sysmon`, often under
`/opt/aurora/<release>/support/tools/pti-gpu/<version>/bin/sysmon`. Locate the
installed version or obtain its path from deployment configuration; do not use
another user's private binary path. Check `sysmon -h`: observed versions support
`-p` for processes, `-l` for devices, and `-d` for details.

`sysmon -p` can report device memory, clocks, temperature, and attached processes
with engine labels. Process presence and allocated memory show attachment, not
necessarily active computation. Clock frequency, temperature, and engine labels
alone are not utilization percentages. When a sysmon version has no utilization
counter, use sampled `xpu-smi` or Level Zero Sysman engine-activity metrics to
measure activity. Keep cross-node comparisons consistent in tool version and
sampling interval.

## ChemGraph execution

`get_aurora_config` uses `LocalProvider` within an existing PBS allocation and
requires `PBS_NODEFILE`; it does not acquire an allocation. Configure worker
setup through `worker_init`, `CHEMGRAPH_WORKER_INIT`, or the intended Python
environment. Its fallback loads `frameworks`, so check the resulting GPU
hierarchy before applying any affinity wrapper.

Ensure inputs, worker environments, and output directories are visible throughout
the allocation. Skill files, the `execute` shell, an HPC MCP server, and remote
workers may have different filesystems. Copy required templates/helpers into the
actual execution workspace and establish worker visibility before submission.
