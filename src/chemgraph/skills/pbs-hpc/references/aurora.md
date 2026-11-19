# ChemGraph on Aurora

Use the shared environment `/lus/flare/projects/catalyst/world_shared/chemgraph/alcf_chemgraph_env`
unless the user supplies another. Both submission routes source its `bin/activate`
and run its `bin/python`. Report activation failures with the compute-node path.

For a MACE installation with Intel GPU support, set `"device": "xpu"` and
`"default_dtype": "float32"` in the calculator input.

- **Direct PBS.** Create the run under `/lus/flare/projects/<project>/<user>/chemgraph-runs`.
  Copy the [Aurora ChemGraph template](../assets/aurora-chemgraph.pbs.template)
  and fill in the job name, user's project, queue (usually `debug`), and walltime.
  It requests one node and `filesystems=flare`. Follow the
  [PBS workflow](../SKILL.md#prepare-and-track-a-job) to submit and inspect the result.
- **IRI.** Follow [IRI setup](../../iri-hpc/SKILL.md#setup) with `--system aurora`.
  The helper configures the shared environment and stages runs on Flare.

Check `qstat -Qf QUEUE` for current limits. For other resource requests, see the
[ALCF Aurora job guide](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/).
