# ChemGraph on Crux

Use the shared environment `/grand/catalyst/world-shared/chemgraph/crux/alcf_chemgraph_env`
unless the user supplies another. Both submission routes source its `bin/activate`
and run its `bin/python`. Report activation failures with the compute-node path.

Crux compute nodes are CPU-only. Set `"device": "cpu"` for MACE.

- **Direct PBS.** Create the run under `/eagle/<project>/<user>/chemgraph-runs`.
  Copy the [Crux ChemGraph template](../assets/crux-chemgraph.pbs.template)
  and fill in the job name, user's project, queue (usually `debug`), and walltime.
  It requests one node and `filesystems=home:eagle:grand`. Follow the
  [PBS workflow](../SKILL.md#prepare-and-track-a-job) to submit and inspect the result.
- **IRI.** Follow [IRI setup](../../iri-hpc/SKILL.md#setup) with `--system crux`.
  The helper configures the shared environment and stages runs on Eagle.

Check `qstat -Qf QUEUE` for current limits. For other resource requests, see the
[ALCF Crux job guide](https://docs.alcf.anl.gov/crux/queueing-and-running-jobs/running-jobs/).
