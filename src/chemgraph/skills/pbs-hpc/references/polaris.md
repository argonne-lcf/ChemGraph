# Polaris

Read the official [Polaris job guide](https://docs.alcf.anl.gov/polaris/running-jobs/)
for current queues, resource requests, and MPI launch examples. Determine the
project allocation and required filesystems before preparing the job.

For GPU work, follow the [Polaris GPU guide](https://docs.alcf.anl.gov/polaris/running-jobs/using-gpus/).
Validate the application's accelerator support inside the compute environment;
do not infer that every calculator supports GPUs.

ChemGraph's `get_polaris_config` uses `PBS_NODEFILE` to size an existing allocation
and launches workers using `LocalProvider` and `MpiExecLauncher`. Its fallback
to one node outside PBS is intended for local/testing contexts, not automatic
scheduler submission. Preserve the application-specific MPI/GPU layout.
