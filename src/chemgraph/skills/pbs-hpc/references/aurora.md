# Aurora

Consult the official [Aurora job guide](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/)
for current resource syntax, queue limits, filesystem options, and launch
examples. Queue names and limits should be verified when preparing the job.

Do not reuse Polaris CUDA settings on Aurora. Check the calculator and runtime's
support for the target accelerator and the site's required environment.

ChemGraph's `get_aurora_config` uses an existing PBS allocation and requires
`PBS_NODEFILE`; it does not submit a new allocation. Keep environment setup in
deployment configuration or `CHEMGRAPH_WORKER_INIT` and ensure worker paths are
visible within the allocation.
