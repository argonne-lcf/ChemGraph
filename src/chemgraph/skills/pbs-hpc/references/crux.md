# Crux

Use the official [Crux guides](https://docs.alcf.anl.gov/crux/) together with
[ALCF PBS guidance](https://docs.alcf.anl.gov/running-jobs/). Verify the applicable
queue, allocation, node request, modules, and launch command on the target system.

ChemGraph's `get_crux_config` uses `LocalProvider` within an existing allocation
and requires `PBS_NODEFILE`. Its worker initialization can be configured through
`CHEMGRAPH_WORKER_INIT`. Check CPU process/thread settings for the actual
application; do not copy a GPU configuration from another facility.
