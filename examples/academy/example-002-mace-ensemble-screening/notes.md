# Notes

This root example directory is for user-facing explanation only. The CLI loads
the actual campaign from package data so installed ChemGraph environments can
run the same campaign without relying on a source checkout's root `examples/`
directory.

Packaged assets:

```text
src/chemgraph/academy/campaigns/example-002-mace-ensemble-screening/
  campaign.jsonc
  lm_config.json
  prompt_profiles/
  data/
  models/
```

The campaign declares MCP server subprocesses for general ChemGraph tools, MACE
screening, and HPC utility inspection. The Academy runtime places one logical
agent per MPI rank, launches the declared MCP servers for each agent, and uses
Academy exchange handles for peer communication.
