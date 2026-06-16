# Example 002: MACE Ensemble Screening

This example demonstrates five persistent ChemGraph Academy logical agents
running under MPI:

```text
coordinator-agent
structure-agent-a
structure-agent-b
mace-agent
assessment-agent
```

The coordinator delegates 20 SMILES candidates, structure agents generate XYZ
files, the MACE agent runs an ensemble energy screen, and the assessment agent
summarizes readiness/ranking evidence.

The campaign assets are packaged under:

```text
src/chemgraph/academy/campaigns/example-002-mace-ensemble-screening/
```

Run it by campaign name:

```bash
chemgraph academy run-compute \
  --system aurora \
  --run-id aurora-mace-ensemble-screening-001 \
  --campaign mace-ensemble-screening-20 \
  --lm-user <argo-user>
```

See `notes.md` for the high-level architecture notes. The internal E2E user
guide is intentionally not stored in this public example directory.
