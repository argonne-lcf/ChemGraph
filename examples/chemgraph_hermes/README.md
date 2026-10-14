# ChemGraph with Hermes

This opt-in example lets Hermes delegate chemistry tasks to the ChemGraph CLI.
For Hermes installation and setup, follow the
[official Hermes quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart).

Once Hermes is working:

1. Install [ChemGraph from this checkout](../../CONTRIBUTING.md#development-setup)
   and [configure its model and credentials](../../docs/models.md) in the
   environment used by Hermes's terminal tools. Check that
   `/absolute/chemgraph-env/bin/chemgraph run --help` lists `--result-file` and
   `--answers-file`.

2. From the repository root, copy the bundled skill into Hermes:

   ```bash
   mkdir -p ~/.hermes/skills
   cp -R examples/chemgraph_hermes/skills/chemgraph-delegate ~/.hermes/skills/
   ```

   For a custom Hermes profile, use that profile's `skills/` directory instead.

3. Start a new Hermes conversation with terminal, process, file, and skill tools
   enabled. Invoke `/chemgraph-delegate` and provide the absolute ChemGraph
   executable path and a directory for runs. If you use a
   [TOML config](../../docs/configuration_with_toml.md), provide its absolute path
   too. For example, replacing the paths with your own:

   ```text
   /chemgraph-delegate Build water from SMILES O, optimize it with EMT as a
   demonstration, and report its energy and convergence. Use ChemGraph at
   /absolute/chemgraph-env/bin/chemgraph and save runs under /absolute/cg-runs.
   ```

See the [skill](skills/chemgraph-delegate/SKILL.md) for operating requirements
and the [reference](skills/chemgraph-delegate/reference.md) for handoff details.
