# ChemGraph Academy Sim

`chemgraph.academy_sim` is a side-by-side replacement path for the current
`chemgraph.academy` integration. It treats Academy as a graph-to-graph message
transport only. ChemGraph owns scientific reasoning, MCP/science tool loading,
and peer-tool invocation decisions.

## Demo 1: two single-agent graphs on one site

Start the ChemGraph MCP server:

```bash
python -m chemgraph.mcp.mcp_tools --transport streamable_http --host 127.0.0.1 --port 9003
```

Start the executor first, then the planner. The demo uses Academy's HTTP
exchange and deterministic graph UIDs, so no registration file copy is needed.

```bash
LAUNCH_TOKEN="$(date +%s)"

python -m chemgraph.academy_sim.launcher \
  --config scripts/academy_sim/configs/demo1_same_site_two_single_agent.jsonc \
  --graph executor \
  --run-token demo1 \
  --launch-token "${LAUNCH_TOKEN}"

python -m chemgraph.academy_sim.launcher \
  --config scripts/academy_sim/configs/demo1_same_site_two_single_agent.jsonc \
  --graph planner \
  --run-token demo1 \
  --launch-token "${LAUNCH_TOKEN}"
```

`LAUNCH_TOKEN` is only used by the legacy file-registration path. The provided
demo configs use exchange registration, where every graph derives peer UIDs from
`run_id` and graph name.

Peer-send tools are deterministic turn boundaries. After a graph calls
`send_message_to_*`, the current ChemGraph turn ends and the process waits for a
future inbound peer message. When a graph produces a normal final assistant
answer without calling a peer-send tool, the graph is marked complete and the
process shuts down after queued work drains.

If a graph is waiting and no new peer message arrives, the process exits through
the graph's `idle_timeout_s` fallback after there are no queued messages and no
active graph runs. The demo configs set this to 60 seconds.

## Demo 2: two single-agent graphs on two sites

Run `executor` on the site with the MCP server, and run `planner` on the other
site. Both processes use the hosted Academy HTTP exchange and the same `run_id`.
After both processes print that peers are live, send the startup message once:

```bash
python -m chemgraph.academy_sim.bootstrap \
  --config scripts/academy_sim/configs/demo2_two_site_two_single_agent.jsonc
```

The HTTP exchange uses Globus Auth. On ALCF systems, make sure the Academy token
is cached and `http_proxy` / `https_proxy` point at the site proxy before
launching the graph processes.

## Demo 3: multi-agent peer graph

Use `demo3_two_site_multi_agent_peer.jsonc`. The `executor` graph is a
ChemGraph `multi_agent` graph with the same peer communication surface. Launch
the two graph processes independently, then run:

```bash
python -m chemgraph.academy_sim.bootstrap \
  --config scripts/academy_sim/configs/demo3_two_site_multi_agent_peer.jsonc
```
