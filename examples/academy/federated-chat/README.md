# Federated-Chat

Two ChemGraph Academy logical agents running on **different HPCs**
(Aurora and Crux), discovering each other through Academy's hosted
HTTP exchange and exchanging messages across the HPC boundary:

```text
agent-aurora   (running on Aurora)
agent-crux     (running on Crux)
```

The agents play a counter-bouncing game: agent-aurora sends `counter=1`
to agent-crux, agent-crux replies `counter=2`, and so on until the
counter reaches 10. Tiny on purpose — exercises the federated stack
(deterministic peer UIDs, HTTP exchange, multi-site dashboard) without
needing any science tools.

The campaign assets are packaged under:

```text
src/chemgraph/academy/campaigns/federated-chat/
```

Two operator walkthroughs:

- [`e2e_guide.md`](e2e_guide.md) -- four-terminal manual flow
  (dashboard + Aurora compute + Crux compute + bootstrap
  kickoff). Works on both Aurora and Crux. Use this for true
  cross-HPC runs today.
- [`launcher_guide.md`](launcher_guide.md) -- single local
  `chemgraph academy launch` command that ssh's into your
  existing interactive PBS allocations. **Aurora-only for now**
  (Crux compute nodes block laptop-side ssh-to-compute; see
  guide's Crux Limitation section).
