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

See [`e2e_guide.md`](e2e_guide.md) for the full four-terminal walkthrough
(dashboard + Aurora compute + Crux compute + bootstrap kickoff).
