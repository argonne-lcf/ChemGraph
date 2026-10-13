# Hermes integration

This directory preserves Connect's Hermes integration at upstream `7232594`,
including stable message IDs, unknown-acceptance recovery, clarification replies,
CLI delegation and approval identity checks. Install/copy this plugin in Hermes's
own environment. `__init__.py` uses only Python's standard library; it never
imports ChemGraph.

Set `CHEMGRAPH_CONNECT_URL` and the appropriate peer token
`CHEMGRAPH_CONNECT_TOKEN`. Keep `CHEMGRAPH_CONNECT_OPERATOR_TOKEN` separately
controlled. Preserve the existing Hermes plugin registration procedure and
trusted Telegram/WhatsApp approval-actor configuration. Discord support is a
later milestone. Check compatibility with the deployed Hermes plugin API before
upgrading either host.

Run the notification helper without installing scientific dependencies:

```sh
python -m pip install 'httpx>=0.28,<0.29'
python notify.py --url https://service.example --token-env CHEMGRAPH_NOTIFIER_TOKEN --recipients recipients.json
```

`recipients.json` maps authenticated service peer names to fixed Telegram targets,
for example `{"researcher": "telegram:12345"}`. Use a notifier-role token.
The helper sends status references, never arbitrary model-supplied media paths.
Use `--once` for a deployment check. Ambiguous delivery requires operator evidence
at `/admin/notifications/{identifier}/resolve`; it is never blindly resent.
Artifacts continue to require authenticated retrieval from their owner account.
