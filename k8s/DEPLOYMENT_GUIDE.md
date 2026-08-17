# ChemGraph Kubernetes deployment guide

The maintained deployment documentation is now consolidated in:

- [`docs/kubernetes.md`](../docs/kubernetes.md) for the complete user guide;
- [`k8s/README.md`](README.md) for a concise manifest runbook.

Both guides cover the Streamlit and MCP deployments, namespace requirements,
image selection, secrets, cluster-specific proxies, port forwarding,
persistence, exposure, troubleshooting, and cleanup.

Use the checked-in YAML as a template rather than a production-ready security
configuration. Review every manifest for the target cluster before applying it.
