# Kubernetes deployment for ChemGraph

This directory contains templates for deploying the ChemGraph Streamlit UI and
general MCP server. Read the canonical
[Kubernetes guide](../docs/kubernetes.md) before applying them; it covers image
tags, cluster-specific proxy settings, storage, exposure, and security limits.

## Contents

| File | Purpose |
| --- | --- |
| `deployment.yaml` / `service.yaml` | Streamlit deployment and LoadBalancer service on port 8501 |
| `mcp-deployment.yaml` / `mcp-service.yaml` | General MCP deployment and LoadBalancer service on port 9003 |
| `ingress.yaml` | Optional, placeholder Streamlit ingress |
| `secrets.yaml.template` | Untracked provider-secret template |
| `deploy.sh` | Deploy, inspect, forward, or delete both services |

## Quickstart

The helper requires an existing namespace:

```bash
kubectl create namespace chemgraph
export NAMESPACE=chemgraph
```

If the namespace already exists, omit the create command.

Create the ignored secret file and replace only the credentials you need:

```bash
cp secrets.yaml.template secrets.yaml
```

Never commit `secrets.yaml`. For production, use the secret-management system
approved for your cluster.

Deploy both components with an explicit image tag:

```bash
IMAGE_TAG=<published-tag> NAMESPACE="$NAMESPACE" ./deploy.sh deploy
```

The manifests default to the `dev` image and include ALCF proxy environment
variables. Review both deployment YAML files before deploying to another site.

## Inspect and access

```bash
NAMESPACE="$NAMESPACE" ./deploy.sh status
NAMESPACE="$NAMESPACE" ./deploy.sh logs streamlit
NAMESPACE="$NAMESPACE" ./deploy.sh logs mcp
```

Use separate terminals for local-only access:

```bash
NAMESPACE="$NAMESPACE" ./deploy.sh port-forward streamlit
# http://localhost:8501
```

```bash
NAMESPACE="$NAMESPACE" ./deploy.sh port-forward mcp
# http://localhost:9003/mcp/
```

## Manual deployment

```bash
kubectl apply -f secrets.yaml -n "$NAMESPACE"
kubectl apply -f deployment.yaml -f service.yaml -n "$NAMESPACE"
kubectl apply -f mcp-deployment.yaml -f mcp-service.yaml -n "$NAMESPACE"
```

Customize `ingress.yaml` before applying it; the checked-in hostname and ingress
class are placeholders.

## Cleanup

```bash
NAMESPACE="$NAMESPACE" ./deploy.sh delete
```

The helper keeps `chemgraph-secrets`. Delete it separately when it is no longer
needed.

The templates do not provide persistent volumes or application authentication.
Do not expose either LoadBalancer publicly until persistence, TLS,
authentication, RBAC, network policy, and secret management are addressed.
