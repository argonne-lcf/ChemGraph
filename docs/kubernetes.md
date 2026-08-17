# Kubernetes

The repository provides Kubernetes manifests for two services:

| Component | Deployment and service | Port | Endpoint |
| --- | --- | --- | --- |
| Streamlit UI | `chemgraph-streamlit` | 8501 | `/` |
| General MCP server | `chemgraph-mcp` | 9003 | `/mcp/` |

The files under [`k8s/`](https://github.com/argonne-lcf/ChemGraph/tree/main/k8s)
are deployment templates. Review image tags, proxy settings, secrets, exposure,
storage, and resource limits for your cluster before applying them.

## Prerequisites

- a Kubernetes cluster and permission to create deployments, services, and
  secrets in a namespace;
- `kubectl` configured for that cluster;
- a ChemGraph image tag available from
  `ghcr.io/argonne-lcf/chemgraph` or a custom registry;
- credentials for the model provider used by Streamlit or an MCP client.

The helper script expects its namespace to exist. The manifests themselves do
not create one.

## Quickstart

Clone the repository and enter the manifest directory:

```bash
git clone https://github.com/argonne-lcf/ChemGraph.git
cd ChemGraph/k8s
```

Create or select a namespace:

```bash
kubectl create namespace chemgraph
export NAMESPACE=chemgraph
```

If `chemgraph` already exists, omit the create command. To use another existing
namespace, set `NAMESPACE` to its name.

Create the untracked secret manifest and replace the needed values:

```bash
cp secrets.yaml.template secrets.yaml
```

`k8s/secrets.yaml` is ignored by Git. Do not commit it, paste its contents into
issues, or leave placeholder strings as live credentials. For production, use
your cluster's approved external-secret mechanism instead of a plaintext local
manifest.

Deploy both services with an explicit image tag:

```bash
IMAGE_TAG=<published-tag> NAMESPACE="$NAMESPACE" ./deploy.sh deploy
```

The YAML manifests currently default to the `dev` tag. `IMAGE_TAG` overrides
both deployments without editing the files. Pin a tested version for repeatable
deployments; edit the image references directly if your policy requires an
immutable digest.

## Cluster-specific settings

Both deployment manifests currently contain the ALCF HTTP/HTTPS proxy. On a
non-ALCF cluster, remove or replace the `HTTP_PROXY`, `HTTPS_PROXY`,
`http_proxy`, and `https_proxy` environment entries before deployment.

Also review:

- resource requests of 1 CPU and 2 GiB memory per pod;
- limits of 2 CPUs and 4 GiB memory per pod;
- registry authentication and `imagePullPolicy: Always`;
- whether outbound access to model providers, PubChem, and model downloads is
  permitted;
- whether model/calculator caches need persistent storage.

## Check status and logs

```bash
NAMESPACE="$NAMESPACE" ./deploy.sh status
NAMESPACE="$NAMESPACE" ./deploy.sh logs streamlit
NAMESPACE="$NAMESPACE" ./deploy.sh logs mcp
```

Useful direct checks include:

```bash
kubectl get pods,deployments,services -l app=chemgraph -n "$NAMESPACE"
kubectl describe deployment chemgraph-streamlit -n "$NAMESPACE"
kubectl describe deployment chemgraph-mcp -n "$NAMESPACE"
```

The Streamlit deployment probes `/_stcore/health`. The MCP deployment uses a TCP
probe on port 9003.

## Access locally with port forwarding

Avoid public exposure while validating a deployment:

```bash
NAMESPACE="$NAMESPACE" ./deploy.sh port-forward streamlit
# http://localhost:8501
```

In another terminal:

```bash
NAMESPACE="$NAMESPACE" ./deploy.sh port-forward mcp
# http://localhost:9003/mcp/
```

The `/mcp/` path and trailing slash are required by streamable-HTTP clients.

## LoadBalancer and Ingress

Both services use `type: LoadBalancer` by default. Check assigned addresses:

```bash
kubectl get svc -l app=chemgraph -n "$NAMESPACE"
```

`ingress.yaml` covers only Streamlit. Before applying it, replace
`chemgraph.example.com`, select the installed ingress class, configure TLS, and
remove controller annotations that do not apply to your cluster.

The supplied services and ingress do not add application authentication. Do not
publish Streamlit or the MCP tool server directly to an untrusted network.

## Storage and replicas

The templates do not mount persistent volumes. `CHEMGRAPH_LOG_DIR` points to
`/app/cg_logs`, and local sessions, checkpoints, artifacts, and downloaded data
inside a pod can be lost when that pod is replaced.

Add appropriate persistent volumes or external state services before relying on
durable sessions or artifacts. Do not scale Streamlit horizontally until session
routing and shared persistence are designed; independent replicas do not share
in-process or pod-local state automatically.

## Apply manifests manually

If you do not use `deploy.sh`:

```bash
kubectl apply -f secrets.yaml -n "$NAMESPACE"
kubectl apply -f deployment.yaml -f service.yaml -n "$NAMESPACE"
kubectl apply -f mcp-deployment.yaml -f mcp-service.yaml -n "$NAMESPACE"
```

Apply `ingress.yaml` separately only after customizing it.

## Remove the deployment

```bash
NAMESPACE="$NAMESPACE" ./deploy.sh delete
```

The script intentionally keeps `chemgraph-secrets`. Delete that secret and the
namespace separately only when you are certain they are no longer needed:

```bash
kubectl delete secret chemgraph-secrets -n "$NAMESPACE"
```

## Security checklist

- Use least-privilege RBAC and a dedicated namespace/service account.
- Store credentials with an approved secret manager and rotate them regularly.
- Add network policies for inbound and outbound traffic.
- Put TLS and authentication in front of every externally reachable service.
- Restrict MCP access more tightly than a read-only web application because its
  tools can execute calculations and write files.
- Mount only the data directories that a calculation needs.

For container behavior and local image testing, see [Docker](docker_support.md).
For MCP client details, see [MCP servers](mcp_servers.md).
