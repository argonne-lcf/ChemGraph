# Kubernetes Deployment for ChemGraph Streamlit App

This directory contains Kubernetes manifests to deploy the ChemGraph Streamlit application on a Kubernetes cluster.

## Prerequisites

- A running Kubernetes cluster
- `kubectl` configured to communicate with your cluster
- Docker image available at `ghcr.io/argonne-lcf/chemgraph:latest` (or build your own)
- API keys for the LLM providers you want to use

## Files

- `deployment.yaml` - Deployment manifest for the Streamlit app
- `service.yaml` - Service manifest to expose the app
- `secrets.yaml.template` - Template for storing API keys securely
- `ingress.yaml` - (Optional) Ingress configuration for external access

## Quick Start

### 1. Create the Secrets

First, copy the secrets template and fill in your API keys:

```bash
cp secrets.yaml.template secrets.yaml
```

Edit `secrets.yaml` and replace the placeholder values with your actual API keys:

```yaml
stringData:
  openai-api-key: "sk-..."
  anthropic-api-key: "sk-ant-..."
  gemini-api-key: "..."
  groq-api-key: "..."
```

**Important:** Never commit `secrets.yaml` to version control! Add it to `.gitignore`.

Apply the secret:

```bash
kubectl create namespace chemgraph  # Optional: create a dedicated namespace
kubectl apply -f secrets.yaml
```

### 2. Deploy the Application

Apply the deployment and service manifests:

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### 3. Access the Application

Check the status of your deployment:

```bash
kubectl get pods -l app=chemgraph
kubectl get svc chemgraph-streamlit
```

Get the external IP (if using LoadBalancer type):

```bash
kubectl get svc chemgraph-streamlit
```

Once the service has an external IP, access the Streamlit app at:
```
http://<EXTERNAL-IP>:8501
```

If using a cloud provider, it may take a few minutes for the LoadBalancer to provision an external IP.

## Alternative Access Methods

### Using NodePort

If your cluster doesn't support LoadBalancer, edit `service.yaml` and change the service type:

```yaml
spec:
  type: NodePort
```

Then access the app using any node IP and the assigned NodePort:

```bash
kubectl get svc chemgraph-streamlit
# Access at http://<NODE-IP>:<NODE-PORT>
```

### Using Port Forwarding (Development)

For local testing, use port forwarding:

```bash
kubectl port-forward svc/chemgraph-streamlit 8501:8501
```

Then access at `http://localhost:8501`

### Using Ingress (Production)

For production deployments with a domain name, use the included ingress configuration:

```bash
kubectl apply -f ingress.yaml
```

Make sure you have an Ingress controller installed (like nginx-ingress or traefik) and update the host in `ingress.yaml`.

## Customization

### Resource Limits

Edit `deployment.yaml` to adjust resource requests and limits:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Replicas

To run multiple replicas for high availability:

```yaml
spec:
  replicas: 3
```

### Using a Custom Image

If you've built your own ChemGraph image:

```yaml
containers:
- name: streamlit
  image: your-registry/chemgraph:your-tag
```

## Monitoring and Troubleshooting

### Check Pod Status

```bash
kubectl get pods -l app=chemgraph
kubectl describe pod <pod-name>
```

### View Logs

```bash
kubectl logs -f deployment/chemgraph-streamlit
```

### Check Health Probes

The deployment includes liveness and readiness probes that check the Streamlit health endpoint:

```bash
kubectl describe pod <pod-name> | grep -A 10 "Liveness\|Readiness"
```

### Common Issues

**Pod not starting:**
- Check if the image can be pulled: `kubectl describe pod <pod-name>`
- Verify secrets are created: `kubectl get secrets`

**Application crashes:**
- Check logs: `kubectl logs <pod-name>`
- Verify API keys are correct
- Check resource limits are sufficient

**Cannot access the service:**
- Verify service is running: `kubectl get svc`
- Check if LoadBalancer has an external IP assigned
- Verify firewall rules allow traffic on port 8501

## Cleanup

To remove all ChemGraph resources:

```bash
kubectl delete -f deployment.yaml
kubectl delete -f service.yaml
kubectl delete secret chemgraph-secrets
```

## Security Considerations

1. **Never commit secrets to version control**
2. **Use RBAC** to limit access to the namespace
3. **Enable TLS** for production deployments (use Ingress with cert-manager)
4. **Rotate API keys** regularly
5. **Use network policies** to restrict pod-to-pod communication if needed
6. **Consider using a secrets management solution** like HashiCorp Vault or Sealed Secrets

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [ChemGraph README](../README.md)
- [Streamlit Documentation](https://docs.streamlit.io/)
