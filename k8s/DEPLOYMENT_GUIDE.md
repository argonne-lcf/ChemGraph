# ChemGraph Streamlit Kubernetes Deployment Guide

This guide provides step-by-step instructions for deploying the ChemGraph Streamlit application on a Kubernetes cluster.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Configuration Options](#configuration-options)
5. [Accessing the Application](#accessing-the-application)
6. [Troubleshooting](#troubleshooting)
7. [Production Considerations](#production-considerations)

## Prerequisites

Before you begin, ensure you have:

- A Kubernetes cluster (v1.19+)
- `kubectl` installed and configured
- Access to push/pull Docker images (or use the public `ghcr.io/argonne-lcf/chemgraph:latest`)
- API keys for at least one LLM provider (OpenAI, Anthropic, Google, or Groq)

## Quick Start

The fastest way to deploy ChemGraph Streamlit:

```bash
cd k8s

# 1. Create and configure secrets
cp secrets.yaml.template secrets.yaml
# Edit secrets.yaml with your API keys

# 2. Deploy using the deployment script
./deploy.sh deploy

# 3. Check status
./deploy.sh status

# 4. Access via port-forward (for testing)
./deploy.sh port-forward
# Then open http://localhost:8501 in your browser
```

## Detailed Setup

### Step 1: Create Secrets

Kubernetes Secrets are used to securely store your API keys.

```bash
# Copy the template
cp secrets.yaml.template secrets.yaml

# Edit the file with your actual API keys
vim secrets.yaml
```

Your `secrets.yaml` should look like:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: chemgraph-secrets
type: Opaque
stringData:
  openai-api-key: "sk-proj-..."
  anthropic-api-key: "sk-ant-..."
  gemini-api-key: "..."
  groq-api-key: "..."
```

**Important:** Do NOT commit this file! It's already in `.gitignore`.

Apply the secret:

```bash
kubectl apply -f secrets.yaml
```

### Step 2: Deploy the Application

Apply the deployment manifest:

```bash
kubectl apply -f deployment.yaml
```

This creates a Deployment with:
- 1 replica (can be scaled)
- 2Gi memory request, 4Gi limit
- 1 CPU request, 2 CPU limit
- Health checks (liveness and readiness probes)
- Environment variables for API keys

### Step 3: Create the Service

Apply the service manifest:

```bash
kubectl apply -f service.yaml
```

This creates a LoadBalancer service that exposes the Streamlit app on port 8501.

### Step 4: Verify Deployment

Check the status:

```bash
# Check pods
kubectl get pods -l app=chemgraph

# Check deployment
kubectl get deployment chemgraph-streamlit

# Check service
kubectl get svc chemgraph-streamlit

# View logs
kubectl logs -l app=chemgraph,component=streamlit -f
```

## Configuration Options

### Using a Custom Namespace

Create and use a dedicated namespace:

```bash
kubectl create namespace chemgraph
kubectl apply -f secrets.yaml -n chemgraph
kubectl apply -f deployment.yaml -n chemgraph
kubectl apply -f service.yaml -n chemgraph
```

Or use the deploy script:

```bash
NAMESPACE=chemgraph ./deploy.sh deploy
```

### Scaling the Deployment

To run multiple replicas:

```bash
kubectl scale deployment chemgraph-streamlit --replicas=3
```

Or edit `deployment.yaml`:

```yaml
spec:
  replicas: 3
```

### Adjusting Resources

Edit `deployment.yaml` to change resource allocation:

```yaml
resources:
  requests:
    memory: "4Gi"    # Increase for larger workloads
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Using a Custom Docker Image

If you've built your own image:

1. Build and push to your registry:
   ```bash
   docker build -t your-registry/chemgraph:v1.0 .
   docker push your-registry/chemgraph:v1.0
   ```

2. Update `deployment.yaml`:
   ```yaml
   spec:
     containers:
     - name: streamlit
       image: your-registry/chemgraph:v1.0
   ```

## Accessing the Application

### Method 1: LoadBalancer (Cloud Providers)

If your cluster supports LoadBalancer services:

```bash
# Get the external IP
kubectl get svc chemgraph-streamlit

# Access the app
# http://<EXTERNAL-IP>:8501
```

### Method 2: NodePort (On-Premise)

Edit `service.yaml`:

```yaml
spec:
  type: NodePort
```

Apply and get the NodePort:

```bash
kubectl apply -f service.yaml
kubectl get svc chemgraph-streamlit

# Access at http://<NODE-IP>:<NODE-PORT>
```

### Method 3: Port Forwarding (Development)

For local development/testing:

```bash
kubectl port-forward svc/chemgraph-streamlit 8501:8501

# Access at http://localhost:8501
```

Or use the deploy script:

```bash
./deploy.sh port-forward
```

### Method 4: Ingress (Production)

For production with a domain name:

1. Install an Ingress controller (e.g., nginx-ingress)
2. Edit `ingress.yaml` with your domain
3. Apply:
   ```bash
   kubectl apply -f ingress.yaml
   ```

## Troubleshooting

### Pods Not Starting

Check pod events:

```bash
kubectl describe pod <pod-name>
```

Common issues:
- **ImagePullBackOff**: Check image name/tag and registry access
- **CrashLoopBackOff**: Check logs with `kubectl logs <pod-name>`
- **Pending**: Check if nodes have sufficient resources

### Application Not Accessible

1. Verify the pod is running:
   ```bash
   kubectl get pods -l app=chemgraph
   ```

2. Check service endpoints:
   ```bash
   kubectl get endpoints chemgraph-streamlit
   ```

3. Test from inside the cluster:
   ```bash
   kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
     curl http://chemgraph-streamlit:8501/_stcore/health
   ```

### API Key Issues

If the app can't access LLMs:

1. Verify secrets are created:
   ```bash
   kubectl get secret chemgraph-secrets -o yaml
   ```

2. Check if environment variables are set in pod:
   ```bash
   kubectl exec <pod-name> -- env | grep API_KEY
   ```

3. View application logs:
   ```bash
   kubectl logs <pod-name>
   ```

### Performance Issues

1. Check resource usage:
   ```bash
   kubectl top pod <pod-name>
   ```

2. Increase resource limits in `deployment.yaml`

3. Scale horizontally:
   ```bash
   kubectl scale deployment chemgraph-streamlit --replicas=3
   ```

## Production Considerations

### Security

1. **Use Secrets Management**: Consider using external secrets (Vault, AWS Secrets Manager)

   Example with External Secrets Operator:
   ```yaml
   apiVersion: external-secrets.io/v1beta1
   kind: ExternalSecret
   metadata:
     name: chemgraph-secrets
   spec:
     secretStoreRef:
       name: vault-backend
       kind: SecretStore
     target:
       name: chemgraph-secrets
     data:
     - secretKey: openai-api-key
       remoteRef:
         key: chemgraph/openai
   ```

2. **Enable RBAC**: Create a service account with minimal permissions

3. **Network Policies**: Restrict pod-to-pod communication

   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: chemgraph-network-policy
   spec:
     podSelector:
       matchLabels:
         app: chemgraph
     policyTypes:
     - Ingress
     - Egress
     ingress:
     - from:
       - podSelector: {}
       ports:
       - protocol: TCP
         port: 8501
     egress:
     - to:
       - namespaceSelector: {}
   ```

4. **Enable TLS**: Use Ingress with cert-manager for HTTPS

### High Availability

1. **Multiple Replicas**: Run at least 3 replicas

2. **Pod Disruption Budget**:
   ```yaml
   apiVersion: policy/v1
   kind: PodDisruptionBudget
   metadata:
     name: chemgraph-pdb
   spec:
     minAvailable: 1
     selector:
       matchLabels:
         app: chemgraph
   ```

3. **Anti-Affinity**: Spread pods across nodes
   ```yaml
   affinity:
     podAntiAffinity:
       preferredDuringSchedulingIgnoredDuringExecution:
       - weight: 100
         podAffinityTerm:
           labelSelector:
             matchExpressions:
             - key: app
               operator: In
               values:
               - chemgraph
           topologyKey: kubernetes.io/hostname
   ```

### Monitoring

1. **Prometheus + Grafana**: Monitor resource usage and application metrics

2. **Logging**: Use Fluentd/Fluent Bit to collect logs

3. **Health Checks**: Already configured in `deployment.yaml`

### Backup and Recovery

1. **Persistent Storage**: If needed, mount a PVC for data persistence

   ```yaml
   volumes:
   - name: data
     persistentVolumeClaim:
       claimName: chemgraph-data-pvc
   ```

2. **Disaster Recovery**: Keep deployment manifests in version control

## Helper Commands

```bash
# Deploy
./deploy.sh deploy

# Check status
./deploy.sh status

# View logs
./deploy.sh logs

# Port forward
./deploy.sh port-forward

# Delete
./deploy.sh delete

# Manual commands
kubectl get all -l app=chemgraph
kubectl describe pod <pod-name>
kubectl logs -f <pod-name>
kubectl exec -it <pod-name> -- /bin/bash
```

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [ChemGraph Documentation](../README.md)
- [Streamlit Deployment Guide](https://docs.streamlit.io/deploy)
- [Docker Guide](../docs/docker_support.md)
