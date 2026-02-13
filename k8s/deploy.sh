#!/bin/bash
# ChemGraph Streamlit Kubernetes Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install it first."
    exit 1
fi

# Check kubectl connection
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

print_info "Connected to Kubernetes cluster:"
kubectl cluster-info | head -1

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default namespace
NAMESPACE="${NAMESPACE:-default}"

# Parse command line arguments
ACTION="${1:-deploy}"

case "$ACTION" in
    deploy)
        print_info "Deploying ChemGraph Streamlit to namespace: $NAMESPACE"

        # Check if secrets file exists
        if [ ! -f "$SCRIPT_DIR/secrets.yaml" ]; then
            print_warn "secrets.yaml not found. Creating from template..."
            cp "$SCRIPT_DIR/secrets.yaml.template" "$SCRIPT_DIR/secrets.yaml"
            print_warn "Please edit k8s/secrets.yaml with your API keys before proceeding."
            print_warn "Press Enter to continue after editing secrets.yaml, or Ctrl+C to cancel..."
            read -r
        fi

        # Apply secrets
        print_info "Creating secrets..."
        kubectl apply -f "$SCRIPT_DIR/secrets.yaml" -n "$NAMESPACE"

        # Apply deployment
        print_info "Creating deployment..."
        kubectl apply -f "$SCRIPT_DIR/deployment.yaml" -n "$NAMESPACE"

        # Apply service
        print_info "Creating service..."
        kubectl apply -f "$SCRIPT_DIR/service.yaml" -n "$NAMESPACE"

        print_info "Waiting for deployment to be ready..."
        kubectl rollout status deployment/chemgraph-streamlit -n "$NAMESPACE" --timeout=5m

        print_info "Deployment successful!"
        print_info "Getting service information..."
        kubectl get svc chemgraph-streamlit -n "$NAMESPACE"

        # Get external IP
        print_info "Waiting for LoadBalancer IP (this may take a few minutes)..."
        for i in {1..30}; do
            EXTERNAL_IP=$(kubectl get svc chemgraph-streamlit -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
            if [ -n "$EXTERNAL_IP" ]; then
                print_info "ChemGraph Streamlit is available at: http://$EXTERNAL_IP:8501"
                break
            fi
            sleep 5
        done

        if [ -z "$EXTERNAL_IP" ]; then
            print_warn "LoadBalancer IP not assigned yet. Run 'kubectl get svc chemgraph-streamlit -n $NAMESPACE' to check."
            print_info "Alternatively, use port-forward: kubectl port-forward svc/chemgraph-streamlit 8501:8501 -n $NAMESPACE"
        fi
        ;;

    status)
        print_info "Checking deployment status in namespace: $NAMESPACE"
        echo ""
        print_info "Pods:"
        kubectl get pods -l app=chemgraph -n "$NAMESPACE"
        echo ""
        print_info "Service:"
        kubectl get svc chemgraph-streamlit -n "$NAMESPACE"
        echo ""
        print_info "Deployment:"
        kubectl get deployment chemgraph-streamlit -n "$NAMESPACE"
        ;;

    logs)
        print_info "Fetching logs from namespace: $NAMESPACE"
        kubectl logs -l app=chemgraph,component=streamlit -n "$NAMESPACE" --tail=100 -f
        ;;

    delete)
        print_warn "This will delete the ChemGraph deployment from namespace: $NAMESPACE"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Deleting resources..."
            kubectl delete -f "$SCRIPT_DIR/deployment.yaml" -n "$NAMESPACE" || true
            kubectl delete -f "$SCRIPT_DIR/service.yaml" -n "$NAMESPACE" || true
            print_info "Keeping secrets (delete manually if needed)"
            print_info "Cleanup complete!"
        else
            print_info "Deletion cancelled"
        fi
        ;;

    port-forward)
        print_info "Setting up port forwarding to localhost:8501"
        kubectl port-forward svc/chemgraph-streamlit 8501:8501 -n "$NAMESPACE"
        ;;

    *)
        echo "Usage: $0 {deploy|status|logs|delete|port-forward}"
        echo ""
        echo "Commands:"
        echo "  deploy        - Deploy ChemGraph Streamlit to Kubernetes"
        echo "  status        - Check deployment status"
        echo "  logs          - View application logs"
        echo "  delete        - Remove ChemGraph deployment"
        echo "  port-forward  - Forward port 8501 to localhost"
        echo ""
        echo "Environment variables:"
        echo "  NAMESPACE     - Kubernetes namespace (default: default)"
        exit 1
        ;;
esac
