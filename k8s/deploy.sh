#!/bin/bash
# ChemGraph Kubernetes Deployment Script
# Deploys both Streamlit UI and MCP server

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

# Default namespace (set early so the connection check can use it)
NAMESPACE="${NAMESPACE:-chemgraph}"

# Check kubectl connection using the target namespace
# (cluster-info requires kube-system access which RBAC-limited users may not have)
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    print_error "Cannot access namespace '$NAMESPACE'. Check your kubeconfig and permissions."
    exit 1
fi

print_info "Connected to Kubernetes cluster (namespace: $NAMESPACE)"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Parse command line arguments
ACTION="${1:-deploy}"
TARGET="${2:-}"

# Helper: wait for LoadBalancer IP and print access URL
wait_for_lb() {
    local svc_name="$1"
    local port="$2"
    local label="$3"

    print_info "Waiting for $label LoadBalancer IP..."
    local external_ip=""
    for i in {1..30}; do
        external_ip=$(kubectl get svc "$svc_name" -n "$NAMESPACE" \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        if [ -n "$external_ip" ]; then
            print_info "$label is available at: http://$external_ip:$port"
            return
        fi
        sleep 5
    done
    print_warn "$label LoadBalancer IP not assigned yet."
    print_info "  Check: kubectl get svc $svc_name -n $NAMESPACE"
    print_info "  Or use: $0 port-forward ${svc_name#chemgraph-}"
}

case "$ACTION" in
    deploy)
        print_info "Deploying ChemGraph (Streamlit + MCP) to namespace: $NAMESPACE"

        # Check if secrets file exists
        if [ ! -f "$SCRIPT_DIR/secrets.yaml" ]; then
            print_warn "secrets.yaml not found. Creating from template..."
            cp "$SCRIPT_DIR/secrets.yaml.template" "$SCRIPT_DIR/secrets.yaml"
            print_warn "Please edit k8s/secrets.yaml with your API keys before proceeding."
            print_warn "Press Enter to continue after editing secrets.yaml, or Ctrl+C to cancel..."
            read -r
        fi

        # Apply secrets (shared by both services)
        print_info "Creating secrets..."
        kubectl apply -f "$SCRIPT_DIR/secrets.yaml" -n "$NAMESPACE"

        # Deploy Streamlit
        print_info "Creating Streamlit deployment..."
        kubectl apply -f "$SCRIPT_DIR/deployment.yaml" -n "$NAMESPACE"
        kubectl apply -f "$SCRIPT_DIR/service.yaml" -n "$NAMESPACE"

        # Deploy MCP
        print_info "Creating MCP deployment..."
        kubectl apply -f "$SCRIPT_DIR/mcp-deployment.yaml" -n "$NAMESPACE"
        kubectl apply -f "$SCRIPT_DIR/mcp-service.yaml" -n "$NAMESPACE"

        # Wait for both rollouts
        print_info "Waiting for deployments to be ready..."
        kubectl rollout status deployment/chemgraph-streamlit -n "$NAMESPACE" --timeout=5m
        kubectl rollout status deployment/chemgraph-mcp -n "$NAMESPACE" --timeout=5m

        print_info "Deployment successful!"
        echo ""

        # Show service info
        print_info "Service information:"
        kubectl get svc -l app=chemgraph -n "$NAMESPACE"
        echo ""

        # Wait for LoadBalancer IPs
        wait_for_lb "chemgraph-streamlit" 8501 "Streamlit"
        wait_for_lb "chemgraph-mcp" 9003 "MCP server"
        ;;

    status)
        print_info "Checking deployment status in namespace: $NAMESPACE"
        echo ""
        print_info "Pods:"
        kubectl get pods -l app=chemgraph -n "$NAMESPACE"
        echo ""
        print_info "Services:"
        kubectl get svc -l app=chemgraph -n "$NAMESPACE"
        echo ""
        print_info "Deployments:"
        kubectl get deployment -l app=chemgraph -n "$NAMESPACE"
        ;;

    logs)
        case "$TARGET" in
            streamlit)
                print_info "Fetching Streamlit logs..."
                kubectl logs -l app=chemgraph,component=streamlit -n "$NAMESPACE" --tail=100 -f
                ;;
            mcp)
                print_info "Fetching MCP server logs..."
                kubectl logs -l app=chemgraph,component=mcp -n "$NAMESPACE" --tail=100 -f
                ;;
            *)
                print_info "Fetching logs from all ChemGraph pods..."
                kubectl logs -l app=chemgraph -n "$NAMESPACE" --tail=100 -f --prefix
                ;;
        esac
        ;;

    delete)
        print_warn "This will delete all ChemGraph deployments from namespace: $NAMESPACE"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Deleting resources..."
            kubectl delete -f "$SCRIPT_DIR/mcp-deployment.yaml" -n "$NAMESPACE" || true
            kubectl delete -f "$SCRIPT_DIR/mcp-service.yaml" -n "$NAMESPACE" || true
            kubectl delete -f "$SCRIPT_DIR/deployment.yaml" -n "$NAMESPACE" || true
            kubectl delete -f "$SCRIPT_DIR/service.yaml" -n "$NAMESPACE" || true
            print_info "Keeping secrets (delete manually if needed)"
            print_info "Cleanup complete!"
        else
            print_info "Deletion cancelled"
        fi
        ;;

    port-forward)
        case "$TARGET" in
            streamlit)
                print_info "Port forwarding Streamlit to localhost:8501"
                kubectl port-forward svc/chemgraph-streamlit 8501:8501 -n "$NAMESPACE"
                ;;
            mcp)
                print_info "Port forwarding MCP server to localhost:9003"
                kubectl port-forward svc/chemgraph-mcp 9003:9003 -n "$NAMESPACE"
                ;;
            *)
                print_info "Port forwarding both services (Ctrl+C to stop)..."
                print_info "  Streamlit: http://localhost:8501"
                print_info "  MCP:       http://localhost:9003/mcp/"
                kubectl port-forward svc/chemgraph-streamlit 8501:8501 -n "$NAMESPACE" &
                PF_PID_STREAMLIT=$!
                kubectl port-forward svc/chemgraph-mcp 9003:9003 -n "$NAMESPACE" &
                PF_PID_MCP=$!
                trap "kill $PF_PID_STREAMLIT $PF_PID_MCP 2>/dev/null; exit" INT TERM
                wait
                ;;
        esac
        ;;

    *)
        echo "Usage: $0 {deploy|status|logs|delete|port-forward} [target]"
        echo ""
        echo "Commands:"
        echo "  deploy                  - Deploy Streamlit and MCP server to Kubernetes"
        echo "  status                  - Check deployment status"
        echo "  logs [streamlit|mcp]    - View logs (default: all pods)"
        echo "  delete                  - Remove all ChemGraph deployments"
        echo "  port-forward [streamlit|mcp] - Forward ports to localhost (default: both)"
        echo ""
        echo "Environment variables:"
        echo "  NAMESPACE     - Kubernetes namespace (default: default)"
        exit 1
        ;;
esac
