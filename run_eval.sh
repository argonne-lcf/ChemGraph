#!/usr/bin/env bash
set -euo pipefail

: "${ARGO_USER:?Set ARGO_USER to your hosted Argo identity}"
: "${PODMAN_IMAGE:?Set PODMAN_IMAGE to the prepared worker image}"
: "${PODMAN_PROXY_IMAGE:?Set PODMAN_PROXY_IMAGE to the prepared Squid image}"

# Extra CLI flags override these defaults. No shell startup or personal config is loaded.
explicit_output=${OUTPUT_DIR:-}
resume=false
for argument in "$@"; do
    case "$argument" in
        --resume) resume=true ;;
        --output-dir|--output-dir=*) explicit_output=yes ;;
    esac
done
if "$resume" && [[ -z "$explicit_output" ]]; then
    echo 'Resume requires OUTPUT_DIR or --output-dir pointing to the original run.' >&2
    exit 2
fi
output=${OUTPUT_DIR:-eval_results/podman-$(date +%Y%m%d-%H%M%S)}
dataset=${DATASET:-evaluation/output_data.json}
exec chemgraph-eval \
    --eval-runtime podman \
    --podman-image "$PODMAN_IMAGE" --podman-proxy-image "$PODMAN_PROXY_IMAGE" \
    --models argo:gpt-5.6-sol argo:gpt-5.6-luna argo:claude-opus-5 argo:claude-opus-4.8 \
    --workflows deep_agent --judge-type structured --recursion-limit 200 \
    --deepagent-auto-approve --dataset "$dataset" --output-dir "$output" "$@"
