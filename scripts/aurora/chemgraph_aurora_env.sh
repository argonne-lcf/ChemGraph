#!/bin/bash
# Point ChemGraph at a running Aurora llama.cpp LLM endpoint.
#
# Reads the ENDPOINT.txt written by the two-node serve PBS and exports the
# variables the `aurora:` provider uses. Source it (do not execute):
#
#   source scripts/aurora/chemgraph_aurora_env.sh [/path/to/ENDPOINT.txt]
#   chemgraph run --model aurora:gpt-oss-120b -q "What is the SMILES for water?"
#
# If ChemGraph runs on a different node than the server, first open a tunnel from
# a login node (compute nodes have no public IP), e.g.
#   ssh -N -L 8000:<node_ip>:8000 <you>@aurora.alcf.anl.gov
# and set AURORA_BASE_URL=http://127.0.0.1:8000/v1 instead.

_ep="${1:-ENDPOINT.txt}"
if [ ! -f "$_ep" ]; then
  echo "ENDPOINT file not found: $_ep" >&2
  echo "Start the server first: scripts/aurora/serve_llamacpp_maxctx.sh <model-key>" >&2
  return 1 2>/dev/null || exit 1
fi

_url=$(grep '^url=' "$_ep" | cut -d= -f2-)
_model=$(grep '^chemgraph_model=' "$_ep" | cut -d= -f2-)
export AURORA_BASE_URL="$_url"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"   # llama-server does not enforce auth by default

echo "AURORA_BASE_URL=$AURORA_BASE_URL"
echo "Use: chemgraph run --model ${_model:-aurora:gpt-oss-120b} -q \"...\""
