#!/usr/bin/env bash
# Scripted demo launcher for the RAG Knowledge Engine.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
TENANT="${2:-acme}"

echo "========================================================"
echo "  RAG Knowledge Engine - Live Demonstration Launcher"
echo "========================================================"

# Check if service is responsive
if ! curl -sf "${BASE_URL}/health/live" > /dev/null 2>&1; then
    echo "Service at ${BASE_URL} is not responding. Starting via Docker Compose..."
    docker compose up -d --wait
fi

# Run the scripted demo
python3 scripts/demo.py --base-url "${BASE_URL}" --tenant "${TENANT}"
