#!/usr/bin/env bash
# start_with_proxy.sh
# Launches the Claude Code LLM proxy then the MiroFish backend.
# Run from the repo root: ./start_with_proxy.sh

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=== MiroFish + Claude Code Proxy ==="

# 1. Start the proxy in the background
echo "[1/2] Starting Claude proxy on http://localhost:8765/v1 ..."
python "$ROOT/claude_proxy.py" &
PROXY_PID=$!

# Wait for the proxy to come up
for i in $(seq 1 10); do
    if curl -sf http://localhost:8765/health > /dev/null 2>&1; then
        echo "      Proxy is ready."
        break
    fi
    sleep 1
done

# 2. Start the MiroFish backend
echo "[2/2] Starting MiroFish backend on http://localhost:5001 ..."
cd "$ROOT/backend"
python run.py

# On exit, stop the proxy too
trap "echo 'Stopping proxy...'; kill $PROXY_PID 2>/dev/null" EXIT
