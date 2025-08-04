#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PID_FILE="$SCRIPT_DIR/vllm_server.pid"
LOG_FILE="$SCRIPT_DIR/vllm_server.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Stopping vLLM API Server (PID: $PID)..."
    pkill -P $PID
    # Wait for the process to terminate
    wait $PID 2>/dev/null
    rm "$PID_FILE"
    echo "vLLM API Server stopped"
else
    echo "No PID file found. Server might not be running."
fi