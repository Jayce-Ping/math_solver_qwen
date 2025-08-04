#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# List all .pid files in the script directory
PID_FILES=$(find "$SCRIPT_DIR" -maxdepth 1 -name "*.pid")

# Check if any PID files were found
if [ -z "$PID_FILES" ]; then
    echo "No PID files found. The server might not be running."
    exit 0
fi

# Loop through each PID file and stop the server
for PID_FILE in $PID_FILES; do
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
done