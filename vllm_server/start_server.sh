#!/bin/bash

# Get the current time in the format "+%Y%m%d_%H%M%S"
CURRENT_TIME=$(date '+%Y%m%d_%H%M%S')

# Get the directory of the current script
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Define log and PID files according to the script directory
LOG_FILE="$SCRIPT_DIR/vllm_server_$CURRENT_TIME.log"
PID_FILE="$SCRIPT_DIR/vllm_server_$CURRENT_TIME.pid"

# Get command line arguments
MODEL_NAME=${1}
MODEL_PATH=${2}
TENSOR_PARALLEL_SIZE=${3}
GPU_MEMORY_UTILIZATION=${4:-0.9}  # Default to 0.9 if not provided
DTYPE=${5:-"auto"}  # Default to "auto" if not provided

# Start the VLLM server and redirect output to the log file
nohup python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --trust-remote-code \
    --served-model-name $MODEL_NAME \
    --dtype $DTYPE \
    > $LOG_FILE 2>&1 &


# Save the PID of the server process
echo $! > $PID_FILE

echo "VLLM server started with PID $(cat $PID_FILE). Logs are being written to $LOG_FILE."


