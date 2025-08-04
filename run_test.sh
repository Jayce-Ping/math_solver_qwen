# conda activate pytorch-env

# Start the VLLM server
python vllm_server/deploy_server.py

# Sleep for 60 seconds to allow the server to start
sleep 60

python run_api.py \
  ../math_solver_qwen_data/sample_data \
  ../math_solver_qwen_data/sample_data/input.jsonl \
  ../math_solver_qwen_data/sample_data/output.jsonl

# python run_api.py \
#   $IMAGE_INPUT_DIR \
#   $QUERY_PATH \
#   $OUPUT_PATH

# Stop the VLLM server
bash vllm_server/stop_server.sh