import yaml
import os
import sys
import torch

# Load configuration from YAML file
def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def run_start_command(model_name, model_path, devices, tensor_parallel_size, gpu_memory_utilization, dtype):
    # Set CUDA_VISIBLE_DEVICES environment variable as needed
    if devices != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = devices

    tensor_parallel_size = min(tensor_parallel_size, torch.cuda.device_count())
    # Construct the command to start the VLLM server
    command = [
        'bash', 'vllm_server/start_server.sh',
        model_name,
        model_path,
        str(tensor_parallel_size),
        str(gpu_memory_utilization),
        dtype
    ]
    # Execute the command
    os.system(' '.join(command))

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    model_name = config['model_name']
    model_path = config['model_path']
    devices = config.get('devices', '-1')
    tensor_parallel_size = config.get('tensor_parallel_size', 1)
    gpu_memory_utilization = config.get('gpu_memory_utilization', 0.9)
    dtype = config.get('dtype', 'auto')

    # Run the start command
    run_start_command(model_name, model_path, devices, tensor_parallel_size, gpu_memory_utilization, dtype)