import os
from tqdm import tqdm
import json
import re
import torch
import sys
from transformers import AutoProcessor
from openai import OpenAI
from utils import load_jsonl, extract_steps_and_answer, format_answer, format_chat_history, encode_image
from utils import load_config, default_inference_kwargs, default_model_load_kwargs
from prompt import initial_prompt, initial_system_prompt, system_prompt_with_tool_calls
from math_tools import execute_tool_call, execute_tool_call_str
from utils import get_image_mimetype
import time
import requests

def check_vllm_health(base_url="http://localhost:8000", timeout=600):
    """
    Check if the vLLM server is healthy and ready to accept requests.
    """
    start_time = time.time()
    try_cnt = 0
    
    while time.time() - start_time < timeout:
        try_cnt += 1
        print(f"Attempt {try_cnt}: Checking vLLM server health...")
        try:
            # Check server health endpoint
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code != 200:
                print(f"Health check failed with status: {response.status_code}")
                time.sleep(5)
                continue
            
            # Try to get the list of models
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.status_code != 200:
                print(f"Models endpoint failed with status: {response.status_code}")
                time.sleep(5)
                continue
                
            models = response.json()
            if not models.get('data') or len(models['data']) == 0:
                print("No models available yet")
                time.sleep(5)
                continue
            
            print(f"Available models: {[m['id'] for m in models['data']]}")
            
            # Try a simple test request to ensure the model is actually loaded
            test_client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy-key")
            test_response = test_client.chat.completions.create(
                model=models['data'][0]['id'],
                messages=[{"role": "user", "content": "test"}],
                max_tokens=100,
                timeout=5
            )
            
            if test_response.choices[0].message.content:
                print("vLLM server is fully ready!")
                # Close the test client
                del test_client
                return True
            
        except Exception as e:
            print(f"Server check failed: {e}")
        
        print("Waiting for server to be fully ready...")
        time.sleep(5)
    
    print("Timeout waiting for vLLM server to respond")
    return False

def load_model(base_url="http://localhost:8000/v1"):
    # Load local vllm model as openai client
    client = OpenAI(
        base_url=base_url,
        api_key="dummy-key"
    )
    return client

def run_model(client : OpenAI, messages, **kwargs):
    model_name = kwargs.get('model_name', default_model_load_kwargs['model_name'])
    max_tokens = kwargs.get('max_tokens', default_inference_kwargs['max_tokens'])
    temperature = kwargs.get('temperature', default_inference_kwargs['temperature'])
    top_p = kwargs.get('top_p', default_inference_kwargs['top_p'])

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    return completion.choices[0].message.content

def inference(client, image_dir, input_data, output_jsonl, **kwargs):
    """
    Perform inference on the input data and save results to output JSONL file.
    """
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for item in tqdm(input_data, desc="Processing items"):
            image_path = os.path.join(image_dir, item['image'])
            # image_mimetype = get_image_mimetype(item['image'])
            messages = [
                {
                    "role": "system",
                    "content": initial_system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}},
                        {"type": "text", "text": initial_prompt}
                    ],
                }
            ]

            response = run_model(client, messages, **kwargs)
            steps, answer = extract_steps_and_answer(response)
            formatted_answer = format_answer(answer, item.get('tag', None))

            result = {
                'image': item['image'],
                'answer': formatted_answer,
                'raw_answer': answer,
                'steps': steps,
                'output': response
            }
            f_out.write(json.dumps(result) + '\n')

def main(image_dir, input_jsonl, output_jsonl):
    config = load_config()
    model_load_kwargs = {
        k: config.get(k, v) for k, v in default_model_load_kwargs.items()
    }
    inference_kwargs = {
        k: config.get(k, v) for k, v in default_inference_kwargs.items()
    }

    use_tool_calls = config.get('use_tool_calls', False)

    # Check if the vLLM server is healthy
    if not check_vllm_health():
        # Raise an error or exit if the server is not healthy
        raise RuntimeError("vLLM server is not healthy or ready to accept requests within the timeout period.")

    print("vLLM server is healthy and ready to accept requests.")

    print("Creating OpenAI client...")
    # Load the model
    client = load_model()
    print("OpenAI client created successfully.")

    # Load the input JSONL file
    input_data = load_jsonl(input_jsonl)

    # Clear the output file if it exists
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)

    if not use_tool_calls:
        # Perform batch inference
        inference(client, image_dir, input_data, output_jsonl, **inference_kwargs)
    else:
        # Perform batch inference with tool calls
        inference(client, image_dir, input_data, output_jsonl, **inference_kwargs)



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
