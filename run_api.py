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

def build_client(model_name, base_url="http://localhost:8000", timeout=600):
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
            
            # Try a simple test request to ensure the model is actually loaded
            client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy-key")
            test_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=100,
                timeout=5
            )
            
            if test_response.choices[0].message.content:
                print("vLLM server is fully ready!")
                return client
            
        except Exception as e:
            print(f"Server check failed: {e}")
        
        print("Waiting for server to be fully ready...")
        time.sleep(5)
    
    print("Timeout waiting for vLLM server to respond")
    return None

def run_model(client : OpenAI, messages, **kwargs):
    model_name = kwargs.get('model_name', default_model_load_kwargs['model_name'])
    max_tokens = kwargs.get('max_tokens', default_inference_kwargs['max_tokens'])
    temperature = kwargs.get('temperature', default_inference_kwargs['temperature'])
    top_p = kwargs.get('top_p', default_inference_kwargs['top_p'])

    timeout=30

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
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
                        # {"type": "text", "text": ""}
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


def inference_with_tool_calls(client, image_dir, input_data, output_jsonl, **kwargs):
    
    max_tool_calls = kwargs.get('max_tool_calls', 3)
    max_chat_round = kwargs.get('max_chat_round', 3)
    tool_call_count = 0

    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for item in tqdm(input_data, desc="Processing items with tool calls"):
            image_path = os.path.join(image_dir, item['image'])
            messages = [
                {
                    "role": "system",
                    "content": system_prompt_with_tool_calls
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}},
                        {"type": "text", "text": initial_prompt}
                    ],
                }
            ]

            chat_history = messages            
            for _ in range(max_chat_round):
                if tool_call_count >= max_tool_calls:
                    break

                response = run_model(client, chat_history, **kwargs)
                chat_history.append({
                    "role": "assistant",
                    "content": response
                })

                # Extract tool calls from the response and execute them
                match_tool_calls = re.finditer(r'<tool_call>(.*?)</tool_call>', response)
                for match_tool_call in match_tool_calls:
                    tool_call_str = match_tool_call.group(1)
                    tool_call_res = execute_tool_call_str(tool_call_str)
                    if 'result' in tool_call_res:
                        chat_history.append({
                            "role": "assistant",
                            "content": f"{tool_call_str} returned: {tool_call_res['result']}"
                        })
                    else:
                        # An error occurred in tool execution
                        chat_history.append({
                            "role": "assistant",
                            "content": f"Error executing tool {tool_call_str}: {tool_call_res['error']}"
                        })
                    
                    tool_call_count += 1
                    if tool_call_count >= max_tool_calls:
                        break

                # Check if the response contains the final answer
                steps, answer = extract_steps_and_answer(response)
                if len(answer) > 0:
                    break
        
            # Extract steps and answer from the final response
            final_response = chat_history[-1]['content']
            steps, answer = extract_steps_and_answer(final_response)
            formatted_answer = format_answer(answer, item.get('tag', None))

            result = {
                'image': item['image'],
                'answer': formatted_answer,
                'raw_answer': answer,
                'steps': steps,
                'output': format_chat_history(chat_history, roles=['assistant'])
            }
            
            # Write to output JSONL file
            with open(output_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main(image_dir, input_jsonl, output_jsonl):
    config = load_config()
    model_load_kwargs = {
        k: config.get(k, v) for k, v in default_model_load_kwargs.items()
    }
    inference_kwargs = {
        k: config.get(k, v) for k, v in default_inference_kwargs.items()
    }

    use_tool_calls = config.get('use_tool_calls', False)

    print("Creating OpenAI client...")
    client = build_client(model_name=model_load_kwargs['model_name'])
    if client is None:
        # Raise an error
        raise RuntimeError("Failed to create OpenAI client to connect to vLLM server in time.")

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
        inference_with_tool_calls(client, image_dir, input_data, output_jsonl, **inference_kwargs)



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
