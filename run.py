from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from utils import load_jsonl, extract_steps_and_answer, format_answer
from utils import load_config, default_inference_kwargs, default_model_load_kwargs
import os
from tqdm import tqdm
import json
import re
import torch
import sys
from prompt import initial_prompt, format_query

def load_model(**kwargs):
    model_path = kwargs.get('model_path', None)
    if model_path is None:
        raise ValueError("Model path must be specified in the arguments.")

    # Get keyword arguments for model loading
    gpu_memory_utilization = kwargs.get('gpu_memory_utilization', default_model_load_kwargs['gpu_memory_utilization'])
    max_model_len = kwargs.get('max_model_len', default_model_load_kwargs['max_model_len'])
    dtype = kwargs.get('dtype', default_model_load_kwargs['dtype'])

    # default: Load the model on the available device(s)
    torch_device_cnt = torch.cuda.device_count()
    device_cnt = kwargs.get('device_cnt', torch_device_cnt)
    device_cnt = min(device_cnt, torch_device_cnt)
    if device_cnt < 1:
        raise ValueError("No CUDA devices available for model loading.")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=device_cnt,  # Use all available GPUs
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return llm, processor

def run_model(model, processor, messages_list, **kwargs):
    if isinstance(messages_list, dict):
        messages_list = [messages_list]
    
    if not isinstance(messages_list, list):
        raise ValueError("messages_list must be a list of messages or a single message dictionary.")

    max_tokens = kwargs.get('max_tokens', default_inference_kwargs['max_tokens'])
    temperature = kwargs.get('temperature', default_inference_kwargs['temperature'])
    top_p = kwargs.get('top_p', default_inference_kwargs['top_p'])

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )

    batch_inputs = []
    for messages in messages_list:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, _ = process_vision_info(messages)

        mm_data = {'image': image_inputs}

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        batch_inputs.append(llm_inputs)

    outputs = model.generate(batch_inputs, sampling_params=sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]

    # Return the generated texts
    return generated_texts


def main(image_dir, input_jsonl, output_jsonl):
    config = load_config()
    model_load_kwargs = {
        k: config.get(k, v) for k, v in default_model_load_kwargs.items()
    }
    inference_kwargs = {
        k: config.get(k, v) for k, v in default_inference_kwargs.items()
    }
    # Load the model and processor
    model, processor = load_model(**model_load_kwargs)

    # Load the input JSONL file
    input_data = load_jsonl(input_jsonl)

    # Clear the output file if it exists
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)

    # Process the input data in batches
    batches = []
    for start_index in range(0, len(input_data), inference_kwargs['batch_size']):
        end_index = min(start_index + inference_kwargs['batch_size'], len(input_data))
        batch = input_data[start_index:end_index]
        batches.append(batch)

    res = []
    for batch in tqdm(batches, desc="Processing batches"):
        messages_list = []
        for obj in batch:
            image_path = os.path.join(image_dir, obj['image'])
            prompt = obj.get('query', initial_prompt)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages_list.append(messages)

        generated_texts = run_model(
            model,
            processor,
            messages_list,
            **inference_kwargs
        )
        for obj, generated_text in zip(batch, generated_texts):
            # Extract steps and answer from the generated text
            steps, answer = extract_steps_and_answer(generated_text)
            formatted_answer = format_answer(answer, obj.get('tag', None))
            obj["answer"] = formatted_answer
            obj['raw_answer'] = answer
            obj["steps"] = steps
            obj['output'] = generated_text

            # res.append(obj)
            with open(output_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
