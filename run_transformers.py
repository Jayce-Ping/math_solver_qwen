import os
from tqdm import tqdm
import json
import re
import torch
import sys
from prompt import initial_prompt, format_query, format_prompt
from utils import load_jsonl, extract_steps_and_answer, format_answer
from utils import default_inference_kwargs, default_model_load_kwargs,load_config
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def load_model(model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
    # default: Load the model on the available device(s)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    # default processor
    processor = AutoProcessor.from_pretrained(model_name)

    processor.tokenizer.padding_side = 'left'

    return model, processor

def run_model(model, processor, messages_list, **kwargs):

    if isinstance(messages_list, dict):
        messages_list = [messages_list]
    
    if not isinstance(messages_list, list):
        raise ValueError("messages_list must be a list of messages or a single message dictionary.")

    max_tokens = kwargs.get('max_tokens', default_inference_kwargs['max_tokens'])
    temperature = kwargs.get('temperature', default_inference_kwargs['temperature'])
    top_p = kwargs.get('top_p', default_inference_kwargs['top_p'])

    # Batch all inputs
    batch_texts = []
    batch_image_inputs = []
    batch_video_inputs = []
    
    for messages in messages_list:
        # Prepare input for each input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        batch_texts.append(text)
        batch_image_inputs.extend(image_inputs if image_inputs else [])
        batch_video_inputs.extend(video_inputs if video_inputs else [])

    # Batch input
    inputs = processor(
        text=batch_texts,
        images=batch_image_inputs if batch_image_inputs else None,
        videos=batch_video_inputs if batch_video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Batch inference
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True if temperature > 0 else False,
    )
    
    # Batch decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_texts

def main(image_dir,input_jsonl, output_jsonl):
    config = load_config()
    
    model_path = config.get('model_path', "Qwen/Qwen2.5-VL-3B-Instruct")
    
    inference_kwargs = {
        k: config.get(k, v) for k, v in default_inference_kwargs.items()
    }

    # Load the model and processor
    model, processor = load_model(model_path)

    # Load the input JSONL file
    input_data = load_jsonl(input_jsonl)

    # Clear the output file if it exists
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)

    # Process the input file in batches
    batch_size = inference_kwargs.pop('batch_size', 4)
    batches = []
    for start_index in range(0, len(input_data), batch_size):
        end_index = min(start_index + batch_size, len(input_data))
        batch = input_data[start_index:end_index]
        batches.append(batch)

    res = []
    for batch in tqdm(batches, desc="Processing batches"):
        # Prepare messages for each batch
        messages_list = []
        for obj in batch:
            image_path = os.path.join(image_dir, obj['image'])
            prompt = obj.get('query', format_prompt(obj['tag']))
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

        # Batch inference
        output_texts = run_model(
            model,
            processor,
            messages_list,
            **inference_kwargs
        )

        # Extract steps and answers from the output texts
        for obj, output_text in zip(batch, output_texts):
            steps, answer = extract_steps_and_answer(output_text)
            formatted_answer = format_answer(answer, obj.get('tag', None))
            obj["answer"] = formatted_answer
            obj['raw_answer'] = answer
            obj["steps"] = steps
            obj['output'] = output_text
            res.append(obj)

            with open(output_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
