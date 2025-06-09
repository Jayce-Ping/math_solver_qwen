from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import json
import re
import torch
import sys

initial_prompt="""You are an expert in solving mathematical problems. Please begin by extracting the math problem from the provided image, and then solve it.

Requirements:
	1.	All mathematical formulas and symbols in your response must be written in LaTeX format.
	2.	Organize your response according to the following structure:
	•	Restate the Problem: Clearly and concisely describe the math problem shown in the image.
	•	Solution Approach: Outline your reasoning and the steps taken to solve the problem.
	•	Final Answer: Present the complete solution.
	3. The final answer can only contain the final result number or option.

Strictly follow the format below in your output:
### Think ###
<Restate the problem and outline the solution approach>

### Answer ###
<Final answer>"""

def extract_steps_and_answer(response):
    """
    从模型响应中提取 <Restate the problem and outline the solution approach> 和 <Final answer>
    """
    # 定义正则表达式模式
    restate_pattern = r"### Think ###\n(.*?)\n### Answer ###"
    answer_pattern = r"### Answer ###\n(.*)"

    # 匹配 <Restate the problem and outline the solution approach>
    restate_match = re.search(restate_pattern, response, re.DOTALL)
    step = restate_match.group(1).strip() if restate_match else ""

    # 匹配 <Final answer>
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""

    return step, answer

def load_model(model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    # default processor
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def load_jsonl(input_file):
    """加载jsonl文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def main(image_dir,input_jsonl, output_jsonl):
    model, processor = load_model("Qwen/Qwen2.5-VL-3B-Instruct")
    input_file = load_jsonl(input_jsonl)


    res = []
    for obj in tqdm(input_file, desc="Processing"):
        image_path = os.path.join(image_dir, obj['image'])
        # Prepare the messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": initial_prompt},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        step, answer = extract_steps_and_answer(output_text)
        obj["step"] = step
        obj["answer"] = answer
        res.append(obj)
    # Save the results to a JSONL file
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in res:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')



if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
