from utils import extract_steps_and_answer, format_answer, calculate_answer
from utils import load_jsonl
import json
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the results of the math solver.")
    parser.add_argument("--input-jsonl", type=str, required=True, help="Input JSONL file with image paths.")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Output JSONL file to save results.")
    parser.add_argument("--eval-jsonl", type=str, help="JSONL file to save evaluation results.")
    return parser.parse_args()


def compare_choice_answer(choice, answer):
    """Compare a choice with an answer and return True if they match."""
    if choice is None or answer is None:
        return False

    return choice.lower() == answer.lower()

def compare_numerical_answer(predicted, ground_truth):
    """Compare a numerical answer with the ground truth."""
    if predicted is None or ground_truth is None:
        return False

    predicted_value = calculate_answer(predicted)
    ground_truth_value = calculate_answer(ground_truth)
    try:
        return abs(predicted_value - ground_truth_value) < 1e-3
    except:
        return False
    

def compare_answer(a1: str, a2: str, problem_type=None):
    """Compare two answers and return True if they are similar."""
    if problem_type == '选择题':
        return compare_choice_answer(a1, a2)
    elif problem_type in ['填空题', '计算应用题']:
        return compare_numerical_answer(a1, a2)
    else:
        return compare_choice_answer(a1, a2) or compare_numerical_answer(a1, a2)
    

def eval(input_data, output_data):
    # Pair data items by their 'image' key  
    paired_data = {item['image']: item for item in input_data}
    results: list[dict] = []
    
    for item in output_data:
        image = item['image']
        if image in paired_data:
            input_item = paired_data[image]
            problem_type = input_item.get('tag', None)
            ground_truth_answer = input_item.get('answer', '')
            steps, predicted_answer = extract_steps_and_answer(item.get('output', ''))
            predicted_answer = format_answer(predicted_answer, problem_type)

            is_correct = compare_answer(predicted_answer, ground_truth_answer, problem_type)
            result = {
                "image": image,
                "predicted_answer": predicted_answer,
                "ground_truth_answer": ground_truth_answer,
                "problem_type": problem_type,
                "is_correct": is_correct,
                "steps": steps
            }
            results.append(result)

    return results


if __name__ == "__main__":
    args = parse_args()
    input_data = load_jsonl(args.input_jsonl)
    output_data = load_jsonl(args.output_jsonl)
    eval_jsonl = args.eval_jsonl
    if not eval_jsonl:
        eval_jsonl = args.output_jsonl.replace('.jsonl', '_eval.jsonl')

    results = eval(input_data, output_data)

    with open(eval_jsonl, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


    # Print summary of evaluation results
    print(f"Total items evaluated: {len(results)}")
    print(f"Average accuracy: {sum(1 for r in results if r['is_correct']) / len(results) * 100:.2f}%")

    # Group results by problem type
    results_by_type = {}
    for result in results:
        if result['problem_type'] not in results_by_type:
            results_by_type[result['problem_type']] = []

        results_by_type[result['problem_type']].append(result)

    for problem_type, counts in results_by_type.items():
        accuracy = sum(1 for r in counts if r['is_correct']) / len(counts) * 100 if len(counts) > 0 else 0
        print(f"{problem_type}={accuracy:.2f}% ({sum(1 for r in counts if r['is_correct'])} correct out of {len(counts)})")