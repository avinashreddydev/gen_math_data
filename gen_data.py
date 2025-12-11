import os
import time
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset
from utils import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR


# This script evaluates a model on a dataset using vLLM server via OpenAI SDK

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
parser.add_argument("--api_key", type=str, default="EMPTY")
parser.add_argument("--tok_limit", type=int, default=32768)
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--max_samples", type=int, default=-1)
parser.add_argument("--num_copies", type=int, default=1)
args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize OpenAI client for vLLM server
client = OpenAI(
    api_key=args.api_key,
    base_url=args.api_base,
)

# Get available model from vLLM server
models = client.models.list()
model = models.data[0].id
print(f"Using model: {model}")

# Initialize tokenizer for token counting
tokenizer = AutoTokenizer.from_pretrained(model)
print(f"Tokenizer loaded: {model}")

dataset_name = args.dataset
tok_limit = args.tok_limit

print("Dataset:", dataset_name)

QUESTION_KEY = DATASET_KEYS[dataset_name]["question"]
ANSWER_KEY = DATASET_KEYS[dataset_name]["answer"]
eq = RESPONSE_COMPARATOR[dataset_name]
extract_answer = RESPONSE_EXTRACTOR[dataset_name]

# Load dataset based on name
if dataset_name == "datasets/converted_aime_dataset":
    dataset = load_from_disk(dataset_name)
    MAX_TEST_SAMPLES = args.max_samples or 100
elif dataset_name == "di-zhang-fdu/MATH500":
    dataset = load_dataset(dataset_name)
    MAX_TEST_SAMPLES = args.max_samples or 500
elif dataset_name == "openai/gsm8k":
    dataset = load_dataset(dataset_name, "main")
    MAX_TEST_SAMPLES = args.max_samples or 1319
elif dataset_name == "di-zhang-fdu/AIME_1983_2024":
    print("Loading AIME_1983_2024 dataset...")
    dataset = load_dataset(dataset_name, split="train")
    MAX_TEST_SAMPLES = args.max_samples
else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def count_tokens(text: str) -> int:
    """Count tokens using the model's tokenizer."""
    if text is None:
        return 0
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def run_inference(problem: str) -> tuple:
    """Run inference on a single problem using OpenAI SDK with vLLM server.

    Returns:
        tuple: (reasoning, solution, num_reasoning_tokens, num_solution_tokens)
    """
    messages = [
        {
            "role": "user",
            "content": f"Please reason step by step, and put your final answer within \\boxed{{}}. Question: {problem}",
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=tok_limit,
        )

        # Get reasoning and content from response
        # Note: reasoning field is available for models that support it (e.g., reasoning models)
        reasoning = getattr(response.choices[0].message, "reasoning", None)
        content = response.choices[0].message.content

        # Count tokens
        num_reasoning_tokens = count_tokens(reasoning)
        num_solution_tokens = count_tokens(content)

        return reasoning, content, num_reasoning_tokens, num_solution_tokens

    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None, 0, 0


def evaluate_and_save(output_file: str, dataset, args):
    """Run evaluation on the dataset and save results as JSONL."""

    # Get test dataset - replicate each question 16 times for diverse solutions

    if MAX_TEST_SAMPLES > 0:
        dataset = dataset.select(range(MAX_TEST_SAMPLES))

    test_ds = []
    for sample in dataset:
        for _ in range(args.num_copies):
            test_ds.append(sample)

    results = []
    correct_count = 0
    total_count = 0

    print(f"Evaluating {len(test_ds)} samples...")
    start_time = time.time()

    # Open JSONL file for incremental saving
    jsonl_file = open(output_file, "w")

    for idx, sample in enumerate(tqdm(test_ds, desc="Evaluating")):
        problem = sample[QUESTION_KEY]
        gold_answer_raw = sample[ANSWER_KEY]
        gold_answer = extract_answer(gold_answer_raw)

        # Run inference
        reasoning, solution, num_reasoning_tokens, num_solution_tokens = run_inference(
            problem
        )

        # Extract answer from solution
        extracted_answer = extract_answer(solution) if solution else None

        # Check correctness
        is_correct = eq(gold_answer, extracted_answer) if extracted_answer else False

        if is_correct:
            correct_count += 1
        total_count += 1

        # Create result record
        result = {
            "idx": idx,
            "problem": problem,
            "reasoning": reasoning,
            "solution": solution,
            "gold_answer": gold_answer,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "num_reasoning_tokens": num_reasoning_tokens,
            "num_solution_tokens": num_solution_tokens,
        }
        results.append(result)

        # Save to JSONL incrementally
        jsonl_file.write(json.dumps(result) + "\n")
        jsonl_file.flush()

    # Close JSONL file
    jsonl_file.close()

    end_time = time.time()

    # Print summary statistics
    accuracy = correct_count / total_count if total_count > 0 else 0
    avg_reasoning_tokens = sum(r["num_reasoning_tokens"] for r in results) / len(
        results
    )
    avg_solution_tokens = sum(r["num_solution_tokens"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"{'='*50}")
    print(f"Total samples: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average reasoning tokens: {avg_reasoning_tokens:.1f}")
    print(f"Average solution tokens: {avg_solution_tokens:.1f}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"{'='*50}")

    # Save results as JSON with indent=4
    json_output_file = output_file.replace(".jsonl", ".json")
    with open(json_output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {output_file} (JSONL)")
    print(f"Results saved to: {json_output_file} (JSON)")

    # Also save summary statistics
    summary = {
        "dataset": dataset_name,
        "model": model,
        "total_samples": total_count,
        "correct": correct_count,
        "accuracy": accuracy,
        "avg_reasoning_tokens": avg_reasoning_tokens,
        "avg_solution_tokens": avg_solution_tokens,
        "time_taken": end_time - start_time,
    }

    summary_file = output_file.replace(".jsonl", "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Summary saved to: {summary_file}")

    return results, summary


# Main execution
if __name__ == "__main__":
    # Determine output file name
    if args.output_file:
        output_file = args.output_file
    else:
        os.makedirs("outputs", exist_ok=True)
        output_file = f"outputs/{dataset_name.replace('/', '_')}_results_{model.replace('/', '_')}_{tok_limit}.jsonl"

    print(f"Model: {model}")
    print(f"Dataset: {dataset_name}")
    print(f"Output file: {output_file}")
    print(f"Max tokens: {tok_limit}")
    print(f"Temperature: {args.temperature}")

    results, summary = evaluate_and_save(output_file, dataset, args)
