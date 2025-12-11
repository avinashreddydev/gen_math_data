# 🧮 Math Reasoning Data Generator

Generate large-scale math reasoning datasets using LLMs served via vLLM. This project evaluates models on mathematical benchmarks (AIME, MATH500, GSM8K) and produces structured datasets with reasoning traces.

## ✨ Features

- **vLLM Integration**: Uses OpenAI-compatible API to interact with vLLM server
- **Multiple Datasets**: Supports AIME (1983-2024), MATH500, GSM8K, and custom datasets
- **Reasoning Extraction**: Captures both reasoning traces and final solutions
- **Automatic Grading**: Evaluates correctness using symbolic math comparison
- **Token Counting**: Accurate token counts using the model's tokenizer
- **Incremental Saving**: Results saved as JSONL during generation (crash-safe)
- **Multiple Solutions**: Generate N solutions per problem for diversity

## 📁 Project Structure

```
├── gen_data.py          # Main data generation script
├── job.slurm            # SLURM job script (UCF Newton cluster)
├── utils/
│   ├── grader.py        # Math equivalence checking
│   ├── parser.py        # Answer extraction utilities
│   └── utils.py         # Dataset configurations
├── outputs/             # Generated datasets
└── logs/                # SLURM job logs
```

## 🚀 Quick Start

### 1. Start vLLM Server

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000 --reasoning-parser deepseek_r1
```

### 2. Run Data Generation

```bash
python gen_data.py --dataset "di-zhang-fdu/AIME_1983_2024" --num_copies 16
```

## ⚙️ Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | *required* | Dataset name (see supported datasets) |
| `--api_base` | `http://localhost:8000/v1` | vLLM server URL |
| `--api_key` | `EMPTY` | API key (usually not needed for local vLLM) |
| `--tok_limit` | `32768` | Maximum tokens for generation |
| `--temperature` | `0.6` | Sampling temperature |
| `--output_file` | Auto-generated | Custom output file path |
| `--max_samples` | `-1` (all) | Limit number of samples |
| `--num_copies` | `1` | Number of solutions per problem |

## 📊 Supported Datasets

| Dataset | Key |
|---------|-----|
| AIME 1983-2024 | `di-zhang-fdu/AIME_1983_2024` |
| MATH500 | `di-zhang-fdu/MATH500` |
| GSM8K | `openai/gsm8k` |
| Custom AIME | `datasets/converted_aime_dataset` |

## 📄 Output Format

Results are saved in both JSONL (incremental) and JSON (formatted) formats:

```json
{
    "idx": 0,
    "problem": "Find the value of x...",
    "reasoning": "<think>Let me analyze...</think>",
    "solution": "The answer is \\boxed{42}",
    "gold_answer": "42",
    "extracted_answer": "42",
    "is_correct": true,
    "num_reasoning_tokens": 1234,
    "num_solution_tokens": 567
}
```

**Output files:**
- `outputs/<dataset>_results_<model>_<tokens>.jsonl` - Line-by-line results
- `outputs/<dataset>_results_<model>_<tokens>.json` - Formatted JSON array
- `outputs/<dataset>_results_<model>_<tokens>_summary.json` - Evaluation summary

## 🖥️ SLURM Usage (UCF Newton Cluster)

Submit the job:

```bash
sbatch job.slurm
```

The SLURM script will:
1. Restore modules
2. Start vLLM server with DeepSeek-R1-Distill-Qwen-7B
3. Wait for server readiness
4. Run evaluation
5. Clean up server process

## 📈 Example Output

```
==================================================
Evaluation Results:
==================================================
Total samples: 900
Correct: 612
Accuracy: 0.6800 (68.00%)
Average reasoning tokens: 2456.3
Average solution tokens: 234.1
Time taken: 3421.56 seconds
==================================================
```

## 📦 Requirements

```
openai
vllm
transformers
datasets
tqdm
sympy
latex2sympy2
word2number
```

## 📝 License

MIT License
