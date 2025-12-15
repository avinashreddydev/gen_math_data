#!/bin/bash
#SBATCH --job-name=aime_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --time=24:00:00

# Restore modules
module restore

# Set model
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


# Change to project directory
cd /home/av131693/aime_data_gen

# Activate virtual environment
source .venv/bin/activate




# Run the evaluation script
echo "Running gen_data_offline.py..."
python gen_data_offline.py --dataset di-zhang-fdu/AIME_1983_2024 --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 
# Cleanup: kill vLLM server
echo "Shutting down vLLM server..."
kill $VLLM_PID

echo "Job completed!"
