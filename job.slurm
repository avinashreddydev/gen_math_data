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


# Start vLLM server in background
echo "Starting vLLM server with model: $MODEL"
vllm serve $MODEL --port 9453 --reasoning-parser deepseek_r1 --dtype bfloat16 &
VLLM_PID=$!

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to start..."
sleep 60

# Check if server is running
until curl -s http://localhost:9453/v1/models > /dev/null 2>&1; do
    echo "Waiting for vLLM server..."
    sleep 10
done
echo "vLLM server is ready!"



# Run the evaluation script
echo "Running gen_data.py..."
python gen_data.py --dataset di-zhang-fdu/AIME_1983_2024 --num_copies 16 --inference_split "[0:10]" 
# Cleanup: kill vLLM server
echo "Shutting down vLLM server..."
kill $VLLM_PID

echo "Job completed!"
