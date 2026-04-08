#!/bin/bash
#SBATCH --job-name=medgemma_test
#SBATCH --output=logs/medgemma_test_%j.out
#SBATCH --error=logs/medgemma_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1

source .venv/bin/activate
export HF_TOKEN=hf_your_token_here
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python src/rag_system.py \
    --limit 10 \
    --retriever hybrid \
    --generator medgemma