#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=medconvo_retrieval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/retrieval_%j.out
#SBATCH --error=logs/retrieval_%j.err

source ~/bioasq-qa/venv/bin/activate
export HF_HOME=/data/hf_cache

cd ~/bioasq-qa

python run_pipeline.py
