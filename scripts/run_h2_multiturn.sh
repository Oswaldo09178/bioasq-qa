#!/bin/bash
#SBATCH --job-name=h2_multiturn
#SBATCH --output=logs/h2_multiturn_%j.out
#SBATCH --error=logs/h2_multiturn_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1

source .venv/bin/activate
python scripts/test_multiturn.py