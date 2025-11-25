#!/bin/bash
#SBATCH --job-name=baseline           # Name of the job
#SBATCH --partition=scavenger       # Partition name
#SBATCH --qos=scavenger             # QOS level
#SBATCH --gres=gpu:rtxa6000:2      # GPU request
#SBATCH --time=4:00:00              # Max wall time (hh:mm:ss)
#SBATCH --account=scavenger         # Account name
#SBATCH --mem=20G                  # Memory
#SBATCH --output=slurm-%j.out       # Standard output and error log

source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 uv run python search-and-learn/scripts/test_time_compute.py search-and-learn/recipes/MATH-500/Llama-3.2-3B-Instruct/best_of_n/qwen_math_prm.yaml
