#!/bin/bash
#SBATCH --job-name=baseline           # Name of the job
#SBATCH --partition=scavenger       # Partition name
#SBATCH --qos=scavenger             # QOS level
#SBATCH --time=4:00:00              # Max wall time (hh:mm:ss)
#SBATCH --account=scavenger         # Account name
#SBATCH --mem=20G                  # Memory
#SBATCH --output=slurm-%j.out       # Standard output and error log

source LiveCodeBench/.venv/bin/activate

python evaluate_completions_for_n.py 

