#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=single
#SBATCH --gpus-per-node=1
#SBATCH --time=3:30:00
#SBATCH --job-name eval-b


python experiments/eval.py --type "refusal"